import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Starting train_chdt.py with project root: {PROJECT_ROOT}", flush=True)

import hydra
import numpy as np
import torch
print(f"Imported torch {torch.__version__}", flush=True)
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

from models.CHDT.utils import AlphaRise, FilteringMlFlowLogger
from models.CHDT.counterfactual_trajectory_exporter import save_counterfactual_trajectory_csvs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


class TrainingErrorTracker(Callback):
    def __init__(self, metric_name: str = "chdt_train_loss"):
        self.metric_name = metric_name
        self.train_errors = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        train_error = trainer.callback_metrics.get(self.metric_name)
        if train_error is None:
            return
        if hasattr(train_error, "detach"):
            train_error = train_error.detach().cpu().item()
        self.train_errors.append(float(train_error))


class WarningLogCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


def _to_report_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().item()
    return value


def _cfg_get(cfg, key, default="N/A"):
    value = OmegaConf.select(cfg, key, default=default)
    return _to_report_value(value)


def _dataset_stats(dataset):
    return getattr(dataset, "dataset_statistics", {}) or {}


def _stats_get(stats, key, default="N/A"):
    return _to_report_value(stats.get(key, default))


def _shape_dict(data):
    if not isinstance(data, dict):
        return "N/A"
    shapes = {}
    for key, value in data.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[key] = tuple(shape)
    return shapes if shapes else "N/A"


def _infer_elapsed_time_dimension(train_data, use_time_gap: bool):
    if not use_time_gap:
        return 0
    if "time_gap" not in train_data:
        raise ValueError('model.chdt.use_time_gap is true but train data is missing "time_gap".')
    time_gap = train_data["time_gap"]
    return 1 if getattr(time_gap, "ndim", 0) == 2 else time_gap.shape[-1]


def _effective_chdt_output_size(args: DictConfig):
    outcome_type = str(OmegaConf.select(args, "model.chdt.outcome_type", default="continuous")).lower()
    if outcome_type == "categorical":
        outcome_output_size = OmegaConf.select(args, "model.chdt.num_outcome_classes", default=None)
        if outcome_output_size is None:
            class_values = OmegaConf.select(args, "model.chdt.outcome_class_values", default=None)
            if class_values is None:
                raise ValueError(
                    "model.chdt.num_outcome_classes or model.chdt.outcome_class_values "
                    "must be set when outcome_type is categorical."
                )
            outcome_output_size = len(class_values)
    elif outcome_type == "continuous":
        outcome_output_size = args.model.dim_outcomes
    else:
        raise ValueError(f"Unsupported model.chdt.outcome_type: {outcome_type}")

    return int(outcome_output_size) + int(args.model.dim_vitals)


def _format_block(value, indent=""):
    value = _to_report_value(value)
    if value == "N/A":
        return f"{indent}N/A"
    if isinstance(value, dict):
        return "\n".join(f"{indent}{key}: {_to_report_value(val)}" for key, val in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(f"{indent}- {_to_report_value(item)}" for item in value)
    return f"{indent}{value}"


def _metric(metrics, key):
    return _to_report_value(metrics.get(key, "N/A"))


def _std_metric(metrics, key):
    return _metric(metrics, f"{key}_std")


def _format_metric_with_std(metrics, key):
    value = _metric(metrics, key)
    std_value = _std_metric(metrics, key)
    if std_value == "N/A":
        return value
    return f"{value} (std: {std_value})"


def _arm_metric(metrics, arm_name, suffix):
    return _metric(metrics, f"encoder_{arm_name}_cf_rmse_{suffix}")


def _decoder_arm_metric(metrics, arm_name, step):
    return _metric(metrics, f"decoder_{arm_name}_cf_rmse_{step}")


def _step_sort_key(step_name: str):
    try:
        return int(step_name.split("-")[0])
    except (TypeError, ValueError):
        return step_name


def _format_horizon_metrics(metrics, prefix, indent="  "):
    step_values = []
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        step_name = key[len(prefix):]
        if not step_name.endswith("-step"):
            continue
        step_values.append((step_name, value))

    if not step_values:
        return f"{indent}N/A"

    return "\n".join(
        f"{indent}{step_name}: {_format_metric_with_std(metrics, prefix + step_name)}"
        for step_name, value in sorted(step_values, key=lambda item: _step_sort_key(item[0]))
    )


OUTCOME_CLASSIFICATION_SCALAR_METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "support",
)
OUTCOME_CLASSIFICATION_DETAIL_METRIC_NAMES = (
    "labels",
    "confusion_matrix",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "per_class_support",
)
OUTCOME_CLASSIFICATION_METRIC_NAMES = (
    OUTCOME_CLASSIFICATION_SCALAR_METRIC_NAMES + OUTCOME_CLASSIFICATION_DETAIL_METRIC_NAMES
)


def _compute_outcome_classification_metrics_enabled(args: DictConfig) -> bool:
    return bool(OmegaConf.select(args, "model.chdt.compute_outcome_classification_metrics", default=False))


def _add_outcome_classification_results(results: dict, base_key: str, metrics: dict):
    for metric_name in OUTCOME_CLASSIFICATION_METRIC_NAMES:
        results[f"{base_key}_{metric_name}"] = _to_report_value(metrics.get(metric_name, np.nan))


def _add_n_step_outcome_classification_results(results: dict, base_prefix: str, metrics_by_step):
    for step_idx, metrics in enumerate(metrics_by_step):
        _add_outcome_classification_results(results, f"{base_prefix}_{step_idx + 2}-step", metrics)


def _log_metrics(mlf_logger, args: DictConfig, metrics: dict):
    if not args.exp.logging:
        return
    scalar_metrics = {}
    for key, value in metrics.items():
        value = _to_report_value(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scalar_metrics[key] = value
    mlf_logger.log_metrics(scalar_metrics)


def _format_confusion_matrix(metrics, base_key, indent="  "):
    matrix = _to_report_value(metrics.get(f"{base_key}_confusion_matrix", []))
    if matrix == "N/A" or not matrix:
        return f"{indent}N/A"
    return "\n".join(f"{indent}{row}" for row in matrix)


def _format_per_class_metrics(metrics, base_key, indent="  "):
    labels = _to_report_value(metrics.get(f"{base_key}_labels", []))
    precision = _to_report_value(metrics.get(f"{base_key}_per_class_precision", []))
    recall = _to_report_value(metrics.get(f"{base_key}_per_class_recall", []))
    f1 = _to_report_value(metrics.get(f"{base_key}_per_class_f1", []))
    support = _to_report_value(metrics.get(f"{base_key}_per_class_support", []))
    if labels == "N/A" or not labels:
        return f"{indent}N/A"

    lines = []
    for idx, label in enumerate(labels):
        lines.append(
            f"{indent}Class {label}: precision={_to_report_value(precision[idx])}, "
            f"recall={_to_report_value(recall[idx])}, "
            f"F1={_to_report_value(f1[idx])}, "
            f"support={_to_report_value(support[idx])}"
        )
    return "\n".join(lines)


def _format_outcome_classification_metrics(metrics, base_key, indent="  "):
    if f"{base_key}_support" not in metrics:
        return f"{indent}N/A"
    return "\n".join([
        f"{indent}Accuracy: {_metric(metrics, base_key + '_accuracy')}",
        f"{indent}Macro precision: {_metric(metrics, base_key + '_macro_precision')}",
        f"{indent}Macro recall: {_metric(metrics, base_key + '_macro_recall')}",
        f"{indent}Macro F1: {_metric(metrics, base_key + '_macro_f1')}",
        f"{indent}Weighted precision: {_metric(metrics, base_key + '_weighted_precision')}",
        f"{indent}Weighted recall: {_metric(metrics, base_key + '_weighted_recall')}",
        f"{indent}Weighted F1: {_metric(metrics, base_key + '_weighted_f1')}",
        f"{indent}Support: {_metric(metrics, base_key + '_support')}",
        f"{indent}Labels: {_metric(metrics, base_key + '_labels')}",
        f"{indent}Confusion matrix (rows=true, columns=predicted):",
        _format_confusion_matrix(metrics, base_key, indent=f"{indent}  "),
        f"{indent}Per-class metrics:",
        _format_per_class_metrics(metrics, base_key, indent=f"{indent}  "),
    ])


def _format_outcome_classification_horizon_metrics(metrics, prefix, indent="  "):
    step_values = []
    suffix = "_support"
    for key in metrics:
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        step_name = key[len(prefix):-len(suffix)]
        if not step_name.endswith("-step"):
            continue
        step_values.append(step_name)

    if not step_values:
        return f"{indent}N/A"

    lines = []
    for step_name in sorted(step_values, key=_step_sort_key):
        lines.append(f"{indent}{step_name}:")
        lines.append(_format_outcome_classification_metrics(metrics, prefix + step_name, indent=f"{indent}  "))
    return "\n".join(lines)


def _format_horizon_active_counts(dataset, indent="  "):
    data_processed_seq = getattr(dataset, "data_processed_seq", None)
    if not isinstance(data_processed_seq, dict) or "active_entries" not in data_processed_seq:
        return f"{indent}N/A"

    active_entries = data_processed_seq["active_entries"]
    if not hasattr(active_entries, "sum") or active_entries.ndim < 2:
        return f"{indent}N/A"

    counts = active_entries.sum(axis=tuple(axis for axis in range(active_entries.ndim) if axis != 1))
    lines = []
    for idx, count in enumerate(counts):
        count_value = float(count)
        if count_value.is_integer():
            count_value = int(count_value)
        lines.append(f"{indent}{idx + 2}-step: {count_value}")
    return "\n".join(lines) if lines else f"{indent}N/A"


def _module_param_count(model, module_name):
    module = getattr(model, module_name, None)
    if module is None:
        return "N/A"
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _residual_holdout_summary(model):
    holdout_resid = getattr(model, "holdout_resid", None)
    holdout_resid_len = getattr(model, "holdout_resid_len", None)
    if holdout_resid is None:
        return "N/A"
    resid_shape = getattr(holdout_resid, "shape", None)
    len_shape = getattr(holdout_resid_len, "shape", None)
    return {
        "holdout_resid_shape": tuple(resid_shape) if resid_shape is not None else "N/A",
        "holdout_resid_len_shape": tuple(len_shape) if len_shape is not None else "N/A",
    }


def _format_tuned_hparams(model, indent="  "):
    best_config = getattr(model, "best_tuning_config", None)
    best_metrics = getattr(model, "best_tuning_metrics", None)
    if best_config is None:
        return f"{indent}N/A"

    lines = []
    if isinstance(best_metrics, dict):
        for key, value in best_metrics.items():
            lines.append(f"{indent}{key}: {_to_report_value(value)}")

    lines.append(f"{indent}best_configuration:")
    best_config_yaml = OmegaConf.to_yaml(OmegaConf.create(best_config), resolve=True).rstrip()
    lines.extend(f"{indent}  {line}" for line in best_config_yaml.splitlines())
    return "\n".join(lines)


def _chdt_report_stem(args: DictConfig, timestamp: str) -> str:
    return (
        f"chdt_report_seed_{_cfg_get(args, 'exp.seed')}_"
        f"epochs_{_cfg_get(args, 'exp.max_epochs')}_{timestamp}"
    )


def _format_duration(duration_seconds):
    if duration_seconds == "N/A":
        return "N/A"
    minutes, seconds = divmod(float(duration_seconds), 60.0)
    return f"{float(duration_seconds):.2f} seconds ({int(minutes)}m {seconds:.2f}s)"


def write_chdt_evaluation_report(
    args: DictConfig,
    dataset_collection,
    model,
    trainer,
    training_error_tracker: TrainingErrorTracker,
    training_loss_plot_path: Path,
    training_time_seconds,
    total_running_time_seconds,
    report_timestamp: str,
    results: dict,
    warning_messages: list,
) -> Path:
    report_dir = Path(__file__).resolve().parent / "chdt_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{_chdt_report_stem(args, report_timestamp)}.txt"

    train_stats = _dataset_stats(getattr(dataset_collection, "train_f", None))
    val_stats = _dataset_stats(getattr(dataset_collection, "val_f", None))
    test_stats = _dataset_stats(getattr(dataset_collection, "test_f", None))
    cf_one_step = getattr(dataset_collection, "test_cf_one_step_by_arm", {}) or {}
    placebo_stats = _dataset_stats(cf_one_step.get("placebo"))
    statin_stats = _dataset_stats(cf_one_step.get("statin"))

    device = getattr(trainer, "root_device", None) or getattr(getattr(trainer, "strategy", None), "root_device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) if hasattr(model, "parameters") else "N/A"
    final_training_loss = training_error_tracker.train_errors[-1] if training_error_tracker.train_errors else "N/A"
    epochs_completed = len(training_error_tracker.train_errors) or getattr(trainer, "current_epoch", "N/A")
    training_time = _format_duration(training_time_seconds)
    total_running_time = _format_duration(total_running_time_seconds)

    tensor_shape_sections = {
        "train": _shape_dict(getattr(getattr(dataset_collection, "train_f", None), "data", None)),
        "validation": _shape_dict(getattr(getattr(dataset_collection, "val_f", None), "data", None)),
        "test": _shape_dict(getattr(getattr(dataset_collection, "test_f", None), "data", None)),
        "test_original": _shape_dict(getattr(getattr(dataset_collection, "test_f", None), "data_original", None)),
        "test_sequential": _shape_dict(getattr(getattr(dataset_collection, "test_f", None), "data_processed_seq", None)),
    }
    for arm_name, cf_dataset in cf_one_step.items():
        tensor_shape_sections[f"{arm_name}_counterfactual_one_step"] = _shape_dict(getattr(cf_dataset, "data", None))
    for arm_name, cf_dataset in (getattr(dataset_collection, "test_cf_treatment_seq_by_arm", {}) or {}).items():
        tensor_shape_sections[f"{arm_name}_counterfactual_treatment_seq"] = _shape_dict(getattr(cf_dataset, "data", None))
        tensor_shape_sections[f"{arm_name}_counterfactual_sequential"] = _shape_dict(getattr(cf_dataset, "data_processed_seq", None))

    warnings = warning_messages or ["N/A"]
    compute_outcome_classification_metrics = _compute_outcome_classification_metrics_enabled(args)
    use_time_gap = bool(OmegaConf.select(args, "model.chdt.use_time_gap", default=False))
    use_reverse_component_conditioning = bool(
        OmegaConf.select(args, "model.chdt.use_reverse_component_conditioning", default=False)
    )
    model_display_name = "CHDT"
    if use_reverse_component_conditioning:
        model_display_name = "CHDT" if use_time_gap else "CHDT without elapsed-time input"
    metric_definition_lines = [
        (
            "RMSE all: masked RMSE computed by averaging squared errors over all active "
            "patient-time-output entries at once. Each observed entry has equal weight."
        ),
        (
            "RMSE orig: masked RMSE computed like the original paper: first average squared "
            "errors across patients and outputs at each time index, then average across time "
            "indices. Each time index has equal weight after masking."
        ),
        (
            "Autoregressive n-step RMSE: masked RMSE for future horizon positions produced "
            "by recursively feeding predicted outcomes/vitals back into the model. Reported "
            "2-step, 3-step, ... values correspond to successive future positions in "
            "data_processed_seq and are averaged per horizon using the RMSE orig style."
        ),
        (
            "Reported std values: standard deviation of error magnitudes in the same unit as "
            "the RMSE. For RMSE all this is across active entries; for RMSE orig this is "
            "across per-time RMSE values; for autoregressive n-step RMSE this is across active "
            "entries within each horizon."
        ),
    ]
    if compute_outcome_classification_metrics:
        metric_definition_lines.extend([
            (
                "Outcome classification metrics: predicted scalar outcomes are mapped to the nearest "
                "observed class label and evaluated only over active masked entries."
            ),
            "Precision: the share of predicted positives for a class that are correct.",
            "Recall: the share of true entries for a class that are recovered.",
            "F1: the harmonic mean of precision and recall.",
            (
                "Macro averaging gives each observed class equal weight; weighted averaging weights "
                "each class by its active-entry support."
            ),
            (
                "Confusion matrices use observed class labels as row and column order, with rows "
                "as true labels and columns as predicted labels."
            ),
            "Per-class precision, recall, F1, and support are reported for each observed label.",
        ])

    outcome_classification_report_lines = []
    if compute_outcome_classification_metrics:
        outcome_classification_report_lines = [
            "==================================================",
            "Outcome Classification Metrics",
            "==================================================",
            (
                "Predicted scalar outcomes are mapped to the nearest observed class label and "
                "evaluated only over active masked entries."
            ),
            "",
            "One-step factual:",
            "Validation:",
            _format_outcome_classification_metrics(results, "encoder_val_outcome"),
            "",
            "Test:",
            _format_outcome_classification_metrics(results, "encoder_test_outcome"),
            "",
            "One-step counterfactual:",
            "Placebo counterfactual:",
            _format_outcome_classification_metrics(results, "encoder_placebo_cf_outcome"),
            "",
            "Placebo counterfactual only counterfactual entries:",
            _format_outcome_classification_metrics(results, "encoder_placebo_cf_outcome_last"),
            "",
            "Statin counterfactual:",
            _format_outcome_classification_metrics(results, "encoder_statin_cf_outcome"),
            "",
            "Statin counterfactual only counterfactual entries:",
            _format_outcome_classification_metrics(results, "encoder_statin_cf_outcome_last"),
            "",
            "Autoregressive n-step factual:",
            "Test:",
            _format_outcome_classification_horizon_metrics(results, "decoder_test_outcome_"),
            "",
            "Autoregressive n-step counterfactual:",
            "Placebo counterfactual:",
            _format_outcome_classification_horizon_metrics(results, "decoder_placebo_cf_outcome_"),
            "",
            "Statin counterfactual:",
            _format_outcome_classification_horizon_metrics(results, "decoder_statin_cf_outcome_"),
            "",
        ]

    lines = [
        "==================================================",
        "Experiment Summary",
        "==================================================",
        "Script: train_chdt.py",
        f"Dataset: {_cfg_get(args, 'dataset.name')}",
        f"Model: {model_display_name}",
        f"Seed: {_cfg_get(args, 'exp.seed')}",
        f"Device: {device}",
        f"Max epochs: {_cfg_get(args, 'exp.max_epochs')}",
        f"Batch size: {_cfg_get(args, 'model.chdt.batch_size')}",
        "",
        "==================================================",
        "Dataset Summary",
        "==================================================",
        f"Train patients: {_stats_get(train_stats, 'num_patients_used_for_model')}",
        f"Validation patients: {_stats_get(val_stats, 'num_patients_used_for_model')}",
        f"Test patients: {_stats_get(test_stats, 'num_patients_used_for_model')}",
        f"Placebo CF patients: {_stats_get(placebo_stats, 'num_patients_used_for_model')}",
        f"Statin CF patients: {_stats_get(statin_stats, 'num_patients_used_for_model')}",
        (
            "Visits per patient: "
            f"train mean={_stats_get(train_stats, 'mean_visits_per_patient')}, "
            f"val mean={_stats_get(val_stats, 'mean_visits_per_patient')}, "
            f"test mean={_stats_get(test_stats, 'mean_visits_per_patient')}"
        ),
        (
            "Counterfactual visits per patient: "
            f"placebo mean={_stats_get(placebo_stats, 'mean_visits_per_patient')}, "
            f"statin mean={_stats_get(statin_stats, 'mean_visits_per_patient')}"
        ),
        (
            "Transitions per patient: "
            f"train total={_stats_get(train_stats, 'num_transitions_used_for_model')}, "
            f"val total={_stats_get(val_stats, 'num_transitions_used_for_model')}, "
            f"test total={_stats_get(test_stats, 'num_transitions_used_for_model')}"
        ),
        (
            "Visit gap: "
            f"train mean={_stats_get(train_stats, 'mean_visit_gap')}, "
            f"min={_stats_get(train_stats, 'min_visit_gap')}, "
            f"max={_stats_get(train_stats, 'max_visit_gap')}"
        ),
        "",
        f"Static features: {_stats_get(train_stats, 'num_static_features')}",
        f"Treatment features: {_stats_get(train_stats, 'num_treatment_features')}",
        f"Outcome features: {_stats_get(train_stats, 'num_outcome_features')}",
        f"Vital features: {_stats_get(train_stats, 'num_vital_features')}",
        "",
        "Vital feature columns:",
        _format_block(_stats_get(train_stats, "vital_feature_columns")),
        "",
        "Static feature columns:",
        _format_block(_stats_get(train_stats, "static_feature_columns")),
        "",
        "Treatment columns:",
        _format_block(_stats_get(train_stats, "treatment_feature_columns")),
        "",
        "Outcome columns:",
        _format_block(_stats_get(train_stats, "outcome_columns")),
        "",
        "==================================================",
        "Metric Definitions",
        "==================================================",
        *metric_definition_lines,
        "",
        "==================================================",
        "Processed Tensor Shapes",
        "==================================================",
        "\n\n".join(f"{name}:\n{_format_block(shapes, indent='  ')}" for name, shapes in tensor_shape_sections.items()),
        "",
        "==================================================",
        "Autoregressive Horizon Active Samples",
        "==================================================",
        "Test:",
        _format_horizon_active_counts(getattr(dataset_collection, "test_f", None)),
        "",
        "Placebo counterfactual:",
        _format_horizon_active_counts((getattr(dataset_collection, "test_cf_treatment_seq_by_arm", {}) or {}).get("placebo")),
        "",
        "Statin counterfactual:",
        _format_horizon_active_counts((getattr(dataset_collection, "test_cf_treatment_seq_by_arm", {}) or {}).get("statin")),
        "",
        "==================================================",
        "Model Summary",
        "==================================================",
        f"Input size: {getattr(model, 'input_size', 'N/A')}",
        f"Number of components: {_cfg_get(args, 'model.chdt.num_comp')}",
        f"Component sizes: {_cfg_get(args, 'model.chdt.comp_sizes')}",
        f"Outcome type: {_cfg_get(args, 'model.chdt.outcome_type', 'continuous')}",
        f"Outcome classes: {_cfg_get(args, 'model.chdt.num_outcome_classes')}",
        f"Outcome class values: {_cfg_get(args, 'model.chdt.outcome_class_values')}",
        f"Loss weights [outcome, vitals]: {_cfg_get(args, 'model.chdt.loss_weights', [1.0, 1.0])}",
        f"Use time gap: {_cfg_get(args, 'model.chdt.use_time_gap')}",
        f"Reverse component conditioning: {_cfg_get(args, 'model.chdt.use_reverse_component_conditioning')}",
        f"Fit vitals: {_cfg_get(args, 'model.chdt.fit_vitals')}",
        f"MC samples: {_cfg_get(args, 'model.chdt.mc_samples')}",
        f"LSTM hidden units: {_cfg_get(args, 'model.chdt.seq_hidden_units')}",
        f"Representation size: {_cfg_get(args, 'model.chdt.r_size')}",
        f"FC hidden units: {_cfg_get(args, 'model.chdt.fc_hidden_units')}",
        f"Dropout rate: {_cfg_get(args, 'model.chdt.dropout_rate')}",
        f"Total trainable parameters: {_to_report_value(total_params)}",
        "",
        "Model modules:",
        f"- repr_net: {_module_param_count(model, 'repr_net')}",
        f"- r_outcome_vitals_head: {_module_param_count(model, 'r_outcome_vitals_head')}",
        f"- component_refinement_heads: {_module_param_count(model, 'component_refinement_heads')}",
        "",
        "Best tuned hyperparameters:",
        _format_tuned_hparams(model),
        "",
        "==================================================",
        "Training Summary",
        "==================================================",
        f"Epochs completed: {_to_report_value(epochs_completed)}",
        f"Training time: {training_time}",
        f"Total running time: {total_running_time}",
        f"Final training loss: {_to_report_value(final_training_loss)}",
        f"Train residual holdout: {_format_block(_residual_holdout_summary(model))}",
        f"Training loss plot: {training_loss_plot_path}",
        "",
        "==================================================",
        "One-step RMSE",
        "==================================================",
        f"Validation RMSE all: {_format_metric_with_std(results, 'encoder_val_rmse_all')}",
        f"Validation RMSE orig: {_format_metric_with_std(results, 'encoder_val_rmse_orig')}",
        "",
        f"Test RMSE all: {_format_metric_with_std(results, 'encoder_test_rmse_all')}",
        f"Test RMSE orig: {_format_metric_with_std(results, 'encoder_test_rmse_orig')}",
        "",
        f"Placebo counterfactual RMSE all: {_format_metric_with_std(results, 'encoder_placebo_cf_rmse_all')}",
        f"Placebo counterfactual RMSE orig: {_format_metric_with_std(results, 'encoder_placebo_cf_rmse_orig')}",
        f"Placebo counterfactual RMSE only counterfactual: {_format_metric_with_std(results, 'encoder_placebo_cf_rmse_last')}",
        "",
        f"Statin counterfactual RMSE all: {_format_metric_with_std(results, 'encoder_statin_cf_rmse_all')}",
        f"Statin counterfactual RMSE orig: {_format_metric_with_std(results, 'encoder_statin_cf_rmse_orig')}",
        f"Statin counterfactual RMSE only counterfactual: {_format_metric_with_std(results, 'encoder_statin_cf_rmse_last')}",
        "",
        "==================================================",
        "Autoregressive n-step RMSE",
        "==================================================",
        "Test:",
        _format_horizon_metrics(results, "decoder_test_rmse_"),
        "",
        "Placebo counterfactual:",
        _format_horizon_metrics(results, "decoder_placebo_cf_rmse_"),
        "",
        "Statin counterfactual:",
        _format_horizon_metrics(results, "decoder_statin_cf_rmse_"),
        "",
        *outcome_classification_report_lines,
        "==================================================",
        "Warnings / Notes",
        "==================================================",
        _format_block(warnings),
        "",
    ]

    report_path.write_text("\n".join(str(line) for line in lines), encoding="utf-8")
    logger.info(f"Structured experiment report saved to: {report_path}")
    print(f"Structured experiment report saved to: {report_path}", flush=True)
    return report_path


def lightning_gpus_arg(gpus):
    gpus = eval(str(gpus))
    if not torch.cuda.is_available():
        return 0
    return gpus


def plot_training_error(train_errors, chdt_config: DictConfig = None, output_dir: Path = None, output_path: Path = None) -> Path:
    """
    Draw and save training error per epoch.
    """
    if output_path is None:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "chdt_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        if chdt_config is None:
            output_filename = "train_error.jpg"
        else:
            output_filename = (
                f"train_loss_chdt_{chdt_config.num_layer}_"
                f"{chdt_config.seq_hidden_units}_{chdt_config.r_size}.jpg"
            )
        output_path = output_dir / output_filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_errors:
        logger.warning("No training errors were collected; skipping training error plot.")
        return output_path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(train_errors) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_errors, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("CHDT Training Error per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="jpg", dpi=200)
    plt.close()

    logger.info(f"Training error plot saved to {output_path}")
    return output_path


def write_tuned_model_report(args: DictConfig, model, config_name: str = "config_alpaca_chdt_tuned.yaml") -> Path:
    """
    Print and save the best tuning result selected by validation RMSE.
    """
    report_dir = Path(__file__).resolve().parent / "chdt_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{Path(config_name).stem}_tuned_model.txt"

    best_config = getattr(model, "best_tuning_config", None)
    best_metrics = getattr(model, "best_tuning_metrics", None)
    if best_config is None or best_metrics is None:
        logger.warning("No tuning result found on the model; skipping tuned model report.")
        return report_path

    report = "\n".join([
        "Best tuned CHDT model",
        f"dataset: {args.dataset.name}",
        f"selection_metric: val_rmse_all",
        f"validation_rmse: {best_metrics.get('val_rmse_all')}",
        "best_configuration:",
        OmegaConf.to_yaml(OmegaConf.create(best_config), resolve=True).rstrip(),
        "",
    ])

    print(report)
    logger.info("\n" + report)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Tuned model report saved to {report_path}")
    return report_path

@hydra.main(config_name="config_alpaca_chdt_tuned.yaml", config_path=".")

def main(args: DictConfig):
    """
    Training and evaluation script for CHDT.
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and multiple-step-ahead RMSEs)
    """

    run_start_time = time.perf_counter()
    results = {}
    warning_collector = WarningLogCollector()
    warning_collector.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(warning_collector)

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data
    seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
    args.model.dim_time_gap = _infer_elapsed_time_dimension(
        dataset_collection.train_f.data,
        bool(OmegaConf.select(args, "model.chdt.use_time_gap", default=False)),
    )

    # Conditional networks outputs are of uniform dim between model outputs.
    # Categorical outcomes use one logit per class, while continuous outcomes use dim_outcomes.
    total_chdt_output_size = _effective_chdt_output_size(args)
    if total_chdt_output_size % args.model.chdt.num_comp != 0:
        raise ValueError(
            "model.chdt.num_comp must divide the effective CHDT output size "
            f"({total_chdt_output_size}), got {args.model.chdt.num_comp}."
        )
    args.model.chdt.comp_sizes = \
        [total_chdt_output_size // args.model.chdt.num_comp] * args.model.chdt.num_comp

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        model_callbacks = [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None
        model_callbacks = []
    training_error_tracker = TrainingErrorTracker()
    model_callbacks.append(training_error_tracker)

    # ============================== Initialisation & Training of CHDT ==============================
    model = instantiate(args.model.chdt, args, dataset_collection, _recursive_=False)
    if args.model.chdt.tune_hparams:
        model.finetune(resources_per_trial=args.model.chdt.resources_per_trial)
        write_tuned_model_report(args, model)

    trainer = Trainer(gpus=lightning_gpus_arg(args.exp.gpus), logger=mlf_logger, max_epochs=args.exp.max_epochs, callbacks=model_callbacks,
                      terminate_on_nan=True)
    training_start_time = time.perf_counter()
    trainer.fit(model)
    training_time_seconds = time.perf_counter() - training_start_time
    report_dir = Path(__file__).resolve().parent / "chdt_reports"
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = _chdt_report_stem(args, report_timestamp)
    training_loss_plot_path = plot_training_error(
        training_error_tracker.train_errors,
        args.model.chdt,
        output_path=report_dir / f"{report_stem}_training_plot.jpg",
    )
    compute_outcome_classification_metrics = _compute_outcome_classification_metrics_enabled(args)

    # Validation factual rmse
    val_rmse_orig, val_rmse_all, val_rmse_stds = model.get_normalised_masked_rmse(
        dataset_collection.val_f,
        include_std=True,
    )
    logger.info(
        f'Val normalised RMSE (all): {val_rmse_all} std: {val_rmse_stds["all"]}; '
        f'Val normalised RMSE (orig): {val_rmse_orig} std: {val_rmse_stds["orig"]}'
    )

    encoder_results = {
        'encoder_val_rmse_all': val_rmse_all,
        'encoder_val_rmse_all_std': val_rmse_stds['all'],
        'encoder_val_rmse_orig': val_rmse_orig,
        'encoder_val_rmse_orig_std': val_rmse_stds['orig'],
    }
    if compute_outcome_classification_metrics:
        _add_outcome_classification_results(
            encoder_results,
            'encoder_val_outcome',
            model.get_masked_outcome_classification_metrics(dataset_collection.val_f),
        )

    if hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all, test_rmse_stds = model.get_normalised_masked_rmse(
            dataset_collection.test_f,
            include_std=True,
        )
        logger.info(
            f'Test normalised RMSE (all): {test_rmse_all} std: {test_rmse_stds["all"]}; '
            f'Test normalised RMSE (orig): {test_rmse_orig} std: {test_rmse_stds["orig"]}.'
        )
        encoder_results.update({
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_all_std': test_rmse_stds['all'],
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_orig_std': test_rmse_stds['orig'],
        })
        if compute_outcome_classification_metrics:
            _add_outcome_classification_results(
                encoder_results,
                'encoder_test_outcome',
                model.get_masked_outcome_classification_metrics(dataset_collection.test_f),
            )

    if hasattr(dataset_collection, 'test_cf_one_step_by_arm'):
        for arm_name, cf_dataset in dataset_collection.test_cf_one_step_by_arm.items():
            test_rmse_orig, test_rmse_all, test_rmse_last, test_rmse_stds = model.get_normalised_masked_rmse(
                cf_dataset,
                one_step_counterfactual=True,
                include_std=True,
            )
            logger.info(
                f'{arm_name} counterfactual normalised RMSE (all): {test_rmse_all} std: {test_rmse_stds["all"]}; '
                f'(orig): {test_rmse_orig} std: {test_rmse_stds["orig"]}; '
                f'(only counterfactual): {test_rmse_last} std: {test_rmse_stds["last"]}'
            )
            encoder_results.update({
                f'encoder_{arm_name}_cf_rmse_all': test_rmse_all,
                f'encoder_{arm_name}_cf_rmse_all_std': test_rmse_stds['all'],
                f'encoder_{arm_name}_cf_rmse_orig': test_rmse_orig,
                f'encoder_{arm_name}_cf_rmse_orig_std': test_rmse_stds['orig'],
                f'encoder_{arm_name}_cf_rmse_last': test_rmse_last,
                f'encoder_{arm_name}_cf_rmse_last_std': test_rmse_stds['last'],
            })
            if compute_outcome_classification_metrics:
                _add_outcome_classification_results(
                    encoder_results,
                    f'encoder_{arm_name}_cf_outcome',
                    model.get_masked_outcome_classification_metrics(cf_dataset),
                )
                _add_outcome_classification_results(
                    encoder_results,
                    f'encoder_{arm_name}_cf_outcome_last',
                    model.get_masked_outcome_classification_metrics(
                        cf_dataset,
                        one_step_counterfactual=True,
                    ),
                )
    elif hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last, test_rmse_stds = model.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step,
            one_step_counterfactual=True,
            include_std=True,
        )
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results.update({
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_all_std': test_rmse_stds['all'],
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_orig_std': test_rmse_stds['orig'],
            'encoder_test_rmse_last': test_rmse_last,
            'encoder_test_rmse_last_std': test_rmse_stds['last'],
        })
        if compute_outcome_classification_metrics:
            _add_outcome_classification_results(
                encoder_results,
                'encoder_test_outcome',
                model.get_masked_outcome_classification_metrics(dataset_collection.test_cf_one_step),
            )
            _add_outcome_classification_results(
                encoder_results,
                'encoder_test_outcome_last',
                model.get_masked_outcome_classification_metrics(
                    dataset_collection.test_cf_one_step,
                    one_step_counterfactual=True,
                ),
            )

    _log_metrics(mlf_logger, args, encoder_results)
    results.update(encoder_results)

    test_rmses = {}
    test_rmse_stds = {}
    test_outcome_classification_metrics = []
    if hasattr(dataset_collection, 'test_cf_treatment_seq_mc'):  # Test n_step_counterfactual rmse
        test_rmses, test_rmse_stds = model.get_normalised_n_step_rmses(
            dataset_collection.test_cf_treatment_seq,
            dataset_collection.test_cf_treatment_seq_mc,
            include_std=True,
        )
        if compute_outcome_classification_metrics:
            test_outcome_classification_metrics = model.get_n_step_outcome_classification_metrics(
                dataset_collection.test_cf_treatment_seq,
                dataset_collection.test_cf_treatment_seq_mc,
            )
    elif hasattr(dataset_collection, 'test_f_mc'):  # Test n_step_factual rmse
        test_rmses, test_rmse_stds = model.get_normalised_n_step_rmses(
            dataset_collection.test_f,
            dataset_collection.test_f_mc,
            include_std=True,
        )
        if compute_outcome_classification_metrics:
            test_outcome_classification_metrics = model.get_n_step_outcome_classification_metrics(
                dataset_collection.test_f,
                dataset_collection.test_f_mc,
            )

    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}
    test_rmse_stds = {f'{k+2}-step': v for (k, v) in enumerate(test_rmse_stds)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_all_std': val_rmse_stds['all'],
        'decoder_val_rmse_orig': val_rmse_orig,
        'decoder_val_rmse_orig_std': val_rmse_stds['orig'],
    }
    decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})
    decoder_results.update({('decoder_test_rmse_' + k + '_std'): v for (k, v) in test_rmse_stds.items()})
    if compute_outcome_classification_metrics:
        _add_n_step_outcome_classification_results(
            decoder_results,
            'decoder_test_outcome',
            test_outcome_classification_metrics,
        )

    if hasattr(dataset_collection, 'test_cf_treatment_seq_by_arm'):
        for arm_name, cf_dataset in dataset_collection.test_cf_treatment_seq_by_arm.items():
            cf_datasets_mc = getattr(dataset_collection, 'test_cf_treatment_seq_mc_by_arm', {}).get(arm_name)
            cf_rmses, cf_rmse_stds = model.get_normalised_n_step_rmses(
                cf_dataset,
                cf_datasets_mc,
                include_std=True,
            )
            cf_rmses = {f'{k+2}-step': v for (k, v) in enumerate(cf_rmses)}
            cf_rmse_stds = {f'{k+2}-step': v for (k, v) in enumerate(cf_rmse_stds)}
            logger.info(f'{arm_name} counterfactual normalised RMSE (n-step prediction): {cf_rmses}')
            decoder_results.update({
                f'decoder_{arm_name}_cf_rmse_{k}': v for (k, v) in cf_rmses.items()
            })
            decoder_results.update({
                f'decoder_{arm_name}_cf_rmse_{k}_std': v for (k, v) in cf_rmse_stds.items()
            })
            if compute_outcome_classification_metrics:
                _add_n_step_outcome_classification_results(
                    decoder_results,
                    f'decoder_{arm_name}_cf_outcome',
                    model.get_n_step_outcome_classification_metrics(cf_dataset, cf_datasets_mc),
                )

    _log_metrics(mlf_logger, args, decoder_results)
    results.update(decoder_results)
    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    if bool(OmegaConf.select(args, "exp.save_generative_horizons", default=False)):
        generative_output_dir = OmegaConf.select(args, "exp.generative_horizons_dir", default=None)
        if generative_output_dir:
            generative_output_dir = Path(hydra.utils.to_absolute_path(str(generative_output_dir)))
        else:
            generative_output_dir = report_dir

        generative_paths = save_counterfactual_trajectory_csvs(
            model=model,
            dataset_collection=dataset_collection,
            output_dir=generative_output_dir,
            report_stem=report_stem,
        )
        for arm_name, output_path in generative_paths.items():
            logger.info("%s generative horizon CSV: %s", arm_name, output_path)
            print(f"{arm_name} generative horizon CSV: {output_path}", flush=True)

    total_running_time_seconds = time.perf_counter() - run_start_time
    report_path = write_chdt_evaluation_report(
        args=args,
        dataset_collection=dataset_collection,
        model=model,
        trainer=trainer,
        training_error_tracker=training_error_tracker,
        training_loss_plot_path=training_loss_plot_path,
        training_time_seconds=training_time_seconds,
        total_running_time_seconds=total_running_time_seconds,
        report_timestamp=report_timestamp,
        results=results,
        warning_messages=warning_collector.messages,
    )
    print(
        f"Total running time: {_format_duration(total_running_time_seconds)} "
        f"(reported in {report_path})",
        flush=True,
    )
    logging.getLogger().removeHandler(warning_collector)

    return results


if __name__ == "__main__":
    main()
