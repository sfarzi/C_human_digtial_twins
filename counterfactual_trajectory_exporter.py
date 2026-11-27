import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

DEFAULT_GENERATIVE_ARMS = ("placebo", "statin")


def save_counterfactual_trajectory_csvs(
    model,
    dataset_collection,
    output_dir,
    report_stem: Optional[str] = None,
    arms: Sequence[str] = DEFAULT_GENERATIVE_ARMS,
) -> Dict[str, Path]:
    exporter = CounterfactualTrajectoryExporter(
        model=model,
        dataset_collection=dataset_collection,
        output_dir=output_dir,
        report_stem=report_stem,
        arms=arms,
    )
    return exporter.save()


class CounterfactualTrajectoryExporter:
    """
    Export CHDT autoregressive trajectory states in the same CSV frame as the
    original placebo/statin arm inputs.
    """

    def __init__(
        self,
        model,
        dataset_collection,
        output_dir,
        report_stem: Optional[str] = None,
        arms: Sequence[str] = DEFAULT_GENERATIVE_ARMS,
    ):
        self.model = model
        self.dataset_collection = dataset_collection
        self.output_dir = Path(output_dir)
        self.report_stem = report_stem
        self.arms = tuple(arms)

    def save(self) -> Dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = {}

        for arm_name in self.arms:
            generated_frame = self.build_intervention_trajectory_frame(arm_name)
            output_path = self.output_dir / self._output_filename(arm_name)
            generated_frame.to_csv(output_path, index=False)
            generated_paths[arm_name] = output_path
            logger.info("Saved %s generative horizon CSV to %s", arm_name, output_path)

        return generated_paths

    def build_intervention_trajectory_frame(self, arm_name: str) -> pd.DataFrame:
        cf_dataset = self._arm_dataset(arm_name)
        template_path = self._arm_template_path(arm_name)
        template_frame = pd.read_csv(template_path)
        generated_frame = template_frame.copy()

        predictions_scaled = self._rollout_scaled_clinical_states(cf_dataset)
        outcomes_unscaled, vitals_unscaled = self._unscale_clinical_states(cf_dataset, predictions_scaled)
        outcomes_unscaled = self._snap_categorical_clinical_outcomes(outcomes_unscaled, cf_dataset)

        self._apply_generated_states_to_frame(
            generated_frame,
            cf_dataset,
            outcomes_unscaled,
            vitals_unscaled,
        )
        return generated_frame

    def _output_filename(self, arm_name: str) -> str:
        suffix = f"{arm_name}_generated_horizons.csv"
        if self.report_stem:
            return f"{self.report_stem}_{suffix}"
        return suffix

    def _arm_dataset(self, arm_name: str):
        arm_datasets = getattr(self.dataset_collection, "test_cf_treatment_seq_by_arm", None)
        if not arm_datasets or arm_name not in arm_datasets:
            raise ValueError(f"Missing counterfactual treatment-sequence dataset for arm {arm_name!r}.")
        return arm_datasets[arm_name]

    def _arm_template_path(self, arm_name: str) -> Path:
        path = getattr(self.dataset_collection, f"{arm_name}_path", None)
        if path is None:
            raise ValueError(f"Dataset collection has no {arm_name}_path for generative export.")
        return Path(path)

    def _rollout_scaled_clinical_states(self, cf_dataset) -> np.ndarray:
        if not hasattr(cf_dataset, "data_processed_seq") or cf_dataset.data_processed_seq is None:
            raise ValueError(f"{cf_dataset.subset_name} is missing data_processed_seq; process it before export.")

        mc_samples = int(getattr(self.model, "mc_samples", 1))
        rollout_datasets = self._create_monte_carlo_rollouts(cf_dataset, mc_samples)
        horizon = int(getattr(self.dataset_collection, "projection_horizon"))
        dim_outcome = int(getattr(self.model, "dim_outcome", len(getattr(cf_dataset, "outcome_cols", []))))
        dim_vitals = len(getattr(cf_dataset, "vital_cols", []))
        full_dim = dim_outcome + dim_vitals

        predicted = np.zeros((mc_samples, len(rollout_datasets[0]), horizon, full_dim), dtype=float)

        for step in range(horizon + 1):
            for sample_idx, rollout_dataset in enumerate(rollout_datasets):
                outputs_next_vitals_scaled = np.asarray(
                    self.model.get_predictions(rollout_dataset, vitals=True),
                    dtype=float,
                )
                split = rollout_dataset.data["future_past_split"].astype(int)
                rows = np.arange(len(rollout_dataset))
                source_time = split - 1 + step

                if step > 0:
                    predicted[sample_idx, :, step - 1, :] = outputs_next_vitals_scaled[
                        rows,
                        source_time,
                        :full_dim,
                    ]

                self._sample_holdout_residuals(outputs_next_vitals_scaled, rows, source_time)

                if step < horizon:
                    target_time = split + step
                    rollout_dataset.data["prev_outputs"][rows, target_time, :] = outputs_next_vitals_scaled[
                        rows,
                        source_time,
                        :dim_outcome,
                    ]
                    if dim_vitals > 0 and "vitals" in rollout_dataset.data:
                        rollout_dataset.data["vitals"][rows, target_time, :] = outputs_next_vitals_scaled[
                            rows,
                            source_time,
                            dim_outcome:full_dim,
                        ]

        return predicted.mean(axis=0)

    def _create_monte_carlo_rollouts(self, cf_dataset, mc_samples: int):
        rollout_datasets = []
        for _ in range(mc_samples):
            dataset_copy = deepcopy(cf_dataset)
            dataset_copy.data = deepcopy(cf_dataset.data)
            dataset_copy.data_processed_seq = deepcopy(cf_dataset.data_processed_seq)
            dataset_copy.data_original = deepcopy(getattr(cf_dataset, "data_original", None))
            rollout_datasets.append(dataset_copy)
        return rollout_datasets

    def _sample_holdout_residuals(self, outputs_next_vitals_scaled, rows, source_time) -> None:
        holdout_resid = getattr(self.model, "holdout_resid", None)
        holdout_resid_len = getattr(self.model, "holdout_resid_len", None)
        if holdout_resid is None or holdout_resid_len is None or len(holdout_resid) == 0:
            return

        rand_resid_ind = np.random.randint(len(holdout_resid), size=len(rows))
        resid_len = holdout_resid_len[rand_resid_ind].astype(int)
        resid_time = np.minimum(source_time, resid_len - 1)
        outputs_next_vitals_scaled[rows, source_time, :] += holdout_resid[rand_resid_ind, resid_time, :]

    def _unscale_clinical_states(self, cf_dataset, predictions_scaled: np.ndarray):
        scaling_params = getattr(cf_dataset, "scaling_params", {}) or {}
        dim_outcome = int(getattr(self.model, "dim_outcome", len(getattr(cf_dataset, "outcome_cols", []))))

        outcomes_scaled = predictions_scaled[:, :, :dim_outcome]
        vitals_scaled = predictions_scaled[:, :, dim_outcome:]

        output_means = np.asarray(scaling_params.get("output_means", np.zeros(dim_outcome)), dtype=float)
        output_stds = np.asarray(scaling_params.get("output_stds", np.ones(dim_outcome)), dtype=float)
        vital_dim = vitals_scaled.shape[-1]
        vital_means = np.asarray(scaling_params.get("vital_means", np.zeros(vital_dim)), dtype=float)
        vital_stds = np.asarray(scaling_params.get("vital_stds", np.ones(vital_dim)), dtype=float)

        outcomes_unscaled = outcomes_scaled * output_stds.reshape(1, 1, -1) + output_means.reshape(1, 1, -1)
        vitals_unscaled = vitals_scaled * vital_stds.reshape(1, 1, -1) + vital_means.reshape(1, 1, -1)
        return outcomes_unscaled, vitals_unscaled

    def _snap_categorical_clinical_outcomes(self, outcomes_unscaled: np.ndarray, cf_dataset) -> np.ndarray:
        if not bool(getattr(self.model, "outcome_is_categorical", False)):
            return outcomes_unscaled

        class_values = self._clinical_outcome_class_values(cf_dataset)
        if class_values.size == 0:
            return outcomes_unscaled

        flat = outcomes_unscaled.reshape(-1)
        nearest = np.abs(flat[:, None] - class_values.reshape(1, -1)).argmin(axis=1)
        return class_values[nearest].reshape(outcomes_unscaled.shape)

    def _clinical_outcome_class_values(self, cf_dataset) -> np.ndarray:
        model_class_values = getattr(self.model, "outcome_class_values", None)
        if model_class_values is not None:
            if hasattr(model_class_values, "detach"):
                values = model_class_values.detach().cpu().numpy()
            else:
                values = np.asarray(model_class_values)
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                return values

        data_processed_seq = getattr(cf_dataset, "data_processed_seq", {}) or {}
        unscaled_outputs = np.asarray(data_processed_seq.get("unscaled_outputs", []), dtype=float)
        if unscaled_outputs.size == 0:
            return np.array([], dtype=float)
        values = unscaled_outputs[np.isfinite(unscaled_outputs)]
        return np.unique(values.astype(float))

    def _apply_generated_states_to_frame(
        self,
        frame: pd.DataFrame,
        cf_dataset,
        outcomes_unscaled: np.ndarray,
        vitals_unscaled: np.ndarray,
    ) -> None:
        id_col = cf_dataset.id_col
        time_col = cf_dataset.time_col
        outcome_cols = list(cf_dataset.outcome_cols)
        vital_cols = list(cf_dataset.vital_cols)
        generated_cols = outcome_cols + vital_cols
        missing_cols = [col for col in [id_col, time_col] + generated_cols if col not in frame.columns]
        if missing_cols:
            raise ValueError(f"Generative template is missing required columns: {missing_cols}")

        data_processed_seq = cf_dataset.data_processed_seq
        patient_ids = data_processed_seq["patient_ids"]
        split = data_processed_seq["future_past_split"].astype(int)
        active_entries = data_processed_seq["active_entries"]
        max_seq_length = getattr(cf_dataset, "max_seq_length", None)

        for patient_idx, patient_id in enumerate(patient_ids):
            patient_rows = frame[frame[id_col] == patient_id].sort_values(time_col)
            if max_seq_length:
                patient_rows = patient_rows.iloc[:max_seq_length]
            patient_row_indices = patient_rows.index.to_numpy()

            for horizon_idx in range(outcomes_unscaled.shape[1]):
                if not np.any(active_entries[patient_idx, horizon_idx, :] > 0):
                    continue

                visit_position = int(split[patient_idx] + horizon_idx + 1)
                if visit_position >= len(patient_row_indices):
                    continue

                row_index = patient_row_indices[visit_position]
                for outcome_idx, col in enumerate(outcome_cols):
                    frame.at[row_index, col] = outcomes_unscaled[patient_idx, horizon_idx, outcome_idx]
                for vital_idx, col in enumerate(vital_cols):
                    frame.at[row_index, col] = vitals_unscaled[patient_idx, horizon_idx, vital_idx]
