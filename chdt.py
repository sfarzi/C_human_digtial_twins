import numpy
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pytorch_lightning import Trainer
from typing import Union, List

from models.g_formula.data.dataset_collection import RealDatasetCollection, SyntheticDatasetCollection
from models.CHDT.time_varying_model import TimeVaryingCausalModel
from models.CHDT.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, ROutcomeVitalsHead
from models.CHDT.utils_lstm import VariationalLSTM


logger = logging.getLogger(__name__)


class CHDT(TimeVaryingCausalModel):
    """
    Intervention-conditioned Counterfactual Human Digital Twin generator.
    """

    model_type = 'chdt'
    possible_model_types = {'chdt'}
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        assert self.autoregressive  # Works only in autoregressive regime
        self.dim_outcome_target = self.dim_outcome
        self._init_outcome_loss_config(args.model.chdt)

        self.dim_time_gap = (
            args.model.get("dim_time_gap", 0) if hasattr(args.model, "get")
            else getattr(args.model, "dim_time_gap", 0)
        )
        self.dim_time_gap = 0 if self.dim_time_gap is None else self.dim_time_gap

        self.input_size = self.dim_treatments + self.dim_time_gap + self.dim_static_features + self.dim_outcome
        self.input_size += self.dim_vitals if self.has_vitals else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_vitals + self.dim_outcome_model

        self._init_specific(args.model.chdt)
        self.save_hyperparameters(args)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _infer_configured_class_values(self):
        if self.dataset_collection is None or not hasattr(self.dataset_collection, "train_f"):
            return None
        data = getattr(self.dataset_collection.train_f, "data", {}) or {}
        outputs = data.get("unscaled_outputs", data.get("outputs"))
        active_entries = data.get("active_entries")
        if outputs is None or active_entries is None:
            return None

        active_mask = np.asarray(active_entries) > 0
        values = np.asarray(outputs)[active_mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return np.unique(values.astype(float))

    def _scaled_class_values(self, class_values):
        if self.dataset_collection is None or not hasattr(self.dataset_collection, "train_f"):
            return class_values

        scaling_params = getattr(self.dataset_collection.train_f, "scaling_params", {}) or {}
        output_means = scaling_params.get("output_means")
        output_stds = scaling_params.get("output_stds")
        if output_means is None or output_stds is None:
            return class_values

        output_means = np.asarray(output_means, dtype=float)
        output_stds = np.asarray(output_stds, dtype=float)
        if output_means.size != 1 or output_stds.size != 1:
            return class_values
        return (class_values - output_means.reshape(-1)[0]) / output_stds.reshape(-1)[0]

    def _init_outcome_loss_config(self, sub_args: DictConfig):
        self.outcome_type = str(self._cfg_get(sub_args, "outcome_type", "continuous")).lower()
        if self.outcome_type not in {"continuous", "categorical"}:
            raise ValueError(f"Unsupported model.chdt.outcome_type: {self.outcome_type}")

        loss_weights = self._cfg_get(sub_args, "loss_weights", [1.0, 1.0])
        self.loss_weights = [float(weight) for weight in loss_weights]
        if len(self.loss_weights) != 2:
            raise ValueError("model.chdt.loss_weights must be [outcome_loss_weight, vitals_loss_weight].")

        self.outcome_is_categorical = self.outcome_type == "categorical"
        if not self.outcome_is_categorical:
            self.dim_outcome_model = self.dim_outcome_target
            self.register_buffer("outcome_class_values", torch.empty(0), persistent=False)
            self.register_buffer("outcome_class_prediction_values", torch.empty(0), persistent=False)
            return

        if self.dim_outcome_target != 1:
            raise ValueError("Categorical CHDT outcome loss currently supports exactly one outcome column.")

        configured_class_values = self._cfg_get(sub_args, "outcome_class_values", None)
        class_values = None if configured_class_values is None else np.asarray(list(configured_class_values), dtype=float)
        if class_values is None:
            class_values = self._infer_configured_class_values()

        num_outcome_classes = self._cfg_get(sub_args, "num_outcome_classes", None)
        if class_values is None:
            if num_outcome_classes is None:
                raise ValueError(
                    "model.chdt.num_outcome_classes or model.chdt.outcome_class_values "
                    "must be set when outcome_type is categorical."
                )
            class_values = np.arange(int(num_outcome_classes), dtype=float)

        if num_outcome_classes is not None and int(num_outcome_classes) != len(class_values):
            raise ValueError(
                "model.chdt.num_outcome_classes must match the number of outcome_class_values "
                f"({int(num_outcome_classes)} != {len(class_values)})."
            )

        configured_prediction_values = self._cfg_get(sub_args, "outcome_class_prediction_values", None)
        if configured_prediction_values is None:
            prediction_values = self._scaled_class_values(class_values)
        else:
            prediction_values = np.asarray(list(configured_prediction_values), dtype=float)
            if len(prediction_values) != len(class_values):
                raise ValueError(
                    "model.chdt.outcome_class_prediction_values must match outcome_class_values length."
                )

        self.dim_outcome_model = len(class_values)
        self.register_buffer(
            "outcome_class_values",
            torch.as_tensor(class_values, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.register_buffer(
            "outcome_class_prediction_values",
            torch.as_tensor(prediction_values, dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of the CHDT-specific sub-network.
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            self.dropout_rate = sub_args.dropout_rate
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.r_size = sub_args.r_size
            self.num_layer = sub_args.num_layer
            self.comp_sizes = sub_args.comp_sizes
            self.num_comp = sub_args.num_comp
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.mc_samples = sub_args.mc_samples
            self.use_reverse_component_conditioning = getattr(
                sub_args, "use_reverse_component_conditioning", False
            )

            # Params for Representation network
            if self.seq_hidden_units is None or self.r_size is None or self.dropout_rate is None or self.fc_hidden_units is None:
                raise MissingMandatoryValue()

            # Params for Conditional distribution networks
            assert len(self.comp_sizes) == self.num_comp
            assert sum(self.comp_sizes) == self.output_size

            # Representation network init + Conditional distribution networks init
            self.repr_net = VariationalLSTM(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate)
            self.r_outcome_vitals_head = ROutcomeVitalsHead(self.seq_hidden_units, self.r_size, self.fc_hidden_units,
                                                            self.dim_outcome_model, self.dim_vitals, self.num_comp, self.comp_sizes)
            if self.use_reverse_component_conditioning:
                self.component_refinement_heads = nn.ModuleList()

                for comp_size in self.comp_sizes:
                    other_comp_size = self.output_size - comp_size

                    self.component_refinement_heads.append(
                        nn.Sequential(
                            nn.Linear(self.r_size + other_comp_size, self.fc_hidden_units),
                            nn.ELU(),
                            nn.Dropout(self.dropout_rate),
                            nn.Linear(self.fc_hidden_units, comp_size)
                        )
                    )

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.dataset_collection is not None:
            self.dataset_collection.split_train_f_holdout(self.hparams.dataset.holdout_ratio)
            self.dataset_collection.explode_cf_treatment_seq(self.mc_samples)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.seq_hidden_units = new_args['seq_hidden_units']
        sub_args.r_size = new_args['r_size']
        sub_args.fc_hidden_units = new_args['fc_hidden_units']
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']

    def forward(self, batch, sample=False):
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        time_gap = batch.get("time_gap", None)

        r = self.build_r(curr_treatments, vitals, prev_outputs, static_features, time_gap)
        initial_pred = self.r_outcome_vitals_head.build_outcome_vitals(r)

        if not self.use_reverse_component_conditioning:
            return initial_pred

        refined_pred = self.refine_mediator_states(r, initial_pred)
        return refined_pred

    def split_mediator_priors(self, pred):
        return torch.split(pred, list(self.comp_sizes), dim=-1)

    def refine_mediator_states(self, r, initial_pred):
        """
        Optional reverse component conditioning: refine each component from the shared
        LSTM representation and the initial predictions of all other components,
        preserving the same output shape and training pipeline.
        """
        components = list(self.split_mediator_priors(initial_pred))
        refined_components = []

        for i, head in enumerate(self.component_refinement_heads):
            other_components = [
                comp for j, comp in enumerate(components) if j != i
            ]

            if len(other_components) > 0:
                other_components = torch.cat(other_components, dim=-1)
            else:
                other_components = torch.empty(
                    r.size(0), r.size(1), 0,
                    device=r.device,
                    dtype=r.dtype
                )

            refinement_input = torch.cat(
                [r, other_components.detach()],
                dim=-1
            )

            refined_component = head(refinement_input)
            refined_components.append(refined_component)

        return torch.cat(refined_components, dim=-1)

    def build_r(self, curr_treatments, vitals, prev_outputs, static_features, time_gap=None):
        input_parts = [curr_treatments]

        if self.dim_time_gap > 0:
            if time_gap is None:
                raise ValueError('dim_time_gap > 0 but batch is missing "time_gap".')
            if time_gap.dim() == 2:
                time_gap = time_gap.unsqueeze(-1)
            input_parts.append(time_gap)

        input_parts.append(vitals) if self.has_vitals else None
        input_parts.append(prev_outputs) if self.autoregressive else None
        input_parts.append(static_features.unsqueeze(1).expand(-1, curr_treatments.size(1), -1))

        x = torch.cat(input_parts, dim=-1)
        x = self.repr_net(x)
        r = self.r_outcome_vitals_head.build_r(x)
        return r

    def split_outcome_next_mediators(self, outcome_next_vitals_pred):
        outcome_pred = outcome_next_vitals_pred[:, :, :self.dim_outcome_model]
        next_vitals_pred = outcome_next_vitals_pred[:, :, self.dim_outcome_model:]
        return outcome_pred, next_vitals_pred

    def _target_class_indices(self, batch):
        target_values = batch.get("unscaled_outputs", batch["outputs"])
        target_values = target_values[:, :, 0]
        class_values = self.outcome_class_values.to(device=target_values.device, dtype=target_values.dtype)
        safe_target_values = torch.where(
            torch.isfinite(target_values),
            target_values,
            class_values[0],
        )
        distances = torch.abs(safe_target_values.unsqueeze(-1) - class_values.view(1, 1, -1))
        return distances.argmin(dim=-1).long()

    def _masked_outcome_loss(self, outcome_pred, batch):
        active_entries = batch['active_entries']
        active_sum = active_entries.sum()
        if active_sum <= 0:
            return outcome_pred.sum() * 0.0

        if self.outcome_is_categorical:
            target_classes = self._target_class_indices(batch)
            outcome_loss = F.cross_entropy(
                outcome_pred.permute(0, 2, 1),
                target_classes,
                reduction='none',
            ).unsqueeze(-1)
        else:
            outcome_loss = F.mse_loss(outcome_pred, batch['outputs'], reduction='none')

        return (active_entries * outcome_loss).sum() / active_sum

    def _masked_mediator_loss(self, next_vitals_pred, batch, reference_loss):
        vitals_mse_loss = F.mse_loss(next_vitals_pred[:, :-1, :], batch['next_vitals'], reduction='none') \
            if self.has_vitals else None

        if self.hparams.model.chdt.fit_vitals and self.has_vitals:
            vitals_active_entries = batch['active_entries'][:, 1:, :]
            vitals_active_sum = vitals_active_entries.sum()
            if vitals_active_sum > 0:
                return (vitals_active_entries * vitals_mse_loss).sum() / vitals_active_sum
            else:
                return reference_loss.new_tensor(0.0)
        return reference_loss.new_tensor(0.0)

    def _combine_losses(self, outcome_loss, vitals_loss):
        if self.outcome_is_categorical:
            return self.loss_weights[0] * outcome_loss + self.loss_weights[1] * vitals_loss
        return outcome_loss + vitals_loss

    def _compute_chdt_training_losses(self, batch):
        outcome_next_vitals_pred = self(batch)  # By convention order is (outcomes, vitals)
        outcome_pred, next_vitals_pred = self.split_outcome_next_mediators(outcome_next_vitals_pred)

        outcome_loss = self._masked_outcome_loss(outcome_pred, batch)
        vitals_loss = self._masked_mediator_loss(next_vitals_pred, batch, outcome_loss)
        total_loss = self._combine_losses(outcome_loss, vitals_loss)
        return total_loss, outcome_loss, vitals_loss

    def training_step(self, batch, batch_ind):
        total_loss, outcome_loss, vitals_loss = self._compute_chdt_training_losses(batch)

        self.log(f'{self.model_type}_train_loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_outcome_loss', outcome_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_vitals_mse_loss', vitals_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_mse_loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
        return total_loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        return self.format_clinical_state_prediction(self(batch)).cpu()

    def format_clinical_state_prediction(self, outcome_next_vitals_pred):
        if not self.outcome_is_categorical:
            return outcome_next_vitals_pred

        outcome_logits, next_vitals_pred = self.split_outcome_next_mediators(outcome_next_vitals_pred)
        class_indices = outcome_logits.argmax(dim=-1)
        prediction_values = self.outcome_class_prediction_values.to(
            device=outcome_logits.device,
            dtype=outcome_logits.dtype,
        )
        outcome_pred = prediction_values[class_indices].unsqueeze(-1)
        return torch.cat([outcome_pred, next_vitals_pred], dim=-1)

    def on_fit_end(self) -> None:
        if self.dataset_collection is not None and hasattr(self.dataset_collection, 'train_f_holdout') \
                and len(self.dataset_collection.train_f_holdout) > 0:
            logger.info('Fitting residuals based on train_f_holdout.')
            self.eval()
            outcome_next_vitals_pred = self.get_predictions(self.dataset_collection.train_f_holdout, vitals=True)

            outcomes_next_vitals = self.dataset_collection.train_f_holdout.data['outputs']
            if self.has_vitals:
                # No ground truth for the last next_vitals
                outcome_next_vitals_pred = outcome_next_vitals_pred[:, :-1, :]
                outcomes_next_vitals = outcomes_next_vitals[:, :-1, :]

                vitals = self.dataset_collection.train_f_holdout.data['next_vitals']
                outcomes_next_vitals = np.concatenate((outcomes_next_vitals, vitals), axis=-1)

            self.holdout_resid = outcomes_next_vitals - outcome_next_vitals_pred
            self.holdout_resid_len = self.dataset_collection.train_f_holdout.data['sequence_lengths']
            if self.has_vitals:
                # No ground truth for the last next_vitals
                self.holdout_resid_len = self.holdout_resid_len - 1

            valid_resid = self.holdout_resid_len > 0
            if valid_resid.sum() == 0:
                logger.info('No positive-length holdout residual trajectories; residual MC-sampling is disabled.')
                self.holdout_resid = self.holdout_resid_len = None
            elif valid_resid.sum() < len(valid_resid):
                logger.info('Dropping %s zero-length holdout residual trajectories.', len(valid_resid) - valid_resid.sum())
                self.holdout_resid = self.holdout_resid[valid_resid]
                self.holdout_resid_len = self.holdout_resid_len[valid_resid]
        else:  # Without MC-sampling of residuals
            self.holdout_resid = self.holdout_resid_len = None

    def get_predictions(self, dataset: Union[Dataset, List[Dataset]], vitals=False) -> np.array:
        if not isinstance(dataset, list):
            logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        if isinstance(dataset, list):
            data_loader = \
                [DataLoader(d, batch_size=self.hparams.dataset.val_batch_size, shuffle=False, num_workers=2) for d in dataset]
        else:
            data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)

        outcome_vitals_pred = self.trainer.predict(self, data_loader)

        if isinstance(dataset, list):
            outcome_vitals_pred = np.stack([torch.cat(pred).numpy() for pred in outcome_vitals_pred], axis=0)
            if vitals:
                return outcome_vitals_pred
            else:
                return outcome_vitals_pred[:, :, :, :self.dim_outcome]

        else:
            outcome_vitals_pred = torch.cat(outcome_vitals_pred).numpy()
            if vitals:
                return outcome_vitals_pred
            else:
                return outcome_vitals_pred[:, :, :self.dim_outcome]

    def get_autoregressive_predictions(self, datasets: List[Dataset]) -> np.array:
        assert hasattr(self, 'holdout_resid') and hasattr(self, 'holdout_resid_len')
        assert len(datasets) == self.mc_samples
        logger.info(f'Autoregressive Prediction for {datasets[0].subset_name} with MC-sampling of trajectories.')

        predicted_outputs = np.zeros((self.mc_samples, len(datasets[0]), self.hparams.dataset.projection_horizon,
                                      self.dim_outcome))

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f't = {t + 1}')

            # MC-sampling of trajectories
            for m in range(self.mc_samples):
                outputs_next_vitals_scaled = self.get_predictions(datasets[m], vitals=True)
                split = datasets[m].data['future_past_split'].astype(int)

                if t > 0:  # Tau >= 2
                    predicted_outputs[m, :, t - 1, :] = \
                        outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, :self.dim_outcome]

                # Adding noise from empirical distribution of residuals
                if self.holdout_resid is not None and len(self.holdout_resid) > 0:
                    rand_resid_ind = np.random.randint(len(self.holdout_resid), size=len(datasets[m]))
                    resid_len = self.holdout_resid_len[rand_resid_ind].astype(int)
                    resid_at_split = self.holdout_resid[rand_resid_ind, np.minimum(split - 1 + t, resid_len - 1), :]
                    outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, :] += resid_at_split

                # Autoregressive feeding of predicted outcomes and vitals
                if t < self.hparams.dataset.projection_horizon:
                    datasets[m].data['prev_outputs'][range(len(datasets[m])), split + t, :] = \
                        outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, :self.dim_outcome]

                    if self.has_vitals:
                        datasets[m].data['vitals'][range(len(datasets[m])), split + t, :] = \
                            outputs_next_vitals_scaled[range(len(datasets[m])), split - 1 + t, self.dim_outcome:]

        predicted_outputs = predicted_outputs.mean(0)  # Averaging over mc_samples
        return predicted_outputs
