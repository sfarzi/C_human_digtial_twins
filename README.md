# CHDT supplementary code package

This anonymous supplementary package provides the CHDT implementation and the fixed simulator-generated ALPACA cohorts needed to train and evaluate the model. It is intended to help reviewers inspect the implementation and reproduce the ALPACA experiment.

## Contents

The implementation includes the factual-history encoder, counterfactual autoregressive trajectory generation, elapsed-time input, mediator refinement heads, one-step evaluation, and multi-horizon factual and intervention-arm evaluation. 

## Tested environment

- Python 3.12.6
- PyTorch 2.11.0 with CUDA 13.0
- One CUDA-capable GPU
- Linux

The complete tested Python package versions are pinned in `requirements.txt`.



## Direct configuration overrides

Hydra command-line overrides can be passed directly to the training entry point. For example, a short diagnostic run is:

```bash
python models/CHDT/train_chdt.py 'exp.gpus=[0]' exp.max_epochs=1
```

The paper configuration is stored in:

```text
models/CHDT/config_alpaca_chdt_tuned.yaml
```

## Outputs

Training writes timestamped files under:

```text
models/CHDT/chdt_reports/
```

Each completed run produces:

- a structured text report containing configuration, data summaries, model size, runtime, RMSE, and classification metrics;
- a training-loss plot.

Hydra may additionally create a timestamped `outputs/` directory containing run metadata.

## Reproducibility notes

- The supplied configuration uses seed 100.
- Cohort splitting is deterministic for the configured seed.
- GPU kernels and differences between hardware/software builds can introduce small numerical variation.
- Long-horizon metrics use fewer active patients as the horizon increases. Results with very small active-support counts should be interpreted cautiously.
- Hyperparameter tuning is disabled; the package uses the fixed configuration directly.
- Experiment logging is disabled, so no external tracking service is required.



## Anonymous-review status

The package excludes author identities, affiliations, email addresses, repository history, machine-specific paths, scheduler artifacts, and real-world cohort data.
