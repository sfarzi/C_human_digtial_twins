# ALPACA cohort files

This package contains three simulator-generated longitudinal cohorts:

- `normal_arm.csv`: development cohort used for training, validation, and factual testing.
- `placebo_arm.csv`: held-out placebo intervention arm.
- `statin_treatment_arm.csv`: held-out statin intervention arm.

The files contain synthetic patient identifiers and simulated clinical trajectories; they do not contain real patient records or direct personal identifiers.

The dataset adapter expects one row per patient visit. The outcome is `AD`, treatment is `medication`, elapsed time is represented by `Gap`, and the remaining longitudinal measurements configured in `dataset.py` are mediator states.

The included cohort files are the fixed inputs used by the supplied ALPACA configuration. Their file contents must not be changed when reproducing the reported run.
