"""
Generate a three-arm synthetic longitudinal Alzheimer's disease clinical trial
dataset using the real ALPACA-DT-Sim package.

Install first:
    pip install ALPACA-DT-Sim pandas numpy gymnasium

Outputs:
    placebo_arm.csv
    statin_treatment_arm.csv
    normal_arm.csv
    cohort_generation_statistics.txt

Scientific purpose:
    Generate a simulated multi-arm clinical trial dataset to investigate
    long-term statin exposure and incident dementia progression.

Diagnosis framework:
    ALPACA generates the observed medical trajectory: cognition, CSF biomarkers,
    brain volumes, demographics, and treatment-conditioned visit transitions.
    This script then adds a stochastic diagnosis layer on top. A hidden disease
    progression process, patient-level frailty, and recent clinical worsening
    determine when a patient converts from AD = 0 to AD = 1. Converted patients
    must show a plausible deterioration signal around conversion. Non-converters
    remain AD = 0 throughout follow-up.

Arms:
    1. Placebo arm:
        - medication = 0
        - no statin action applied

    2. Statin treatment arm:
        - medication = 1
        - statin action applied at every visit transition

    3. Normal mixed arm:
        - some patients use statins
        - statin start visit is randomly assigned among observed visits

Visit structure:
    - Up to 10 visits per patient
    - Some patients have fewer visits to represent missed/incomplete follow-up
    - 12 months between visits

Important caution:
    This generates an in-silico / simulation-based dataset.
    It should be used for hypothesis generation, not causal proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ALPACA_DT_Sim import ALPACAEnv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "ALPACA_DT_Sim is not installed in the Python environment running this "
        "script. In this workspace it is installed in `.venv311`, so run:\n\n"
        "    source .venv311/bin/activate\n"
        "    .venv311/bin/python Alpaca_Cohort_generation/Cohort_generator.py\n\n"
        "The existing `.venv312` environment is missing ALPACA-DT-Sim and its "
        "Python build also cannot import `ssl`, which prevents pip from "
        "installing packages from PyPI."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# ALPACA OBSERVATION FEATURES
# ============================================================

OBSERVATION_FEATURES: List[str] = [
    "ADNI_MEM",
    "ADNI_EF2",
    "TAU_data",
    "subject_age",
    "ABETA",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
    "PTGENDER_Female",
    "PTGENDER_Male",
    "PTRACCAT_Am Indian/Alaskan",
    "PTRACCAT_Asian",
    "PTRACCAT_Black",
    "PTRACCAT_Hawaiian/Other PI",
    "PTRACCAT_More than one",
    "PTRACCAT_Unknown",
    "PTRACCAT_White",
]


# ============================================================
# ALPACA MEDICATION ACTION SPACE
# ============================================================

ACTION_NAMES: List[str] = [
    "AD Treatment_active",
    "Alpha Blocker_active",
    "Analgesic_active",
    "Antidepressant_active",
    "Antihypertensive_active",
    "Bone Health_active",
    "Diabetes Medication_active",
    "Diuretic_active",
    "NSAID_active",
    "No Medication_active",
    "Other_active",
    "PPI_active",
    "SSRI_active",
    "Statin_active",
    "Steroid_active",
    "Supplement_active",
    "Thyroid Hormone_active",
]

STATIN_INDEX = ACTION_NAMES.index("Statin_active")
NO_MEDICATION_INDEX = ACTION_NAMES.index("No Medication_active")


# ============================================================
# AD PATIENT IDENTIFICATION STRATEGIES
# ============================================================

class ADCriterion(str, Enum):
    """
    Different strategies for identifying AD-like patients from ALPACA observations.

    ADNI_MEM_ONLY:
        Uses only memory impairment:
            ADNI_MEM <= -1.0

    COGNITIVE_ONLY:
        Uses both memory and executive-function impairment:
            ADNI_MEM < -1.0
            AND ADNI_EF2 < -0.8

    MRI_ONLY:
        Uses structural MRI abnormality:
            Hippocampus below the 25th percentile
            AND Ventricles above the 75th percentile

    COGNITIVE_AND_MRI:
        Most restrictive phenotype:
            ADNI_MEM < -1.0
            AND ADNI_EF2 < -0.8
            AND Hippocampus below the 25th percentile
            AND Ventricles above the 75th percentile

    Recommendation:
        For a stricter AD-only virtual cohort, use COGNITIVE_AND_MRI.
        For easier sampling, use ADNI_MEM_ONLY or COGNITIVE_ONLY.
    """

    ADNI_MEM_ONLY = "adni_mem_only"
    COGNITIVE_ONLY = "cognitive_only"
    MRI_ONLY = "mri_only"
    COGNITIVE_AND_MRI = "cognitive_and_mri"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrialConfig:
    """
    Configuration object for synthetic clinical trial generation.

    Parameters
    ----------
    n_patients:
        Number of synthetic patients to generate in each arm.
        The number of final dementia converters is controlled by
        target_dementia_probability.

    max_visits:
        Maximum number of visits per patient.

    min_visits:
        Minimum number of visits per patient.

    visit_interval_months:
        Time interval between consecutive visits.

    random_seed:
        Seed for reproducibility.

    alpaca_cohort_type:
        ALPACA baseline cohort sampler. For progression modelling, "all" is the
        safest default because all patients start undiagnosed in the generated
        output and only some later convert.

    target_dementia_probability:
        Patient-level probability of ever developing dementia during follow-up.
        These patients are forced to start AD = 0 and convert to AD = 1 at a
        later observed visit if a clinically plausible conversion window exists.

    min_conversion_visit:
        Earliest visit index at which a dementia conversion may be recorded.
        The default of 1 prevents baseline-prevalent dementia in this generator.

    conversion_min_progression_signal:
        Minimum worsening signal required around a conversion visit. The signal
        uses recent and cumulative deterioration, but it is a plausibility filter,
        not a deterministic label rule.

    latent_* and diagnosis_*:
        Parameters for the hidden stochastic diagnosis process. The hidden
        process is not written to CSV and is not available to downstream models.

    normal_arm_statin_probability:
        Proportion of patients in the Normal_Mixed arm who become statin users.

    max_resampling_attempts:
        Maximum number of attempts to sample acceptable patients.
        Increase this if using strict AD rules.
    """

    n_patients: int = 1000
    max_visits: int = 10
    min_visits: int = 2
    visit_interval_months: int = 12
    random_seed: int = 42

    alpaca_cohort_type: str = "all"
    target_dementia_probability: float = 0.50
    min_conversion_visit: int = 1
    conversion_min_progression_signal: float = 0.05
    latent_frailty_sd: float = 0.85
    latent_progression_mean: float = 0.18
    latent_progression_sd: float = 0.08
    latent_innovation_sd: float = 0.18
    diagnosis_hazard_intercept: float = -1.35
    diagnosis_progression_weight: float = 1.75
    diagnosis_age_weight: float = 0.18
    diagnosis_treatment_log_odds: float = -0.20
    diagnosis_noise_sd: float = 0.35

    # Legacy baseline phenotype parameters retained for optional diagnostics.
    # They are no longer used to assign the AD label.
    ad_criterion: ADCriterion = ADCriterion.COGNITIVE_AND_MRI
    adni_mem_threshold: float = -1.0
    adni_ef2_threshold: float = -0.8
    percentile_reference_samples: int = 5000
    normal_arm_statin_probability: float = 0.30

    max_resampling_attempts: int = 50000


# ============================================================
# ENVIRONMENT CREATION
# ============================================================

def create_alpaca_env(
    visit_interval_months: int = 12,
    cohort_type: str = "all",
) -> ALPACAEnv:
    """
    Create the ALPACA environment.

    ALPACA supplies the observed medical trajectory. The dementia diagnosis
    process is added by this script, so the default baseline sampler is "all"
    rather than preselecting only already-impaired patients.

    Returns
    -------
    ALPACAEnv
        A Gym-compatible ALPACA environment.
    """

    return ALPACAEnv(
        reward_metric="ADNI_MEM",
        cohort_type=cohort_type,
        time_delta_months=float(visit_interval_months),
    )


# ============================================================
# ACTION FUNCTIONS
# ============================================================

def make_placebo_action() -> np.ndarray:
    """
    Create the action vector for the placebo arm.

    In this script, placebo means:
        - no statin exposure
        - medication column will be 0

    We set the ALPACA action as "No Medication" to create a clean comparator arm.

    Returns
    -------
    np.ndarray
        A binary medication-action vector.
    """

    action = np.zeros(len(ACTION_NAMES), dtype=np.int8)
    action[NO_MEDICATION_INDEX] = 1
    return action


def make_statin_action() -> np.ndarray:
    """
    Create the action vector for the statin treatment arm.

    In this script, treatment means:
        - statin is active at each simulated transition
        - medication column will be 1

    Returns
    -------
    np.ndarray
        A binary medication-action vector with Statin_active = 1.
    """

    action = np.zeros(len(ACTION_NAMES), dtype=np.int8)
    action[STATIN_INDEX] = 1
    return action


# ============================================================
# FEATURE ACCESS AND DECODING
# ============================================================

def get_feature_value(obs: np.ndarray, feature_name: str) -> float:
    """
    Extract one named feature from an ALPACA observation vector.

    Parameters
    ----------
    obs:
        ALPACA observation vector.

    feature_name:
        Name of the feature to extract.

    Returns
    -------
    float
        Feature value.
    """

    return float(obs[OBSERVATION_FEATURES.index(feature_name)])


def decode_gender(obs: np.ndarray) -> str:
    """
    Decode ALPACA one-hot gender variables into one readable value.

    Returns
    -------
    str
        "Female" or "Male".
    """

    female = get_feature_value(obs, "PTGENDER_Female")
    male = get_feature_value(obs, "PTGENDER_Male")

    return "Female" if female >= male else "Male"


def decode_race(obs: np.ndarray) -> str:
    """
    Decode ALPACA one-hot race variables into one readable race category.

    Returns
    -------
    str
        Race category inferred from the highest one-hot encoded value.
    """

    race_columns = [
        "PTRACCAT_Am Indian/Alaskan",
        "PTRACCAT_Asian",
        "PTRACCAT_Black",
        "PTRACCAT_Hawaiian/Other PI",
        "PTRACCAT_More than one",
        "PTRACCAT_Unknown",
        "PTRACCAT_White",
    ]

    values = [get_feature_value(obs, col) for col in race_columns]
    selected = race_columns[int(np.argmax(values))]

    return selected.replace("PTRACCAT_", "")


# ============================================================
# AD PATIENT IDENTIFICATION FUNCTIONS
# ============================================================

def estimate_feature_percentiles(
    env: ALPACAEnv,
    feature_names: Sequence[str],
    n_samples: int = 5000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Estimate percentile thresholds for selected ALPACA features.

    Why this is needed
    ------------------
    Some AD-identification criteria are percentile-based rather than fixed-value
    thresholds. For example:

        - Hippocampus below the 25th percentile
        - Ventricles above the 75th percentile

    Because ALPACA observations are generated by the environment, this function
    samples many virtual patients and estimates the empirical feature
    distribution.

    Parameters
    ----------
    env:
        ALPACA environment.

    feature_names:
        Names of features for which percentiles should be estimated.

    n_samples:
        Number of virtual baseline patients to sample.

    seed:
        Random seed.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Example:
        {
            "Hippocampus": {"p25": ..., "p75": ...},
            "Ventricles": {"p25": ..., "p75": ...}
        }
    """

    rng = np.random.default_rng(seed)
    values = {name: [] for name in feature_names}

    for _ in range(n_samples):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        for name in feature_names:
            values[name].append(get_feature_value(obs, name))

    return {
        name: {
            "p25": float(np.percentile(values[name], 25)),
            "p75": float(np.percentile(values[name], 75)),
        }
        for name in feature_names
    }


def is_ad_patient(
    obs: np.ndarray,
    strategy: ADCriterion = ADCriterion.COGNITIVE_AND_MRI,
    adni_mem_threshold: float = -1.0,
    adni_ef2_threshold: float = -0.8,
    percentile_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> bool:
    """
    Decide whether one sampled virtual patient should be included as an AD patient.

    This is one of the most important functions in the script because it defines
    the inclusion criteria for the simulated trial.

    Strategies
    ----------
    1. ADNI_MEM_ONLY
       Includes patient if:
           ADNI_MEM <= -1.0

       Interpretation:
           The patient has strong memory impairment.

    2. COGNITIVE_ONLY
       Includes patient if:
           ADNI_MEM < -1.0
           AND ADNI_EF2 < -0.8

       Interpretation:
           The patient has both memory and executive-function impairment.

    3. MRI_ONLY
       Includes patient if:
           Hippocampus < 25th percentile
           AND Ventricles > 75th percentile

       Interpretation:
           The patient shows structural brain changes compatible with AD-like
           neurodegeneration: hippocampal atrophy and ventricular enlargement.

    4. COGNITIVE_AND_MRI
       Includes patient if:
           ADNI_MEM < -1.0
           AND ADNI_EF2 < -0.8
           AND Hippocampus < 25th percentile
           AND Ventricles > 75th percentile

       Interpretation:
           This is the strictest rule and is recommended when you want a more
           AD-specific virtual cohort.

    Parameters
    ----------
    obs:
        ALPACA observation vector.

    strategy:
        AD-identification strategy.

    adni_mem_threshold:
        Threshold for ADNI_MEM.

    adni_ef2_threshold:
        Threshold for ADNI_EF2.

    percentile_thresholds:
        Percentile dictionary generated by estimate_feature_percentiles.
        Required for MRI_ONLY and COGNITIVE_AND_MRI.

    Returns
    -------
    bool
        True if the patient satisfies the selected AD criterion.
    """

    adni_mem = get_feature_value(obs, "ADNI_MEM")
    adni_ef2 = get_feature_value(obs, "ADNI_EF2")
    hippocampus = get_feature_value(obs, "Hippocampus")
    ventricles = get_feature_value(obs, "Ventricles")

    memory_rule = adni_mem <= adni_mem_threshold

    cognitive_rule = (
        adni_mem < adni_mem_threshold
        and adni_ef2 < adni_ef2_threshold
    )

    if strategy == ADCriterion.ADNI_MEM_ONLY:
        return memory_rule

    if strategy == ADCriterion.COGNITIVE_ONLY:
        return cognitive_rule

    if percentile_thresholds is None:
        raise ValueError(
            "percentile_thresholds is required for MRI-based AD criteria."
        )

    hippocampus_p25 = percentile_thresholds["Hippocampus"]["p25"]
    ventricles_p75 = percentile_thresholds["Ventricles"]["p75"]

    mri_rule = (
        hippocampus < hippocampus_p25
        and ventricles > ventricles_p75
    )

    if strategy == ADCriterion.MRI_ONLY:
        return mri_rule

    if strategy == ADCriterion.COGNITIVE_AND_MRI:
        return cognitive_rule and mri_rule

    raise ValueError(f"Unknown AD criterion strategy: {strategy}")


def reset_until_ad_patient(
    env: ALPACAEnv,
    rng: np.random.Generator,
    config: TrialConfig,
    percentile_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Repeatedly reset ALPACA until a virtual patient satisfies the AD criterion.

    Why this is needed
    ------------------
    ALPACA samples virtual patients from its learned patient distribution.
    Not every sampled patient is necessarily strict AD. Therefore, we repeatedly
    sample candidate patients and keep only those satisfying the selected
    inclusion criteria.

    Parameters
    ----------
    env:
        ALPACA environment.

    rng:
        Random number generator.

    config:
        Trial configuration.

    percentile_thresholds:
        Optional MRI percentile thresholds.

    Returns
    -------
    Tuple[np.ndarray, Dict]
        Observation and ALPACA info dictionary for the accepted patient.

    Raises
    ------
    RuntimeError
        If enough AD-like patients cannot be sampled.
    """

    for _ in range(config.max_resampling_attempts):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=seed)

        if is_ad_patient(
            obs=obs,
            strategy=config.ad_criterion,
            adni_mem_threshold=config.adni_mem_threshold,
            adni_ef2_threshold=config.adni_ef2_threshold,
            percentile_thresholds=percentile_thresholds,
        ):
            return obs, info

    raise RuntimeError(
        "Could not sample an AD-like patient. "
        "Try relaxing the AD criterion, for example use COGNITIVE_ONLY "
        "or ADNI_MEM_ONLY, or increase max_resampling_attempts."
    )


def reset_until_ad_status(
    env: ALPACAEnv,
    rng: np.random.Generator,
    config: TrialConfig,
    desired_ad: int,
    percentile_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Repeatedly reset ALPACA until a virtual patient has the requested AD status.

    Parameters
    ----------
    desired_ad:
        1 to sample a patient satisfying the configured AD criterion.
        0 to sample a patient not satisfying the configured AD criterion.
    """

    if desired_ad not in {0, 1}:
        raise ValueError("desired_ad must be 0 or 1.")

    for _ in range(config.max_resampling_attempts):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=seed)
        has_ad = is_ad_patient(
            obs=obs,
            strategy=config.ad_criterion,
            adni_mem_threshold=config.adni_mem_threshold,
            adni_ef2_threshold=config.adni_ef2_threshold,
            percentile_thresholds=percentile_thresholds,
        )

        if int(has_ad) == desired_ad:
            return obs, info

    label = "AD" if desired_ad == 1 else "non-AD"
    raise RuntimeError(
        f"Could not sample a {label} patient. "
        "Try increasing max_resampling_attempts or changing the AD criterion."
    )


def target_dementia_assignments(config: TrialConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Return shuffled patient-level final dementia assignments.

    Assignment 1 means the patient should convert from AD = 0 to AD = 1 during
    observed follow-up. Assignment 0 means the patient remains AD = 0 throughout.
    """

    if not 0.0 <= config.target_dementia_probability <= 1.0:
        raise ValueError("target_dementia_probability must be between 0 and 1.")

    n_dementia = int(round(config.n_patients * config.target_dementia_probability))

    assignments = np.array(
        [1] * n_dementia + [0] * (config.n_patients - n_dementia),
        dtype=np.int8,
    )
    rng.shuffle(assignments)
    return assignments


def balanced_ad_assignments(config: TrialConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Backward-compatible wrapper for the old static-label function name.
    """

    return target_dementia_assignments(config, rng)


# ============================================================
# STOCHASTIC DEMENTIA PROGRESSION FRAMEWORK
# ============================================================

def sigmoid(value: float) -> float:
    """
    Numerically stable scalar sigmoid.
    """

    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def clinical_worsening_signal(rows: Sequence[Dict[str, object]]) -> np.ndarray:
    """
    Build a bounded, visit-level deterioration signal from ALPACA measurements.

    The signal is used as a plausibility and hazard modifier, not as a hard
    diagnosis rule. Positive values mean the patient has worsened relative to
    baseline and/or the previous visit. ALPACA still supplies the actual
    biomarker/cognition trajectories.
    """

    if not rows:
        return np.array([], dtype=float)

    baseline = rows[0]
    signals: List[float] = []

    for idx, row in enumerate(rows):
        previous = rows[idx - 1] if idx > 0 else row

        cumulative_memory_decline = float(baseline["ADNI_MEM"]) - float(row["ADNI_MEM"])
        cumulative_executive_decline = float(baseline["ADNI_EF2"]) - float(row["ADNI_EF2"])
        cumulative_ventricle_increase = (
            float(row["Ventricles"]) - float(baseline["Ventricles"])
        ) / 10000.0
        cumulative_hippocampus_decline = (
            float(baseline["Hippocampus"]) - float(row["Hippocampus"])
        ) / 500.0
        cumulative_entorhinal_decline = (
            float(baseline["Entorhinal"]) - float(row["Entorhinal"])
        ) / 300.0

        recent_memory_decline = float(previous["ADNI_MEM"]) - float(row["ADNI_MEM"])
        recent_executive_decline = float(previous["ADNI_EF2"]) - float(row["ADNI_EF2"])
        recent_ventricle_increase = (
            float(row["Ventricles"]) - float(previous["Ventricles"])
        ) / 10000.0
        recent_hippocampus_decline = (
            float(previous["Hippocampus"]) - float(row["Hippocampus"])
        ) / 500.0
        recent_entorhinal_decline = (
            float(previous["Entorhinal"]) - float(row["Entorhinal"])
        ) / 300.0

        age_risk = (float(row["subject_age"]) - 75.0) / 10.0

        cumulative_signal = (
            0.32 * cumulative_memory_decline
            + 0.24 * cumulative_executive_decline
            + 0.16 * cumulative_ventricle_increase
            + 0.16 * cumulative_hippocampus_decline
            + 0.08 * cumulative_entorhinal_decline
        )
        recent_signal = (
            0.34 * recent_memory_decline
            + 0.24 * recent_executive_decline
            + 0.16 * recent_ventricle_increase
            + 0.16 * recent_hippocampus_decline
            + 0.08 * recent_entorhinal_decline
        )

        signals.append(float(np.tanh(cumulative_signal + 0.75 * recent_signal + 0.05 * age_risk)))

    return np.asarray(signals, dtype=float)


def conversion_visit_candidates(
    rows: Sequence[Dict[str, object]],
    config: TrialConfig,
) -> np.ndarray:
    """
    Return visit indices that contain enough deterioration to host conversion.

    This is the guard that makes the diagnosis event visible in the medical
    trajectory. It is deliberately permissive: a conversion may be supported by
    cognition or MRI worsening, and the final visit is still sampled
    stochastically by the hidden diagnosis process.
    """

    signals = clinical_worsening_signal(rows)
    if len(rows) < 2 or signals.size == 0:
        return np.array([], dtype=int)

    earliest = max(1, int(config.min_conversion_visit))
    candidates: List[int] = []

    for visit in range(earliest, len(rows)):
        previous = rows[visit - 1]
        current = rows[visit]

        recent_memory_decline = float(previous["ADNI_MEM"]) - float(current["ADNI_MEM"])
        recent_executive_decline = float(previous["ADNI_EF2"]) - float(current["ADNI_EF2"])
        recent_ventricle_increase = float(current["Ventricles"]) - float(previous["Ventricles"])
        recent_hippocampus_decline = float(previous["Hippocampus"]) - float(current["Hippocampus"])
        recent_entorhinal_decline = float(previous["Entorhinal"]) - float(current["Entorhinal"])

        has_clinical_worsening = (
            signals[visit] >= config.conversion_min_progression_signal
            or recent_memory_decline >= 0.05
            or recent_executive_decline >= 0.05
            or recent_ventricle_increase >= 500.0
            or recent_hippocampus_decline >= 50.0
            or recent_entorhinal_decline >= 30.0
        )

        if has_clinical_worsening:
            candidates.append(visit)

    return np.asarray(candidates, dtype=int)


def latent_progression_path(
    rows: Sequence[Dict[str, object]],
    rng: np.random.Generator,
    config: TrialConfig,
) -> np.ndarray:
    """
    Generate a hidden disease progression path for one patient.

    The path is not exported. It links the ALPACA trajectory to diagnosis timing
    while preserving stochasticity and avoiding a deterministic threshold label.
    """

    n_visits = len(rows)
    frailty = rng.normal(0.0, config.latent_frailty_sd)
    slope = max(0.0, rng.normal(config.latent_progression_mean, config.latent_progression_sd))
    innovations = rng.normal(0.0, config.latent_innovation_sd, size=n_visits).cumsum()
    signals = clinical_worsening_signal(rows)
    time_axis = np.arange(n_visits, dtype=float)

    return frailty + slope * time_axis + 0.75 * signals + innovations


def sample_conversion_visit(
    rows: Sequence[Dict[str, object]],
    rng: np.random.Generator,
    config: TrialConfig,
) -> Optional[int]:
    """
    Sample an incident dementia conversion visit for a target converter.

    The hazard for AD at visit t uses information available up to t-1, a hidden
    patient-level progression path, treatment exposure, age, and random noise.
    The accepted visit must also be clinically plausible in the ALPACA vitals.
    """

    candidates = conversion_visit_candidates(rows, config)
    if candidates.size == 0:
        return None

    latent_path = latent_progression_path(rows, rng, config)
    signals = clinical_worsening_signal(rows)
    candidate_weights: List[float] = []

    for visit in candidates:
        prior_visit = max(0, visit - 1)
        prior_row = rows[prior_visit]
        medication_effect = config.diagnosis_treatment_log_odds * float(prior_row["medication"])
        age_component = config.diagnosis_age_weight * ((float(prior_row["subject_age"]) - 75.0) / 10.0)
        stochastic_noise = rng.normal(0.0, config.diagnosis_noise_sd)
        logit = (
            config.diagnosis_hazard_intercept
            + latent_path[prior_visit]
            + config.diagnosis_progression_weight * max(0.0, signals[prior_visit])
            + 0.45 * max(0.0, signals[visit])
            + age_component
            + medication_effect
            + stochastic_noise
        )
        candidate_weights.append(sigmoid(logit))

    weights = np.asarray(candidate_weights, dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)

    weights = weights / weights.sum()
    return int(rng.choice(candidates, p=weights))


def apply_dementia_labels(
    rows: List[Dict[str, object]],
    target_dementia: int,
    rng: np.random.Generator,
    config: TrialConfig,
) -> Optional[List[Dict[str, object]]]:
    """
    Apply monotonic, time-varying AD labels to one patient trajectory.

    Non-converters remain AD = 0 at all visits. Target converters must start at
    AD = 0 and switch to AD = 1 at a later clinically plausible visit.
    """

    if target_dementia not in {0, 1}:
        raise ValueError("target_dementia must be 0 or 1.")

    if target_dementia == 0:
        for row in rows:
            row["AD"] = 0
        return rows

    conversion_visit = sample_conversion_visit(rows, rng, config)
    if conversion_visit is None:
        return None

    for row in rows:
        row["AD"] = int(int(row["visit"]) >= conversion_visit)

    if rows[0]["AD"] != 0 or rows[-1]["AD"] != 1:
        return None

    return rows


# ============================================================
# VISIT GENERATION
# ============================================================

def sample_visit_count(rng: np.random.Generator, config: TrialConfig) -> int:
    """
    Sample the number of visits for one patient.

    The generated data is long format: each row is one visit, and every patient
    contributes between ``config.min_visits`` and ``config.max_visits`` rows.

    Returns
    -------
    int
        Number of visits for the patient.
    """

    if config.min_visits < 1:
        raise ValueError("min_visits must be at least 1.")

    if config.max_visits < config.min_visits:
        raise ValueError("max_visits must be greater than or equal to min_visits.")

    return int(rng.integers(config.min_visits, config.max_visits + 1))


# ============================================================
# ROW CONSTRUCTION
# ============================================================

def observation_to_row(
    obs: np.ndarray,
    patient_id: str,
    arm: str,
    visit_number: int,
    medication: int,
    ad: int,
    visit_interval_months: int,
) -> Dict[str, object]:
    """
    Convert one ALPACA observation into one row of the output dataset.

    Parameters
    ----------
    obs:
        ALPACA observation vector.

    patient_id:
        Synthetic patient identifier.

    arm:
        Trial arm name.

    visit_number:
        Visit index from 0 up to config.max_visits - 1.

    medication:
        0 for placebo / no statin.
        1 for statin treatment.

    ad:
        Time-varying dementia diagnosis at this visit.
        0 = not diagnosed
        1 = diagnosed

    visit_interval_months:
        Number of months between visits.

    Returns
    -------
    Dict[str, object]
        One structured row for the final CSV file.
    """

    return {
        "patient_id": patient_id,
        "arm": arm,
        "visit": visit_number,
        "month": visit_number * visit_interval_months,
        "Gap": 0 if visit_number == 0 else visit_interval_months,
        "AD": ad,
        "Gender": decode_gender(obs),
        "Race": decode_race(obs),
        "subject_age": get_feature_value(obs, "subject_age"),
        "TAU_data": get_feature_value(obs, "TAU_data"),
        "ABETA": get_feature_value(obs, "ABETA"),
        "ADNI_MEM": get_feature_value(obs, "ADNI_MEM"),
        "ADNI_EF2": get_feature_value(obs, "ADNI_EF2"),
        "Ventricles": get_feature_value(obs, "Ventricles"),
        "Hippocampus": get_feature_value(obs, "Hippocampus"),
        "WholeBrain": get_feature_value(obs, "WholeBrain"),
        "Entorhinal": get_feature_value(obs, "Entorhinal"),
        "Fusiform": get_feature_value(obs, "Fusiform"),
        "MidTemp": get_feature_value(obs, "MidTemp"),
        "ICV": get_feature_value(obs, "ICV"),
        "medication": medication,
    }


# ============================================================
# PATIENT SIMULATION
# ============================================================

def simulate_patient(
    env: ALPACAEnv,
    patient_id: str,
    arm: str,
    action: np.ndarray,
    medication: int,
    target_dementia: int,
    n_visits: int,
    rng: np.random.Generator,
    config: TrialConfig,
) -> Optional[List[Dict[str, object]]]:
    """
    Simulate one synthetic patient's longitudinal trajectory.

    ALPACA generates the observed clinical measurements. The AD labels are added
    afterward by the stochastic dementia progression layer.

    Parameters
    ----------
    env:
        ALPACA environment.

    patient_id:
        Synthetic patient identifier.

    arm:
        Trial arm name.

    action:
        Medication action vector.

    medication:
        Output medication indicator:
            0 = no statin
            1 = statin

    target_dementia:
        Patient-level final dementia target:
            0 = remains undiagnosed at all visits
            1 = must convert from AD = 0 to AD = 1 during follow-up

    n_visits:
        Number of observed visits for this patient.

    rng:
        Random number generator.

    config:
        Trial configuration.

    Returns
    -------
    Optional[List[Dict[str, object]]]
        List of visit rows. Returns None if ALPACA terminates the episode early.
    """

    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    rows = [
        observation_to_row(
            obs=obs,
            patient_id=patient_id,
            arm=arm,
            visit_number=0,
            medication=medication,
            ad=0,
            visit_interval_months=config.visit_interval_months,
        )
    ]

    for visit in range(1, n_visits):
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            return None

        rows.append(
            observation_to_row(
                obs=obs,
                patient_id=patient_id,
                arm=arm,
                visit_number=visit,
                medication=medication,
                ad=0,
                visit_interval_months=config.visit_interval_months,
            )
        )

    return apply_dementia_labels(
        rows=rows,
        target_dementia=target_dementia,
        rng=rng,
        config=config,
    )


def simulate_normal_mixed_patient(
    env: ALPACAEnv,
    patient_id: str,
    target_dementia: int,
    rng: np.random.Generator,
    config: TrialConfig,
) -> Optional[List[Dict[str, object]]]:
    """
    Simulate one patient for the Normal_Mixed arm.

    A random 30% are assigned as statin users by default. For statin users, the
    statin start visit is chosen randomly from the patient's observed visits.
    Before statin starts, the patient follows the placebo/no-medication action.
    AD labels are added after the ALPACA trajectory is generated.
    """

    n_visits = sample_visit_count(rng, config)
    is_statin_user = rng.random() < config.normal_arm_statin_probability
    statin_start_visit = int(rng.integers(0, n_visits)) if is_statin_user else None

    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    rows = [
        observation_to_row(
            obs=obs,
            patient_id=patient_id,
            arm="Normal_Mixed",
            visit_number=0,
            medication=1 if statin_start_visit == 0 else 0,
            ad=0,
            visit_interval_months=config.visit_interval_months,
        )
    ]

    for visit in range(1, n_visits):
        medication = 1 if statin_start_visit is not None and visit >= statin_start_visit else 0
        action = make_statin_action() if medication == 1 else make_placebo_action()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            return None

        rows.append(
            observation_to_row(
                obs=obs,
                patient_id=patient_id,
                arm="Normal_Mixed",
                visit_number=visit,
                medication=medication,
                ad=0,
                visit_interval_months=config.visit_interval_months,
            )
        )

    return apply_dementia_labels(
        rows=rows,
        target_dementia=target_dementia,
        rng=rng,
        config=config,
    )


# ============================================================
# ARM GENERATION
# ============================================================

def generate_arm(
    arm_name: str,
    action: np.ndarray,
    medication: int,
    config: TrialConfig,
) -> pd.DataFrame:
    """
    Generate one complete clinical trial arm.

    This function is shared by:
        - generate_placebo_arm
        - generate_treatment_arm

    Parameters
    ----------
    arm_name:
        Name of the trial arm.

    action:
        ALPACA medication action vector to apply.

    medication:
        Output medication indicator:
            0 = no statin
            1 = statin

    config:
        Trial configuration.

    Returns
    -------
    pd.DataFrame
        Longitudinal synthetic trial dataset for one arm.
    """

    rng = np.random.default_rng(config.random_seed + medication)
    env = create_alpaca_env(
        visit_interval_months=config.visit_interval_months,
        cohort_type=config.alpaca_cohort_type,
    )

    all_rows: List[Dict[str, object]] = []
    generated_patients = 0
    attempts = 0
    dementia_assignments = target_dementia_assignments(config, rng)

    while generated_patients < config.n_patients:
        attempts += 1

        if attempts > config.max_resampling_attempts:
            env.close()
            raise RuntimeError(
                f"Failed to generate {config.n_patients} patients for {arm_name}. "
                "Try increasing max_resampling_attempts, lowering "
                "conversion_min_progression_signal, or changing the ALPACA cohort."
            )

        n_visits = sample_visit_count(rng, config)
        patient_id = f"{arm_name}_{generated_patients + 1:04d}"
        target_dementia = int(dementia_assignments[generated_patients])

        rows = simulate_patient(
            env=env,
            patient_id=patient_id,
            arm=arm_name,
            action=action,
            medication=medication,
            target_dementia=target_dementia,
            n_visits=n_visits,
            rng=rng,
            config=config,
        )

        if rows is None:
            continue

        all_rows.extend(rows)
        generated_patients += 1

    env.close()

    return pd.DataFrame(all_rows)


# ============================================================
# PUBLIC FUNCTIONS REQUIRED BY USER
# ============================================================

def generate_placebo_arm(config: TrialConfig = TrialConfig()) -> pd.DataFrame:
    """
    Public function to generate the placebo arm.

    Placebo arm definition:
        - target_dementia_probability controls how many patients convert
        - converters start AD = 0 and later switch to AD = 1
        - non-converters remain AD = 0
        - no statin exposure
        - medication column = 0

    Returns
    -------
    pd.DataFrame
        Placebo arm dataset.
    """

    return generate_arm(
        arm_name="Placebo",
        action=make_placebo_action(),
        medication=0,
        config=config,
    )


def generate_treatment_arm(config: TrialConfig = TrialConfig()) -> pd.DataFrame:
    """
    Public function to generate the statin treatment arm.

    Treatment arm definition:
        - target_dementia_probability controls how many patients convert
        - converters start AD = 0 and later switch to AD = 1
        - non-converters remain AD = 0
        - statin active at each simulated transition
        - medication column = 1

    Returns
    -------
    pd.DataFrame
        Statin treatment arm dataset.
    """

    return generate_arm(
        arm_name="Statin_Treatment",
        action=make_statin_action(),
        medication=1,
        config=config,
    )


def generate_normal_arm(config: TrialConfig = TrialConfig()) -> pd.DataFrame:
    """
    Public function to generate the Normal_Mixed arm.

    Normal_Mixed arm definition:
        - target_dementia_probability controls how many patients convert
        - converters start AD = 0 and later switch to AD = 1
        - non-converters remain AD = 0
        - 30% statin users by default
        - 70% no-statin users by default
        - statin users start statin at a random observed visit
    """

    if not 0.0 <= config.normal_arm_statin_probability <= 1.0:
        raise ValueError("normal_arm_statin_probability must be between 0 and 1.")

    rng = np.random.default_rng(config.random_seed + 2)
    env = create_alpaca_env(
        visit_interval_months=config.visit_interval_months,
        cohort_type=config.alpaca_cohort_type,
    )

    all_rows: List[Dict[str, object]] = []
    generated_patients = 0
    attempts = 0
    dementia_assignments = target_dementia_assignments(config, rng)

    while generated_patients < config.n_patients:
        attempts += 1

        if attempts > config.max_resampling_attempts:
            env.close()
            raise RuntimeError(
                f"Failed to generate {config.n_patients} patients for Normal_Mixed. "
                "Try increasing max_resampling_attempts, lowering "
                "conversion_min_progression_signal, or changing the ALPACA cohort."
            )

        patient_id = f"Normal_Mixed_{generated_patients + 1:04d}"
        target_dementia = int(dementia_assignments[generated_patients])

        rows = simulate_normal_mixed_patient(
            env=env,
            patient_id=patient_id,
            target_dementia=target_dementia,
            rng=rng,
            config=config,
        )

        if rows is None:
            continue

        all_rows.extend(rows)
        generated_patients += 1

    env.close()

    return pd.DataFrame(all_rows)


# ============================================================
# REPORTING
# ============================================================

def summarize_arm(df: pd.DataFrame) -> Dict[str, object]:
    """
    Summarize one generated arm at patient level.

    A statin user is any patient with medication = 1 at one or more visits.
    A patient who developed dementia is represented by a monotonic transition
    from AD = 0 to AD = 1 during follow-up.
    """

    ordered = df.sort_values(["patient_id", "visit"]).copy()
    grouped = ordered.groupby("patient_id", sort=False)
    ad_transition_counts = grouped["AD"].apply(
        lambda values: int((values.diff().fillna(0) > 0).sum())
    )
    non_monotone_counts = grouped["AD"].apply(
        lambda values: int((values.diff().fillna(0) < 0).sum())
    )

    patient_summary = (
        ordered.groupby("patient_id", as_index=False)
        .agg(
            arm=("arm", "first"),
            developed_dementia=("AD", "max"),
            baseline_dementia=("AD", "first"),
            statin_user=("medication", "max"),
            n_visits=("visit", "count"),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
    )
    patient_summary["ad_transition_count"] = patient_summary["patient_id"].map(ad_transition_counts)
    patient_summary["non_monotone_ad_count"] = patient_summary["patient_id"].map(non_monotone_counts)

    statin_user = patient_summary["statin_user"] == 1
    developed_dementia = patient_summary["developed_dementia"] == 1
    baseline_dementia = patient_summary["baseline_dementia"] == 1
    incident_dementia = developed_dementia & ~baseline_dementia
    non_statin_user = ~statin_user
    no_dementia = ~developed_dementia
    transition_patients = patient_summary["ad_transition_count"] > 0
    non_monotone_patients = patient_summary["non_monotone_ad_count"] > 0

    conversion_months = {}
    for patient_id, visits in grouped:
        ad_values = visits["AD"].to_numpy(dtype=int)
        if ad_values[0] == 0 and np.any(ad_values == 1):
            first_positive_idx = int(np.argmax(ad_values == 1))
            conversion_months[patient_id] = int(visits.iloc[first_positive_idx]["month"])

    conversion_month_values = np.asarray(list(conversion_months.values()), dtype=float)

    return {
        "rows": int(len(df)),
        "patients": int(len(patient_summary)),
        "statin_users": int(statin_user.sum()),
        "non_statin_users": int(non_statin_user.sum()),
        "patients_developed_dementia": int(developed_dementia.sum()),
        "patients_without_dementia": int(no_dementia.sum()),
        "patients_with_baseline_dementia": int(baseline_dementia.sum()),
        "patients_with_incident_dementia": int(incident_dementia.sum()),
        "patients_with_ad_transition": int(transition_patients.sum()),
        "patients_with_non_monotone_ad": int(non_monotone_patients.sum()),
        "statin_users_developed_dementia": int((statin_user & developed_dementia).sum()),
        "statin_users_without_dementia": int((statin_user & no_dementia).sum()),
        "non_statin_users_developed_dementia": int((non_statin_user & developed_dementia).sum()),
        "non_statin_users_without_dementia": int((non_statin_user & no_dementia).sum()),
        "mean_conversion_month": (
            float(conversion_month_values.mean()) if conversion_month_values.size else None
        ),
        "median_conversion_month": (
            float(np.median(conversion_month_values)) if conversion_month_values.size else None
        ),
        "min_visits_per_patient": int(patient_summary["n_visits"].min()),
        "max_visits_per_patient": int(patient_summary["n_visits"].max()),
        "mean_visits_per_patient": float(patient_summary["n_visits"].mean()),
        "median_visits_per_patient": float(patient_summary["n_visits"].median()),
        "mean_follow_up_months": float(patient_summary["last_month"].mean()),
        "max_follow_up_months": int(patient_summary["last_month"].max()),
        "visit_count_distribution": {
            int(visit_count): int(patient_count)
            for visit_count, patient_count in patient_summary["n_visits"]
            .value_counts()
            .sort_index()
            .items()
        },
        "conversion_month_distribution": {
            int(month): int(count)
            for month, count in pd.Series(conversion_months)
            .value_counts()
            .sort_index()
            .items()
        },
    }


def build_statistics_report(
    arm_datasets: Dict[str, pd.DataFrame],
    config: TrialConfig,
) -> str:
    """
    Build a plain-text report describing the generated cohort datasets.
    """

    combined_df = pd.concat(arm_datasets.values(), ignore_index=True)
    report_datasets = {**arm_datasets, "All arms combined": combined_df}

    lines = [
        "Synthetic ALPACA Cohort Generation Statistics",
        "============================================",
        "",
        "Initial Simulation Parameters",
        "-----------------------------",
        f"Patients per arm target: {config.n_patients}",
        f"Total target patients across arms: {config.n_patients * len(arm_datasets)}",
        f"Arms generated: {', '.join(arm_datasets.keys())}",
        f"ALPACA cohort type: {config.alpaca_cohort_type}",
        f"ALPACA reward metric: ADNI_MEM",
        f"Visit count range per patient: {config.min_visits} to {config.max_visits}",
        f"Visit interval: {config.visit_interval_months} months",
        f"Target final dementia probability: {config.target_dementia_probability}",
        f"Earliest conversion visit index: {config.min_conversion_visit}",
        f"Minimum conversion progression signal: {config.conversion_min_progression_signal}",
        f"Latent frailty SD: {config.latent_frailty_sd}",
        f"Latent progression mean: {config.latent_progression_mean}",
        f"Latent progression SD: {config.latent_progression_sd}",
        f"Latent innovation SD: {config.latent_innovation_sd}",
        f"Diagnosis hazard intercept: {config.diagnosis_hazard_intercept}",
        f"Diagnosis progression weight: {config.diagnosis_progression_weight}",
        f"Diagnosis age weight: {config.diagnosis_age_weight}",
        f"Diagnosis treatment log-odds: {config.diagnosis_treatment_log_odds}",
        f"Diagnosis noise SD: {config.diagnosis_noise_sd}",
        f"Normal_Mixed statin probability: {config.normal_arm_statin_probability}",
        f"Random seed: {config.random_seed}",
        f"Maximum resampling attempts: {config.max_resampling_attempts}",
        "",
        "Dementia Progression Architecture",
        "---------------------------------",
        "1. ALPACA_DT_Sim generates the observed medical trajectory: cognition, CSF biomarkers, MRI volumes, demographics, and treatment-conditioned transitions.",
        "2. The AD label is not read from ALPACA and is not assigned by a deterministic same-row threshold.",
        "3. Each patient first receives a final dementia target. Target non-converters remain AD = 0 across observed follow-up.",
        "4. Target converters must start as AD = 0 and convert to AD = 1 at a later observed visit.",
        "5. Conversion timing is sampled from a hidden stochastic process using patient frailty, latent progression, prior clinical worsening, age, treatment exposure, and noise.",
        "6. A conversion visit is accepted only if the surrounding ALPACA measurements show plausible deterioration in cognition or brain volumes.",
        "7. Once AD becomes 1, it remains 1 for all later visits. Non-monotone AD trajectories are considered generation errors.",
        "8. Hidden latent severity and conversion timing are not exported to the CSV and should not be used as model predictors.",
        "",
        "Simulation Sample Generation",
        "----------------------------",
        "1. The script creates an ALPACA_DT_Sim environment using the configured baseline cohort.",
        "2. The number of observed visits is sampled independently for each patient between the configured minimum and maximum visit counts.",
        "3. Placebo patients receive the No Medication action at each transition, while Statin_Treatment patients receive the Statin action at each transition.",
        "4. In the Normal_Mixed arm, a patient-level statin-user flag is sampled from the configured probability; statin users start at a random observed visit.",
        "5. ALPACA trajectories that terminate early or target-converter trajectories with no plausible conversion window are resampled.",
        "6. Each row in the CSV is one observed patient visit. Missing visits are represented by shorter patient trajectories rather than empty rows.",
        "",
        "Definitions",
        "-----------",
        "Statin user: patient with medication = 1 in at least one visit.",
        "Developed dementia: patient with AD = 1 in at least one generated visit.",
        "Incident dementia: patient whose first visit has AD = 0 and at least one later visit has AD = 1.",
        "AD transition: a within-patient 0 -> 1 change in the AD column.",
        "",
    ]

    for label, df in report_datasets.items():
        summary = summarize_arm(df)
        lines.extend(
            [
                label,
                "-" * len(label),
                f"Rows: {summary['rows']}",
                f"Number of patients: {summary['patients']}",
                f"Number of statin users: {summary['statin_users']}",
                f"Number of non-statin users: {summary['non_statin_users']}",
                f"Number of patients who developed dementia: {summary['patients_developed_dementia']}",
                f"Number of patients without dementia: {summary['patients_without_dementia']}",
                f"Number of patients with baseline dementia: {summary['patients_with_baseline_dementia']}",
                f"Number of patients with incident dementia: {summary['patients_with_incident_dementia']}",
                f"Number of patients with AD 0->1 transition: {summary['patients_with_ad_transition']}",
                f"Number of patients with non-monotone AD: {summary['patients_with_non_monotone_ad']}",
                f"Number of statin users who developed dementia: {summary['statin_users_developed_dementia']}",
                f"Number of statin users who did not develop dementia: {summary['statin_users_without_dementia']}",
                f"Number of non-statin users who developed dementia: {summary['non_statin_users_developed_dementia']}",
                f"Number of non-statin users who did not develop dementia: {summary['non_statin_users_without_dementia']}",
                f"Mean conversion month: {summary['mean_conversion_month'] if summary['mean_conversion_month'] is not None else 'N/A'}",
                f"Median conversion month: {summary['median_conversion_month'] if summary['median_conversion_month'] is not None else 'N/A'}",
                f"Minimum visits per patient: {summary['min_visits_per_patient']}",
                f"Maximum visits per patient: {summary['max_visits_per_patient']}",
                f"Mean visits per patient: {summary['mean_visits_per_patient']:.2f}",
                f"Median visits per patient: {summary['median_visits_per_patient']:.2f}",
                f"Mean follow-up months: {summary['mean_follow_up_months']:.2f}",
                f"Maximum follow-up months: {summary['max_follow_up_months']}",
                "Visit count distribution:",
            ]
        )

        for visit_count, patient_count in summary["visit_count_distribution"].items():
            lines.append(f"  {visit_count} visits: {patient_count} patients")

        lines.append("Conversion month distribution:")
        if summary["conversion_month_distribution"]:
            for month, patient_count in summary["conversion_month_distribution"].items():
                lines.append(f"  month {month}: {patient_count} patients")
        else:
            lines.append("  N/A")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_trial_arms(
    placebo_path: str | Path = SCRIPT_DIR / "placebo_arm.csv",
    treatment_path: str | Path = SCRIPT_DIR / "statin_treatment_arm.csv",
    normal_path: str | Path = SCRIPT_DIR / "normal_arm.csv",
    statistics_path: str | Path = SCRIPT_DIR / "cohort_generation_statistics.txt",
    config: TrialConfig = TrialConfig(),
) -> None:
    """
    Generate all trial arms and save them as separate CSV files.

    Parameters
    ----------
    placebo_path:
        Output path for placebo arm CSV.

    treatment_path:
        Output path for statin treatment arm CSV.

    normal_path:
        Output path for Normal_Mixed arm CSV.

    statistics_path:
        Output path for the text statistics report.

    config:
        Trial configuration.
    """

    placebo_path = Path(placebo_path)
    treatment_path = Path(treatment_path)
    normal_path = Path(normal_path)
    statistics_path = Path(statistics_path)

    placebo_df = generate_placebo_arm(config)
    treatment_df = generate_treatment_arm(config)
    normal_df = generate_normal_arm(config)

    placebo_path.parent.mkdir(parents=True, exist_ok=True)
    treatment_path.parent.mkdir(parents=True, exist_ok=True)
    normal_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)

    placebo_df.to_csv(placebo_path, index=False)
    treatment_df.to_csv(treatment_path, index=False)
    normal_df.to_csv(normal_path, index=False)

    statistics_report = build_statistics_report(
        arm_datasets={
            "Placebo": placebo_df,
            "Statin_Treatment": treatment_df,
            "Normal_Mixed": normal_df,
        },
        config=config,
    )
    statistics_path.write_text(statistics_report, encoding="utf-8")

    print(f"Saved placebo arm: {placebo_path}")
    print(f"Rows: {len(placebo_df)} | Patients: {placebo_df['patient_id'].nunique()}")

    print(f"Saved treatment arm: {treatment_path}")
    print(f"Rows: {len(treatment_df)} | Patients: {treatment_df['patient_id'].nunique()}")

    print(f"Saved normal mixed arm: {normal_path}")
    print(f"Rows: {len(normal_df)} | Patients: {normal_df['patient_id'].nunique()}")

    print(f"Saved statistics report: {statistics_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    config = TrialConfig(
        n_patients=2000,
        max_visits=10,
        min_visits=2,
        visit_interval_months=12,
        alpaca_cohort_type="all",
        target_dementia_probability=0.50,
        min_conversion_visit=1,
        conversion_min_progression_signal=0.05,
        random_seed=42,
        max_resampling_attempts=50000,
    )

    save_trial_arms(
        placebo_path=SCRIPT_DIR / "placebo_arm.csv",
        treatment_path=SCRIPT_DIR / "statin_treatment_arm.csv",
        normal_path=SCRIPT_DIR / "normal_arm.csv",
        statistics_path=SCRIPT_DIR / "cohort_generation_statistics.txt",
        config=config,
    )
