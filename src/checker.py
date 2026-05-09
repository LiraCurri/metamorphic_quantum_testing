from typing import Dict, Tuple, Any
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd


# CIRCUIT-LEVEL CHECKING

def counts_to_probabilities(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert measurement counts to a normalized probability distribution.
    """
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {state: count / total for state, count in counts.items()}


def distribution_distance(
    counts1: Dict[str, int],
    counts2: Dict[str, int],
) -> float:
    """
    Total Variation Distance (TVD) between two measurement distributions.
    TVD = 0.5 * sum_i |p_i - q_i|
    """
    p1 = counts_to_probabilities(counts1)
    p2 = counts_to_probabilities(counts2)

    states = set(p1.keys()) | set(p2.keys())
    return 0.5 * sum(abs(p1.get(s, 0.0) - p2.get(s, 0.0)) for s in states)


def check_violation(
    counts1: Dict[str, int],
    counts2: Dict[str, int],
    tvd_threshold: float = 0.05,
    chi_alpha: float = 0.05
) -> Tuple[bool, float, bool, float, float]:
    """
    Check a circuit-level metamorphic relation using both TVD and chi-square.

    Returns:
        (
            tvd_violation,
            tvd_distance,
            chi_violation,
            p_value,
            chi2_stat
        )
    """
    dist = distribution_distance(counts1, counts2)
    tvd_violation = dist > tvd_threshold

    states = sorted(set(counts1.keys()) | set(counts2.keys()))
    obs1 = [counts1.get(s, 0) for s in states]
    obs2 = [counts2.get(s, 0) for s in states]
    observed = np.array([obs1, obs2])

    if observed.sum() == 0 or len(states) < 2:
        chi2_stat = 0.0
        p_value = 1.0
        chi_violation = False
    else:
        chi2_stat, p_value, _, _ = chi2_contingency(observed)
        chi_violation = p_value < chi_alpha

    return tvd_violation, dist, chi_violation, p_value, chi2_stat


def print_comparison(
    counts1: Dict[str, int],
    counts2: Dict[str, int],
) -> None:
    """
    Print side-by-side probability distributions for debugging.
    """
    p1 = counts_to_probabilities(counts1)
    p2 = counts_to_probabilities(counts2)

    states = sorted(set(p1.keys()) | set(p2.keys()))

    print("\nState | Source Prob | Follow-up Prob | Diff")
    print("--------------------------------------------")
    for s in states:
        prob1 = p1.get(s, 0.0)
        prob2 = p2.get(s, 0.0)
        diff = abs(prob1 - prob2)
        print(f"{s:>5} | {prob1:11.4f} | {prob2:14.4f} | {diff:6.4f}")


# VQE TRACE CHECKING

def compare_vqe_traces(source_trace: pd.DataFrame, follow_trace: pd.DataFrame) -> Dict[str, float]:
    """
    Compare VQE optimization traces.

    Handles unequal parameter-vector lengths by comparing only the shared prefix.
    """
    n = min(len(source_trace), len(follow_trace))
    if n == 0:
        raise ValueError("Cannot compare empty traces.")

    source_trace = source_trace.iloc[:n].reset_index(drop=True)
    follow_trace = follow_trace.iloc[:n].reset_index(drop=True)

    energy_diffs = []
    sym_diffs = []
    param_diffs = []
    param_dim_mismatch = False

    for i in range(n):
        src = source_trace.iloc[i]
        fol = follow_trace.iloc[i]

        energy_diffs.append(abs(float(src["energy"]) - float(fol["energy"])))
        sym_diffs.append(abs(float(src["symmetry"]) - float(fol["symmetry"])))

        src_params = np.asarray(src["params"], dtype=float)
        fol_params = np.asarray(fol["params"], dtype=float)

        if len(src_params) != len(fol_params):
            param_dim_mismatch = True

        k = min(len(src_params), len(fol_params))
        param_diffs.append(np.linalg.norm(src_params[:k] - fol_params[:k]))

    return {
        "avg_energy_diff": float(np.mean(energy_diffs)),
        "max_energy_diff": float(np.max(energy_diffs)),
        "avg_sym_diff": float(np.mean(sym_diffs)),
        "max_sym_diff": float(np.max(sym_diffs)),
        "avg_param_diff": float(np.mean(param_diffs)),
        "max_param_diff": float(np.max(param_diffs)),
        "trace_len": int(n),
        "param_dim_mismatch": bool(param_dim_mismatch),
    }

def check_vqe_violation(
    source_result: Any,
    follow_result: Any,
    trace_metrics: Dict[str, float],
    energy_threshold: float = 1e-3,
    symmetry_threshold: float = 1e-3,
    param_threshold: float = 1e-2,
) -> Tuple[bool, Dict[str, float]]:
    """
    Decide if a VQE metamorphic relation is violated.

    Uses a composite oracle:
    - final energy difference
    - maximum symmetry deviation along the path
    - maximum parameter deviation along the path

    Also returns separate flags so we can see which part of the oracle
    caused the violation.
    """
    delta_E = abs(float(source_result.fun) - float(follow_result.fun))

    energy_violation = delta_E > energy_threshold
    symmetry_violation = trace_metrics["max_sym_diff"] > symmetry_threshold
    param_violation = trace_metrics["max_param_diff"] > param_threshold

    violation = (
        energy_violation
        or symmetry_violation
        or param_violation
    )

    metrics = {
        "delta_E": float(delta_E),
        "energy_violation": bool(energy_violation),
        "symmetry_violation": bool(symmetry_violation),
        "param_violation": bool(param_violation),
        **trace_metrics,
    }

    return violation, metrics


def evaluate_vqe_pair(
    source_result,
    follow_result,
    source_trace: pd.DataFrame,
    follow_trace: pd.DataFrame,
    energy_threshold: float = 1e-3,
    symmetry_threshold: float = 1e-3,
    param_threshold: float = 1e-2,
) -> Dict[str, float]:
    """
    Full evaluation pipeline for a VQE source/follow-up pair.
    """
    trace_metrics = compare_vqe_traces(source_trace, follow_trace)
    violation, metrics = check_vqe_violation(
        source_result,
        follow_result,
        trace_metrics,
        energy_threshold=energy_threshold,
        symmetry_threshold=symmetry_threshold,
        param_threshold=param_threshold,
    )

    return {
        "violation": violation,
        **metrics,
    }

def evaluate_vqe_pair_shot_based(
    source_result,
    follow_result,
    source_trace,
    follow_trace,
    energy_threshold: float = 5e-2,
    symmetry_threshold: float = 1e-2,
):
    trace_metrics = compare_vqe_traces(source_trace, follow_trace)

    delta_E = abs(float(source_result.fun) - float(follow_result.fun))

    energy_violation = delta_E > energy_threshold
    symmetry_violation = trace_metrics["avg_sym_diff"] > symmetry_threshold

    violation = energy_violation or symmetry_violation

    return {
        "violation": bool(violation),
        "delta_E": float(delta_E),
        "energy_violation": bool(energy_violation),
        "symmetry_violation": bool(symmetry_violation),
        **trace_metrics,
    }


def evaluate_vqe_pair_noisy(
    source_result,
    follow_result,
    source_trace: pd.DataFrame,
    follow_trace: pd.DataFrame,
    energy_threshold: float,
    symmetry_threshold: float,
):
    """
    Noisy oracle:
    - final energy difference
    - average symmetry deviation
    Parameter-path is recorded but not used as active criterion.
    """
    trace_metrics = compare_vqe_traces(source_trace, follow_trace)

    delta_E = abs(float(source_result.fun) - float(follow_result.fun))

    energy_violation = delta_E > energy_threshold
    symmetry_violation = trace_metrics["avg_sym_diff"] > symmetry_threshold

    violation = energy_violation or symmetry_violation

    return {
        "violation": bool(violation),
        "delta_E": float(delta_E),
        "energy_violation": bool(energy_violation),
        "symmetry_violation": bool(symmetry_violation),
        **trace_metrics,
    }
