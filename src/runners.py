from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
    thermal_relaxation_error,
)

import numpy as np
import pandas as pd

from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from qiskit_algorithms.utils import algorithm_globals
try:
    from qiskit_algorithms.optimizers import SPSA
except ImportError:
    from qiskit.algorithms.optimizers import SPSA

# CIRCUIT EXECUTION

def run_circuit(
    qc,
    shots: int = 10000,
    optimization_level: int = 0,
    seed: int | None = None,
    noise_model=None,
):
    """
    Run a quantum circuit on the Qiskit Aer simulator with reproducibility.

    Seed support ensures fair metamorphic comparisons between source
    and follow-up circuits.
    """
    simulator = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()

    if seed is not None:
        simulator.set_options(seed_simulator=seed)

    qc_transpiled = transpile(
        qc,
        backend=simulator,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    if qc_transpiled.num_clbits == 0:
        qc_transpiled = qc_transpiled.copy()
        qc_transpiled.measure_all()

    job = simulator.run(qc_transpiled, shots=shots)
    result = job.result()

    return result.get_counts()


# STATEVECTOR EXPECTATION FUNCTIONS

def energy_expectation(ansatz, params, hamiltonian):
    """
    Compute <psi(theta)|H|psi(theta)> using exact statevector simulation.
    Deterministic baseline.
    """
    bound = ansatz.assign_parameters(params)
    sv = Statevector.from_instruction(bound)
    return float(np.real(sv.expectation_value(hamiltonian)))


def symmetry_expectation(ansatz, params, observable):
    """
    Compute <psi(theta)|O|psi(theta)> using exact statevector simulation.
    """
    bound = ansatz.assign_parameters(params)
    sv = Statevector.from_instruction(bound)
    return float(np.real(sv.expectation_value(observable)))


def run_vqe_trace(
    ansatz,
    hamiltonian,
    symmetry_op,
    initial_point,
    maxiter: int = 120,
    method: str = "COBYLA",
):
    """
    Run a deterministic statevector VQE-style optimization and record trace.

    Returns:
        result: scipy OptimizeResult
        trace_df: DataFrame with energy, symmetry, parameter norm, parameters
    """
    trace = []
    eval_count = 0

    def objective(params):
        nonlocal eval_count
        eval_count += 1

        energy = energy_expectation(ansatz, params, hamiltonian)
        sym_val = symmetry_expectation(ansatz, params, symmetry_op)

        trace.append({
            "eval": eval_count,
            "energy": float(energy),
            "symmetry": float(sym_val),
            "param_norm": float(np.linalg.norm(params)),
            "params": np.array(params, dtype=float).copy(),
        })

        return energy

    result = minimize(
        objective,
        x0=np.array(initial_point, dtype=float),
        method=method,
        options={"maxiter": maxiter},
    )

    trace_df = pd.DataFrame(trace)
    return result, trace_df


def run_vqe_experiment(
    source_ansatz,
    follow_ansatz,
    hamiltonian,
    symmetry_op,
    initial_point,
    maxiter: int = 120,
):
    """
    Run paired source and follow-up deterministic VQE executions.
    """
    source_result, source_trace = run_vqe_trace(
        source_ansatz,
        hamiltonian,
        symmetry_op,
        initial_point,
        maxiter=maxiter,
    )

    follow_result, follow_trace = run_vqe_trace(
        follow_ansatz,
        hamiltonian,
        symmetry_op,
        initial_point,
        maxiter=maxiter,
    )

    return {
        "source_result": source_result,
        "follow_result": follow_result,
        "source_trace": source_trace,
        "follow_trace": follow_trace,
    }


# SHOT-BASED PAULI EXPECTATION ESTIMATION

def _pauli_label_to_qubit_ops(pauli_label: str):
    """
    Convert a Qiskit Pauli label into qubit-indexed Pauli operations.

    Qiskit labels are written left-to-right from highest qubit to lowest qubit.
    Example:
        label "ZZII" means:
            qubit 3: Z
            qubit 2: Z
            qubit 1: I
            qubit 0: I

    This function returns:
        {0: op_on_q0, 1: op_on_q1, ...}
    """
    n = len(pauli_label)
    return {q: pauli_label[n - 1 - q] for q in range(n)}


def _compatible_basis(basis_a: str, basis_b: str) -> bool:
    """
    Check if two Pauli strings can be measured in one shared basis.
    I is compatible with X, Y, or Z.
    """
    if len(basis_a) != len(basis_b):
        return False

    for a, b in zip(basis_a, basis_b):
        if a != "I" and b != "I" and a != b:
            return False

    return True


def _merge_basis(basis_a: str, basis_b: str) -> str:
    """
    Merge two compatible Pauli measurement bases.
    """
    if len(basis_a) != len(basis_b):
        raise ValueError("Cannot merge Pauli bases of different lengths.")

    merged = []

    for a, b in zip(basis_a, basis_b):
        if a == "I":
            merged.append(b)
        elif b == "I":
            merged.append(a)
        else:
            merged.append(a)

    return "".join(merged)


def _group_pauli_terms(observables: dict):
    """
    Group SparsePauliOp terms into compatible measurement bases.

    Input:
        observables = {
            "energy": hamiltonian,
            "symmetry": symmetry_op,
        }

    Returns:
        [
            {
                "basis": "ZZIIII",
                "terms": [("energy", "ZZIIII", coeff), ...]
            },
            ...
        ]
    """
    groups = []

    for obs_name, operator in observables.items():
        for pauli, coeff in zip(operator.paulis, operator.coeffs):
            label = pauli.to_label()

            placed = False

            for group in groups:
                if _compatible_basis(group["basis"], label):
                    group["basis"] = _merge_basis(group["basis"], label)
                    group["terms"].append((obs_name, label, coeff))
                    placed = True
                    break

            if not placed:
                groups.append({
                    "basis": label,
                    "terms": [(obs_name, label, coeff)],
                })

    return groups


def _build_measurement_circuit(ansatz, params, basis: str):
    """
    Build a measurement circuit for one Pauli measurement basis.

    For X measurement: apply H before Z-basis measurement.
    For Y measurement: apply Sdg then H before Z-basis measurement.
    For Z/I: no basis rotation needed.
    """
    bound = ansatz.assign_parameters(params)

    qc = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
    qc.compose(bound, inplace=True)

    qubit_ops = _pauli_label_to_qubit_ops(basis)

    for q, op in qubit_ops.items():
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.sdg(q)
            qc.h(q)
        elif op in ["Z", "I"]:
            pass
        else:
            raise ValueError(f"Unsupported Pauli operator: {op}")

    qc.measure(range(ansatz.num_qubits), range(ansatz.num_qubits))

    return qc


def _expectation_from_counts(counts, pauli_label: str, num_qubits: int) -> float:
    """
    Estimate expectation value of one Pauli string from measurement counts.
    """
    total_shots = sum(counts.values())

    if total_shots == 0:
        return 0.0

    expectation = 0.0
    qubit_ops = _pauli_label_to_qubit_ops(pauli_label)

    for bitstring, count in counts.items():
        eigenvalue = 1

        for q, op in qubit_ops.items():
            if op == "I":
                continue

            # Qiskit count bitstrings are classical bits c[n-1]...c[0].
            bit = bitstring[num_qubits - 1 - q]

            if bit == "1":
                eigenvalue *= -1

        expectation += eigenvalue * count

    return float(expectation / total_shots)


def estimate_observables_shots(
    ansatz,
    params,
    observables: dict,
    shots: int = 4096,
    seed: int | None = None,
    noise_model=None,
    optimization_level: int = 0,
):
    """
    Estimate SparsePauliOp observables using finite-shot measurement.

    Example:
        observables = {
            "energy": H,
            "symmetry": symmetry_op,
        }

    Returns:
        {
            "energy": estimated_energy,
            "symmetry": estimated_symmetry,
        }
    """
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()

    if seed is not None:
        backend.set_options(seed_simulator=seed)

    values = {name: 0.0 for name in observables.keys()}
    groups = _group_pauli_terms(observables)

    for group_index, group in enumerate(groups):
        basis = group["basis"]

        qc = _build_measurement_circuit(
            ansatz=ansatz,
            params=params,
            basis=basis,
        )

        qc_t = transpile(
            qc,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )

        job_seed = None if seed is None else seed + group_index

        job = backend.run(
            qc_t,
            shots=shots,
            seed_simulator=job_seed,
        )

        counts = job.result().get_counts()

        for obs_name, pauli_label, coeff in group["terms"]:
            exp_val = _expectation_from_counts(
                counts=counts,
                pauli_label=pauli_label,
                num_qubits=ansatz.num_qubits,
            )

            values[obs_name] += float(np.real(coeff)) * exp_val

    return values


def run_vqe_trace_shots(
    ansatz,
    hamiltonian,
    symmetry_op,
    initial_point,
    shots: int = 4096,
    maxiter: int = 120,
    seed: int | None = None,
    noise_model=None,
    optimization_level: int = 0,
):
    """
    Run finite-shot VQE and record a noisy optimization trace using SPSA.
    """
    trace = []
    eval_count = 0

    if seed is not None:
        algorithm_globals.random_seed = seed

    rng = np.random.default_rng(seed)

    observables = {
        "energy": hamiltonian,
        "symmetry": symmetry_op,
    }

    def objective(params):
        nonlocal eval_count
        eval_count += 1

        eval_seed = int(rng.integers(0, 2**32 - 1))

        obs_values = estimate_observables_shots(
            ansatz=ansatz,
            params=params,
            observables=observables,
            shots=shots,
            seed=eval_seed,
            noise_model=noise_model,
            optimization_level=optimization_level,
        )

        energy = obs_values["energy"]
        sym_val = obs_values["symmetry"]

        trace.append({
            "eval": eval_count,
            "energy": float(energy),
            "symmetry": float(sym_val),
            "param_norm": float(np.linalg.norm(params)),
            "params": np.array(params, dtype=float).copy(),
        })

        return energy

    spsa = SPSA(
        maxiter=maxiter,
        blocking=True,
        trust_region=True,
        last_avg=5,
        resamplings=1,
    )

    result = spsa.minimize(
        fun=objective,
        x0=np.array(initial_point, dtype=float),
    )

    trace_df = pd.DataFrame(trace)
    return result, trace_df

def run_vqe_experiment_shots(
    source_ansatz,
    follow_ansatz,
    hamiltonian,
    symmetry_op,
    initial_point,
    shots: int = 4096,
    maxiter: int = 120,
    seed: int | None = None,
    noise_model=None,
):
    """
    Run paired source and follow-up shot-based VQE executions.
    """
    source_result, source_trace = run_vqe_trace_shots(
        ansatz=source_ansatz,
        hamiltonian=hamiltonian,
        symmetry_op=symmetry_op,
        initial_point=initial_point,
        shots=shots,
        maxiter=maxiter,
        seed=None if seed is None else seed,
        noise_model=noise_model,
    )

    follow_result, follow_trace = run_vqe_trace_shots(
        ansatz=follow_ansatz,
        hamiltonian=hamiltonian,
        symmetry_op=symmetry_op,
        initial_point=initial_point,
        shots=shots,
        maxiter=maxiter,
        seed=None if seed is None else seed + 100000,
        noise_model=noise_model,
    )

    return {
        "source_result": source_result,
        "follow_result": follow_result,
        "source_trace": source_trace,
        "follow_trace": follow_trace,
    }


# CONTROLLED QISKIT NOISE MODELS

def build_depolarizing_noise_model(
    one_qubit_error_rate: float = 0.001,
    two_qubit_error_rate: float = 0.01,
):
    """
    Build a simple depolarizing noise model.
    """
    noise_model = NoiseModel()

    one_qubit_error = depolarizing_error(
        one_qubit_error_rate,
        1,
    )

    two_qubit_error = depolarizing_error(
        two_qubit_error_rate,
        2,
    )

    noise_model.add_all_qubit_quantum_error(
        one_qubit_error,
        ["x", "sx", "h", "rx", "ry", "rz"],
    )

    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        ["cx"],
    )

    return noise_model


def build_readout_noise_model(
    readout_error_rate: float = 0.02,
):
    """
    Build a symmetric readout-error model.
    """
    noise_model = NoiseModel()

    readout_error = ReadoutError([
        [1 - readout_error_rate, readout_error_rate],
        [readout_error_rate, 1 - readout_error_rate],
    ])

    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def build_thermal_relaxation_noise_model(
    t1: float = 50_000,
    t2: float = 70_000,
    one_qubit_gate_time: float = 50,
    two_qubit_gate_time: float = 300,
):
    """
    Build a simple thermal relaxation noise model.

    Times are in nanoseconds by convention here.
    """
    noise_model = NoiseModel()

    one_qubit_error = thermal_relaxation_error(
        t1,
        t2,
        one_qubit_gate_time,
    )

    two_qubit_error = thermal_relaxation_error(
        t1,
        t2,
        two_qubit_gate_time,
    ).expand(
        thermal_relaxation_error(
            t1,
            t2,
            two_qubit_gate_time,
        )
    )

    noise_model.add_all_qubit_quantum_error(
        one_qubit_error,
        ["x", "sx", "h", "rx", "ry", "rz"],
    )

    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        ["cx"],
    )

    return noise_model


def build_combined_noise_model(
    one_qubit_error_rate: float = 0.001,
    two_qubit_error_rate: float = 0.01,
    readout_error_rate: float = 0.02,
    include_thermal: bool = False,
    t1: float = 50_000,
    t2: float = 70_000,
    one_qubit_gate_time: float = 50,
    two_qubit_gate_time: float = 300,
):
    """
    Build a combined noise model.

    Includes:
        - depolarizing noise
        - readout error
        - optional thermal relaxation
    """
    noise_model = NoiseModel()

    one_qubit_dep = depolarizing_error(
        one_qubit_error_rate,
        1,
    )

    two_qubit_dep = depolarizing_error(
        two_qubit_error_rate,
        2,
    )

    readout_error = ReadoutError([
        [1 - readout_error_rate, readout_error_rate],
        [readout_error_rate, 1 - readout_error_rate],
    ])

    if include_thermal:
        one_qubit_thermal = thermal_relaxation_error(
            t1,
            t2,
            one_qubit_gate_time,
        )

        two_qubit_thermal = thermal_relaxation_error(
            t1,
            t2,
            two_qubit_gate_time,
        ).expand(
            thermal_relaxation_error(
                t1,
                t2,
                two_qubit_gate_time,
            )
        )

        one_qubit_error = one_qubit_dep.compose(one_qubit_thermal)
        two_qubit_error = two_qubit_dep.compose(two_qubit_thermal)
    else:
        one_qubit_error = one_qubit_dep
        two_qubit_error = two_qubit_dep

    noise_model.add_all_qubit_quantum_error(
        one_qubit_error,
        ["x", "sx", "h", "rx", "ry", "rz"],
    )

    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        ["cx"],
    )

    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model
