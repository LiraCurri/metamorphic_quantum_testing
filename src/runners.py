from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize



def run_circuit(
    qc,
    shots: int = 10000,
    optimization_level: int = 0,
    seed: int | None = None,
    noise_model=None,
):
    """
    Run a quantum circuit on the Qiskit Aer simulator with reproducibility.

    This is the exact implementation described in Phase 2 (Section 8.1).
    Seed support ensures fair metamorphic comparisons (source vs follow-up).
    """
    simulator = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()

    # Add seed for reproducibility
    if seed is not None:
        simulator.set_options(seed_simulator=seed)

    # Transpile
    qc_transpiled = transpile(
        qc,
        backend=simulator,
        optimization_level=optimization_level,
        seed_transpiler=seed,         
    )

    # Ensure measurements exist
    if qc_transpiled.num_clbits == 0:
        qc_transpiled = qc_transpiled.copy()
        qc_transpiled.measure_all()

    # Execute
    job = simulator.run(qc_transpiled, shots=shots)
    result = job.result()

    return result.get_counts()

def energy_expectation(ansatz, params, hamiltonian):
    """
    Compute <psi(theta)| H |psi(theta)> using statevector.
    Deterministic → ideal for testing.
    """
    bound = ansatz.assign_parameters(params)
    sv = Statevector.from_instruction(bound)
    return float(np.real(sv.expectation_value(hamiltonian)))


def symmetry_expectation(ansatz, params, observable):
    """
    Compute <psi(theta)| O |psi(theta)>.
    Used for additional monitoring (e.g., ZZ symmetry).
    """
    bound = ansatz.assign_parameters(params)
    sv = Statevector.from_instruction(bound)
    return float(np.real(sv.expectation_value(observable)))

def run_vqe_trace(
    ansatz,
    hamiltonian,
    symmetry_op,
    initial_point,
    maxiter: int = 40,
    method: str = "COBYLA",
):
    """
    Run a VQE-style optimization and record full trace.

    Returns:
        result: scipy optimization result
        trace_df: DataFrame with energy, parameters, symmetry
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
    maxiter: int = 40,
):
    """
    Run source and follow-up VQE and return both traces.
    """
    source_result, source_trace = run_vqe_trace(
        source_ansatz, hamiltonian, symmetry_op, initial_point, maxiter
    )

    follow_result, follow_trace = run_vqe_trace(
        follow_ansatz, hamiltonian, symmetry_op, initial_point, maxiter
    )

    return {
        "source_result": source_result,
        "follow_result": follow_result,
        "source_trace": source_trace,
        "follow_trace": follow_trace,
    }