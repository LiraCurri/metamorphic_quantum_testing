from qiskit import QuantumCircuit
import random

GATES = ["h", "x", "z", "rx", "ry", "rz", "cx"]

def generate_random_circuit(num_qubits=3, depth=5, add_measurements=False, seed=None):
    """
    Generate a random quantum circuit using a small gate set.

    Parameters:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Number of gate applications.
        add_measurements (bool): If True, measure all qubits at the end.
        seed (int or None): Optional random seed for reproducibility.

    Returns:
        QuantumCircuit: A randomly generated quantum circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")

    if depth < 0:
        raise ValueError("depth must be non-negative")

    if seed is not None:
        random.seed(seed)

    qc = QuantumCircuit(num_qubits, num_qubits if add_measurements else 0)

    for _ in range(depth):
        gate = random.choice(GATES)

        if gate in ["h", "x", "z"]:
            q = random.randint(0, num_qubits - 1)

            if gate == "h":
                qc.h(q)
            elif gate == "x":
                qc.x(q)
            elif gate == "z":
                qc.z(q)

        elif gate == "cx" and num_qubits > 1:
            q1 = random.randint(0, num_qubits - 1)
            q2 = random.randint(0, num_qubits - 1)

            while q1 == q2:
                q2 = random.randint(0, num_qubits - 1)

            qc.cx(q1, q2)

    if add_measurements:
        qc.measure(range(num_qubits), range(num_qubits))

    return qc

def generate_bell_circuit(add_measurements=False):
    """
    Generate a simple Bell-state circuit.
    Useful for debugging and demonstration.
    """
    qc = QuantumCircuit(2, 2 if add_measurements else 0)
    qc.h(0)
    qc.cx(0, 1)

    if add_measurements:
        qc.measure([0, 1], [0, 1])

    return qc

def generate_circuit_batch(num_circuits, num_qubits=3, depth=5, seed=None):
    """
    Generate a batch of random circuits for experiments.
    """
    circuits = []

    for i in range(num_circuits):
        circuits.append(
            generate_random_circuit(
                num_qubits=num_qubits,
                depth=depth,
                add_measurements=True,
                seed=(seed + i) if seed is not None else None
            )
        )

    return circuits
