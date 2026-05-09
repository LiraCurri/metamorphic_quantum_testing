from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import RXGate, RYGate, RZGate, XGate
from qiskit.circuit.library import HGate

import random
import numpy as np


# CIRCUIT-LEVEL RELATIONS

def identity_transformation(qc: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    if seed is not None:
        random.seed(seed)

    transformed = qc.copy()
    for q in range(transformed.num_qubits):
        transformed.x(q)
        transformed.x(q)
    return transformed


def double_x_gate(qc: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    if seed is not None:
        random.seed(seed)

    transformed = qc.copy()
    if transformed.num_qubits > 0:
        q = random.randint(0, transformed.num_qubits - 1)
        transformed.x(q)
        transformed.x(q)
    return transformed


def barrier_only(qc: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    if seed is not None:
        random.seed(seed)

    transformed = qc.copy()
    transformed.barrier()
    return transformed


def commute_independent(qc: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Safe metamorphic relation:
    Swap two adjacent single-qubit gates acting on different qubits.
    """
    if seed is not None:
        random.seed(seed)

    if len(qc.data) < 2 or qc.num_qubits < 2:
        return qc.copy()

    transformed = qc.copy()
    n = len(transformed.data)

    # Try adjacent pairs in random order
    indices = list(range(n - 1))
    random.shuffle(indices)

    for i in indices:
        g1 = transformed.data[i]
        g2 = transformed.data[i + 1]

        if g1.operation.name in ['measure', 'barrier', 'reset'] or \
           g2.operation.name in ['measure', 'barrier', 'reset']:
            continue

        # Only single-qubit gates
        if len(g1.qubits) != 1 or len(g2.qubits) != 1:
            continue

        q1 = transformed.qubits.index(g1.qubits[0])
        q2 = transformed.qubits.index(g2.qubits[0])

        if q1 != q2:
            # Safe swap
            transformed.data[i], transformed.data[i + 1] = transformed.data[i + 1], transformed.data[i]
            return transformed

    return transformed

def global_phase_shift(qc: QuantumCircuit, phase: float = np.pi / 3, seed: int | None = None) -> QuantumCircuit:
    """
    Valid metamorphic relation: add a global phase.

    A global phase does not change measurement probabilities,
    expectation values, or VQE optimization behavior.
    """
    new = qc.copy()
    new.global_phase += phase
    return new
    
# CIRCUIT FAULTS

def fault_add_x(qc: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Insert a Pauli-X gate at a random valid position (before measurements).
    """
    rng = np.random.default_rng(seed)
    new = qc.copy()

    if new.num_qubits == 0:
        return new

    # Select random qubit
    qubit_idx = rng.integers(0, new.num_qubits)
    qubit = new.qubits[qubit_idx]

    # Find valid insertion positions (exclude measurements)
    valid_positions = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name != "measure"
    ]

    if valid_positions:
        insert_pos = valid_positions[rng.integers(0, len(valid_positions))]
    else:
        insert_pos = 0

    # Create proper instruction
    x_instr = CircuitInstruction(
        operation=XGate(),
        qubits=(qubit,),
        clbits=()
    )

    # Insert safely
    new.data.insert(insert_pos, x_instr)

    return new

def fault_remove_random_gate(qc: QuantumCircuit) -> QuantumCircuit:
    new = qc.copy()
    non_meas = [i for i, instr in enumerate(new.data) 
                if instr.operation.name != 'measure']
    
    if len(non_meas) >= 2:                     
        idxs = np.random.choice(non_meas, size=min(2, len(non_meas)), replace=False)
        for idx in sorted(idxs, reverse=True):
            del new.data[idx]
    return new

    

def fault_change_target_qubit(qc: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Change the target qubit of a random single-qubit gate.
    
    This mutation simulates misaddressed gate application.
    """
    rng = np.random.default_rng(seed)
    new = qc.copy()

    if new.num_qubits < 2:
        return new

    # Select candidate instructions (single-qubit, non-measure)
    candidates = [
        i for i, instr in enumerate(new.data)
        if len(instr.qubits) == 1 and instr.operation.name != "measure"
    ]

    if not candidates:
        return new

    idx = rng.choice(candidates)
    instr = new.data[idx]

    old_qubit = instr.qubits[0]

    # Pick a different qubit
    possible_qubits = [q for q in new.qubits if q != old_qubit]
    new_qubit = rng.choice(possible_qubits)

    # Recreate instruction with same operation, new target
    new_instr = CircuitInstruction(
        operation=instr.operation,
        qubits=(new_qubit,),
        clbits=instr.clbits
    )

    # Replace instruction
    new.data[idx] = new_instr

    return new


def fault_phase_before_h(qc: QuantumCircuit, angle: float = np.pi / 2, seed: int | None = None) -> QuantumCircuit:
    """
    Fault: insert a local RZ phase before an existing H gate.

    This makes the relative phase more likely to become observable,
    because the following H gate converts phase differences into
    amplitude differences in the computational basis.
    """
    rng = np.random.default_rng(seed)
    new = qc.copy()

    candidates = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name == "h" and len(instr.qubits) == 1
    ]

    if not candidates:
        return fault_add_x(qc, seed=seed)

    idx = int(rng.choice(candidates))
    qubit = new.data[idx].qubits[0]

    rz_instr = CircuitInstruction(
        operation=RZGate(angle),
        qubits=(qubit,),
        clbits=(),
    )

    new.data.insert(idx, rz_instr)

    return new


def fault_relative_phase(
    qc: QuantumCircuit,
    angle: float = np.pi / 4,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Fault: insert a local RZ phase rotation on a random qubit.

    Unlike a global phase, a local relative phase can change interference
    patterns and expectation values.
    """
    rng = np.random.default_rng(seed)
    new = qc.copy()

    if new.num_qubits == 0:
        return new

    q_idx = int(rng.integers(0, new.num_qubits))
    qubit = new.qubits[q_idx]

    # For VQE ansatz circuits, there are usually no measurements.
    insert_pos = int(rng.integers(0, len(new.data) + 1))

    rz_instr = CircuitInstruction(
        operation=RZGate(angle),
        qubits=(qubit,),
        clbits=(),
    )

    new.data.insert(insert_pos, rz_instr)

    return new

    
# VQE VALID RELATIONS

def vqe_identity(ansatz: QuantumCircuit) -> QuantumCircuit:
    """
    Semantics-preserving relation: insert X X on every qubit.
    """
    new = ansatz.copy()
    for q in range(new.num_qubits):
        new.x(q)
        new.x(q)
    return new


def vqe_barrier(ansatz: QuantumCircuit) -> QuantumCircuit:
    """
    Semantics-preserving relation: add a barrier.
    """
    new = ansatz.copy()
    new.barrier()
    return new


def vqe_identity_xx(ansatz: QuantumCircuit, qubit: int = 0) -> QuantumCircuit:
    """
    Semantics-preserving relation: insert X X on a chosen qubit.
    """
    new = ansatz.copy()
    if new.num_qubits > 0:
        q = min(qubit, new.num_qubits - 1)
        new.x(q)
        new.x(q)
    return new

def vqe_global_phase(ansatz: QuantumCircuit, phase: float = np.pi / 3, seed: int | None = None) -> QuantumCircuit:
    """
    Valid VQE relation: add a global phase to the ansatz.
    """
    return global_phase_shift(ansatz, phase=phase, seed=seed)


# VQE FAULTS

def vqe_fault_x(ansatz: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Fault: prepend a Pauli-X on a random qubit.
    """
    rng = np.random.default_rng(seed)
    new = ansatz.copy()

    if new.num_qubits == 0:
        return new

    q = int(rng.integers(0, new.num_qubits))
    temp = QuantumCircuit(new.num_qubits)
    temp.x(q)

    return temp.compose(new)


def vqe_fault_replace_rotation(ansatz: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Fault: replace one rotation gate type with a different one,
    preserving the same parameter expression.
    """
    rng = np.random.default_rng(seed)
    new = ansatz.copy()

    candidates = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name in ["rx", "ry", "rz"] and len(instr.qubits) == 1
    ]

    if not candidates:
        return vqe_fault_x(new, seed=seed)

    idx = int(rng.choice(candidates))
    instr = new.data[idx]

    qubit = instr.qubits[0]
    param = instr.operation.params[0]

    if instr.operation.name == "rx":
        new_gate = RYGate(param)
    elif instr.operation.name == "ry":
        new_gate = RZGate(param)
    else:
        new_gate = RXGate(param)

    new.data[idx] = CircuitInstruction(
        operation=new_gate,
        qubits=(qubit,),
        clbits=instr.clbits
    )

    return new


def vqe_fault_shift_parameter(
    ansatz: QuantumCircuit,
    shift: float = np.pi / 4,
    seed: int | None = None
) -> QuantumCircuit:
    """
    Fault: shift the parameter of one rotation gate by a constant.
    This preserves the gate type but changes the implemented unitary.
    """
    rng = np.random.default_rng(seed)
    new = ansatz.copy()

    candidates = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name in ["rx", "ry", "rz"] and len(instr.qubits) == 1
    ]

    if not candidates:
        return new

    idx = int(rng.choice(candidates))
    instr = new.data[idx]

    qubit = instr.qubits[0]
    old_param = instr.operation.params[0]
    new_param = old_param + shift

    if instr.operation.name == "rx":
        new_gate = RXGate(new_param)
    elif instr.operation.name == "ry":
        new_gate = RYGate(new_param)
    else:
        new_gate = RZGate(new_param)

    new.data[idx] = CircuitInstruction(
        operation=new_gate,
        qubits=(qubit,),
        clbits=instr.clbits
    )

    return new


def vqe_fault_change_entanglement(ansatz: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Fault: remove a subset of CX gates.
    """
    rng = np.random.default_rng(seed)
    new = ansatz.copy()

    cnot_indices = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name == "cx"
    ]

    if not cnot_indices:
        return new

    num_remove = max(1, len(cnot_indices) // 3)
    to_remove = rng.choice(cnot_indices, size=num_remove, replace=False)

    for idx in sorted(to_remove, reverse=True):
        del new.data[idx]

    return new

def vqe_fault_relative_phase(ansatz: QuantumCircuit, angle: float = np.pi / 4, seed: int | None = None) -> QuantumCircuit:
    """
    VQE fault: insert a local RZ phase rotation into the ansatz.
    """
    return fault_relative_phase(ansatz, angle=angle, seed=seed)


def _middle_indices(indices, total_len, fraction=0.5):
    """
    Keep only indices that fall in the middle fraction of the circuit.
    Example: fraction=0.5 keeps the middle 50%.
    """
    if not indices:
        return []

    start = int((1 - fraction) / 2 * total_len)
    end = int((1 + fraction) / 2 * total_len)

    middle = [i for i in indices if start <= i < end]
    return middle if middle else indices


def vqe_fault_replace_rotation_strong(ansatz: QuantumCircuit, seed: int | None = None) -> QuantumCircuit:
    """
    Stronger fault: replace one middle-circuit rotation gate with H.
    This is much more disruptive than RX->RY or RY->RZ.
    """
    rng = np.random.default_rng(seed)
    new = ansatz.copy()

    candidates = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name in ["rx", "ry", "rz"] and len(instr.qubits) == 1
    ]

    if not candidates:
        return vqe_fault_x(new, seed=seed)

    candidates = _middle_indices(candidates, len(new.data), fraction=0.5)

    idx = int(rng.choice(candidates))
    instr = new.data[idx]
    qubit = instr.qubits[0]

    new.data[idx] = CircuitInstruction(
        operation=HGate(),
        qubits=(qubit,),
        clbits=instr.clbits
    )

    return new

def vqe_fault_shift_parameter_strong(
    ansatz: QuantumCircuit,
    shift: float = np.pi / 2,
    seed: int | None = None,
    num_mutations: int = 2,
) -> QuantumCircuit:
    """
    Stronger parameter-shift fault:
    - targets middle-circuit rotation gates
    - applies a larger shift
    - can mutate multiple gates
    """
    rng = np.random.default_rng(seed)
    new = ansatz.copy()

    candidates = [
        i for i, instr in enumerate(new.data)
        if instr.operation.name in ["rx", "ry", "rz"] and len(instr.qubits) == 1
    ]

    if not candidates:
        return new

    candidates = _middle_indices(candidates, len(new.data), fraction=0.5)

    chosen = rng.choice(
        candidates,
        size=min(num_mutations, len(candidates)),
        replace=False
    )

    for idx in chosen:
        idx = int(idx)
        instr = new.data[idx]

        qubit = instr.qubits[0]
        old_param = instr.operation.params[0]
        new_param = old_param + shift

        if instr.operation.name == "rx":
            new_gate = RXGate(new_param)
        elif instr.operation.name == "ry":
            new_gate = RYGate(new_param)
        else:
            new_gate = RZGate(new_param)

        new.data[idx] = CircuitInstruction(
            operation=new_gate,
            qubits=(qubit,),
            clbits=instr.clbits
        )

    return new

