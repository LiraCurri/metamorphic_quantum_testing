# Metamorphic Testing for Quantum Programs

## Overview

This repository contains an implementation of a metamorphic testing framework for quantum software in Qiskit. The framework is designed for oracle-poor settings, where exact expected outputs are unavailable or difficult to define.

The implementation covers two complementary validation settings:

- **Circuit-level validation**, where source and follow-up circuits are compared through statistical analysis of measurement distributions.
- **Hybrid quantum-classical validation**, with a focus on the **Variational Quantum Eigensolver (VQE)**, where metamorphic testing is extended beyond final outputs to include optimization trajectories and physics-aware constraints.

For hybrid validation, the framework incorporates gray-box execution traces and symmetry-aware comparison. The VQE evaluation is studied across three execution regimes:

- deterministic statevector simulation,
- shot-based simulation,
- noisy shot-based simulation with controlled noise models.

These experiments are used to examine how metamorphic validation changes as execution moves from idealized conditions to more realistic stochastic and hardware-inspired settings.

## Repository Structure

```text
metamorphic_quantum_testing/
│
├── notebooks/                 # Notebooks for setup checks, examples, experiments, and analysis
├── results/                   # Saved CSV outputs, plots, and generated figures
├── src/
│   ├── analyzer.py            # Aggregation and summarization of experiment results
│   ├── checker.py             # Circuit-level and hybrid metamorphic oracles
│   ├── circuit_generator.py   # Random circuit generation utilities
│   ├── runners.py             # Circuit and VQE execution with Qiskit Aer
│   └── transformations.py     # Circuit-level and VQE transformations
│
├── main.py                    # Minimal entry point
├── README.md
└── requirements.txt
```
## Setup
Create and activate a virtual environment, then install the required packages.

### Windows

```cmd
python -m venv qiskit-env
qiskit-env\Scripts\activate
pip install -r requirements.txt
```
