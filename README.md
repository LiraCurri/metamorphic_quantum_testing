# Metamorphic Testing for Quantum Programs (CSI 5370 Project)

## Project Overview

This project investigates metamorphic testing for quantum programs using Qiskit. The goal is to explore whether metamorphic relations can be used to detect inconsistencies, anomalous behavior, or potential bugs in quantum software when a traditional test oracle is difficult to define.

The current implementation focuses on small quantum circuits executed with the Qiskit Aer simulator. Source circuits are generated, transformed according to selected metamorphic relations, executed, and then compared using measurement distributions.

## Project Structure

```text
metamorphic_quantum_testing/
│
├── notebooks/                # Jupyter notebooks for experiments
├── results/                 # Saved experiment outputs, CSV files and figures
├── src/
│   ├── analyzer.py          # Records and summarizes experiment results
│   ├── checker.py           # Compares output distributions
│   ├── circuit_generator.py # Generates source circuits
│   ├── runners.py            # Executes circuits with Qiskit Aer
│   └── transformations.py   # Metamorphic transformations
│
├── main.py                  # Minimal entry point
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
