# Metamorphic Testing for Quantum Programs (CSI 5370 Project)

Individual course project – Winter 2026  
**Student:** Elira Shaska – elirashaska@oakland.edu

## Project Overview

This project investigates metamorphic testing for quantum programs using Qiskit. The goal is to explore whether metamorphic relations can be used to detect inconsistencies, anomalous behavior, or potential bugs in quantum software when a traditional test oracle is difficult to define.

The current implementation focuses on small quantum circuits executed with the Qiskit Aer simulator. Source circuits are generated, transformed according to selected metamorphic relations, executed, and then compared using measurement distributions.

## Current Project Structure

```text
metamorphic_quantum_testing/
│
├── notebooks/               # Jupyter notebooks for prototyping and testing
├── results/                 # Saved experiment outputs and CSV files
├── src/
│   ├── analyzer.py          # Records and summarizes experiment results
│   ├── checker.py           # Compares output distributions
│   ├── circuit_generator.py # Generates source circuits
│   ├── runner.py            # Executes circuits with Qiskit Aer
│   └── transformations.py   # Metamorphic transformations
│
├── main.py                  # Main experiment script
├── README.md
└── requirements.txt


## Environment Setup on Windows

From Command Prompt:

```cmd
python -m venv qiskit-env
qiskit-env\Scripts\activate
pip install -r requirements.txt"# metamorphic_quantum_testing" 
