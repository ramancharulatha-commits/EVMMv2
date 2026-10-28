# Machine Learning Assisted Predictive Elastic Memory Management for Containerized Applications in Kubernetes

## Overview
This project explores machine-learning-assisted prediction of future memory usage for applications running in Kubernetes. A Random Forest model predicts upcoming memory usage from historical measurements.

## Problem Statement
Reactive memory allocation responds to observed usage. This project investigates whether predicting future memory usage can help guide more proactive container memory allocation.

## Objectives
- Collect memory measurements from a Kubernetes workload.
- Train a Random Forest prediction model.
- Compare predictions with baseline methods.
- Evaluate prediction errors and safety-margin coverage.
- Demonstrate Kubernetes memory-limit adjustment experiments.

## Technologies
Python, pandas, scikit-learn, Kubernetes, Docker, kubectl, YAML, Matplotlib, and Ubuntu.

## Project Files
- `collect_dynamic.py` — collects dynamic workload measurements.
- `collect_memory.py` — collects memory measurements.
- `train_dynamic_model.py` — trains the prediction model.
- `compare_memory_models.py` — compares prediction methods.
- `evaluate_memory_safety.py` — evaluates safety-margin coverage.
- `recommend_memory.py` — generates memory recommendations.
- `auto_memory_controller.py` — experimental Kubernetes memory controller.
- `test_scale_up.py` — tests a memory-limit increase.
- `predictive-memory-demo.yaml` — Kubernetes demonstration deployment.
- `dynamic_memory_dataset.csv` — collected dataset.
- `dynamic_memory_prediction.png` — prediction graph.
- `evidence/` — execution evidence.

## Experimental Results

| Metric | Random Forest |
|---|---:|
| Usable samples | 110 |
| Training samples | 77 |
| Test samples | 33 |
| MAE | 3.685 MiB |
| RMSE | 5.794 MiB |
| R² | 0.788 |
| Safety-margin coverage | 81.8% |

The persistence baseline achieved an MAE of 2.727 MiB and RMSE of 5.774 MiB. It therefore outperformed Random Forest on these error metrics for this dataset.

Separate Kubernetes tests verified a memory-limit reduction from 96 MiB to 80 MiB and an increase from 80 MiB to 96 MiB. These were separate tests, not proof of uninterrupted autonomous scaling.

## How to Run

Activate the Python environment:

`source .venv/bin/activate`

Train the model:

`python train_dynamic_model.py`

Compare prediction methods:

`python compare_memory_models.py`

Check Kubernetes resources:

`kubectl get deployments`

`kubectl get pods -l app=predictive-memory-demo`

`kubectl top pods -l app=predictive-memory-demo`

## Limitations
- The dataset is small and comes from one test workload.
- Random Forest did not outperform the persistence baseline on MAE.
- The safety margin covered 81.8% of test readings.
- Further testing is needed before claiming production readiness.

## Future Work
- Collect data from more workloads.
- Improve prediction accuracy.
- Test scaling stability over longer periods.
- Measure memory efficiency and application performance.

## Research Reference
See the original `README.md` for the EVMMv2 research-paper citation and licensing information.
