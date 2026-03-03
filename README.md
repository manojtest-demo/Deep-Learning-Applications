# Deep Learning Applications

## Project Title

Explainable Credit Scoring using Deep Learning

## Overview

This project implements a deep learning-based credit scoring model with explainability features. The model is trained on the German Credit dataset and uses SHAP (SHapley Additive exPlanations) to interpret model predictions.

The repository includes model training, evaluation, explainability analysis, saved model weights, and output reports.

## Repository Structure

* `EXPLAINABLE_CREDIT_SCORING_Mastercode.ipynb` – Main Jupyter Notebook containing data preprocessing, model training, evaluation, and explainability analysis.
* `german_credit_data.csv` – Dataset used for training and evaluation.
* `best_credit_mlp.pt` – Saved trained model file.
* `explainability_report.csv` – Generated SHAP explainability report.
* `plots/` – Folder containing generated visualization outputs.
* `sample_data/` – Sample working directory files.

## Dataset

The project uses the German Credit dataset for binary classification (credit risk prediction).

## Model

* Model Type: Multi-Layer Perceptron (MLP)
* Framework: PyTorch
* Task: Binary Classification (Good Credit / Bad Credit)

## Explainability

Model interpretability is implemented using SHAP to:

* Understand feature importance
* Visualize contribution of features
* Generate explainability reports

## Requirements

All required dependencies are listed in the `requirements.txt` file.

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository.
2. Install required dependencies.
3. Open the Jupyter Notebook:

```bash
jupyter notebook EXPLAINABLE_CREDIT_SCORING_Mastercode.ipynb
```

4. Run all cells to reproduce results.

## Output

* Trained model file (`.pt`)
* SHAP plots
* Explainability report (`.csv`)

## License

This project is for academic and research purposes.
