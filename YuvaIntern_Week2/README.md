# YuvaIntern Week 2 – Supervised Machine Learning Models

This project reuses the synthetic customer behaviour dataset prepared in Week 1.

## Tasks
- Classification of `will_purchase` using Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors.
- Regression of `average_order_value` using Linear Regression and Random Forest Regressor.
- Train/test evaluation, 5-fold stratified cross-validation for classification, confusion matrix, ROC curve, and metric comparison charts.
- Export fitted models using Joblib.

## Run in VS Code PowerShell
```powershell
python -m pip install -r requirements.txt
python src/train_models.py
```

## Outputs
Metrics CSV files, JSON summary, charts in `outputs/`, and fitted pipelines in `models/`.

## Data and interpretation
The dataset is synthetic and intended for education. Metrics should not be interpreted as evidence of real customer purchasing behaviour. Preprocessing is fitted within pipelines after splitting, reducing data leakage.
