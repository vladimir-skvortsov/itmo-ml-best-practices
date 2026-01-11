# Models API

## train_model

```bash
make train

python -m src.models.train_model \
    --model-type random_forest \
    --n-estimators 100
```

## predict_model

```bash
make predict
```

## Models

- **Logistic Regression**: `C=1.0, penalty="l2"`
- **Random Forest**: `n_estimators=100, max_depth=10`
- **Gradient Boosting**: `n_estimators=100, learning_rate=0.1`
- **SVM**: `C=1.0, kernel="rbf"`

## Metrics

Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix, Overfitting gap
