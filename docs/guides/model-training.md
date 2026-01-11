# Model Training

## Commands

```bash
make train  # Default
```

```bash
python -m src.models.train_model \
    --model-type random_forest \
    --n-estimators 200
```

## Models

- `logistic_regression`
- `random_forest`
- `gradient_boosting`
- `svm`

## MLflow

```python
import mlflow.sklearn

mlflow.sklearn.autolog()
model.fit(X_train, y_train)
```

## Predictions

```bash
make predict
```
