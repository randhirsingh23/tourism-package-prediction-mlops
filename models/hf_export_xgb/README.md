# Tourism Package Prediction — XGBoost Pipeline

This repository contains a scikit-learn pipeline (OneHotEncoder + XGBClassifier) trained to predict `ProdTaken`.

## Data
- Train: RandhirSingh23/tourism-package-prediction-train
- Test:  RandhirSingh23/tourism-package-prediction-test

## Test Metrics
- Accuracy: 0.935
- Precision (class 1): 0.889
- Recall (class 1): 0.755
- F1 (class 1): 0.816
- ROC AUC: 0.958

## Usage
```python
import joblib, pandas as pd
pipe = joblib.load('model.joblib')
pred = pipe.predict(df)
proba = pipe.predict_proba(df)[:, 1]
