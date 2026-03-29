import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from modelgpt.modelgpt import ModelGPT
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes(as_frame=True)
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="PRICE")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize ModelGPT
mg = ModelGPT(model="ollama/qwen3-coder:480b-cloud")
model = mg.fit(X_train, y_train, task="regression")

#comparing against sklearn model
lr = LinearRegression()
lr_sklearn = lr.fit(X_train, y_train)

# Predict on same dataset
preds = model.predict(X_test)
pred_sklearn = lr_sklearn.predict(X_test)

print("ModelGPT R^2", r2_score(y_test, preds))
print("LinearRegression R^2:", r2_score(y_test, pred_sklearn))