import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from modelgpt import ModelGPT

# ── Regression ─────────────────────────────────────────────────────────────
def test_regression_openai():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    mg = ModelGPT(model="ollama/qwen2.5-coder:7b")   # reads OPENAI_API_KEY from env
    model = mg.fit(X, y, task="regression", metric="RMSE")
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert not np.any(np.isnan(preds))

# ── Classification ──────────────────────────────────────────────────────────
def test_classification_openai():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    mg = ModelGPT(model="ollama/qwen2.5-coder:7b")
    model = mg.fit(X, y, task="classification", metric="AUC")
    preds = model.predict(X)
    assert len(preds) == len(y)

# ── Fallback fires when model string is garbage ─────────────────────────────
def test_fallback_on_bad_model():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    mg = ModelGPT(model="ollama/qwen2.5-coder:7b", max_retries=1)
    # Monkey-patch _call_llm to return broken code
    mg._call_llm = lambda msgs: "this is not valid python !!!"
    model = mg.fit(X, y, task="regression")
    # Should still return a working model via fallback
    preds = model.predict(X)
    assert len(preds) == len(y)

# ── Bad task string raises ValueError ───────────────────────────────────────
def test_bad_task_raises():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    mg = ModelGPT(model="ollama/qwen2.5-coder:7b")
    with pytest.raises(ValueError):
        mg.fit(X, y, task="timeseries")