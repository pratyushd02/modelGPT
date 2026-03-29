# ModelGPT

**LLM-powered AutoML.** Describe your dataset, pick a task — ModelGPT asks a local LLM to design, tune, and fit the best model for you.

```python
from modelgpt import ModelGPT

mg = ModelGPT()
model = mg.fit(X_train, y_train, task="regression")
predictions = model.predict(X_test)
```

---

## How it works

```
Your data  ──▶  Dataset summary  ──▶  LLM prompt
                                          │
                                          ▼
                                 LLM generates Python
                                 (ensemble + CV tuning)
                                          │
                                          ▼
                                   exec() in sandbox
                                          │
                             ┌────────────┴────────────┐
                             ▼                         ▼
                       Fitted model           Error? retry (→ LLM
                       returned               sees error, self-corrects)
                                                       │
                                              Max retries hit?
                                                       ▼
                                              Safe fallback model
```

ModelGPT feeds the LLM a structured dataset summary (shape, dtypes, missing values, target statistics, class distribution). The LLM returns raw Python that uses cross-validated hyperparameter search and an ensemble method. That code runs in a controlled namespace with `X` and `y` already bound. If it fails, the error is sent back to the LLM for self-correction — up to `max_retries` times.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/yourname/modelGPT.git
cd modelGPT

# 2. Install Python dependencies
pip install pandas scikit-learn xgboost lightgbm catboost ollama

# 3. Install and start Ollama  (https://ollama.com)
ollama pull qwen3-vl:235b-cloud       # or any model you prefer
ollama serve
```

---

## Project structure

```
modelGPT/
├── modelgpt/
│   └── modelgpt.py          # Core ModelGPT class
└── example/
    └── example_usage.py     # Quickstart demo (diabetes dataset)
```

---


## API reference

### `ModelGPT(llm_model, max_retries, verbose)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_model` | `str` | `qwen3-vl:235b-cloud` | Ollama model tag to use |
| `max_retries` | `int` | `3` | How many times to retry on code execution failure |
| `verbose` | `bool` | `True` | Print generated code and progress to stdout |

### `.fit(X, y, task, metric)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | — | Feature matrix |
| `y` | `pd.Series` | — | Target vector |
| `task` | `str` | `"regression"` | `"regression"` or `"classification"` |
| `metric` | `str \| None` | `None` | Metric hint passed to the LLM (e.g. `"RMSE"`, `"AUC"`) |

Returns a fitted sklearn-compatible model with a `.predict()` method.

---

## Choosing a model

ModelGPT works with any model available in Ollama. Larger models produce better code:

```python
mg = ModelGPT(llm_model="qwen2.5-coder:32b")
```

---

## Tips for better results

**Tell the LLM which metric matters.** The `metric` argument is forwarded directly into the prompt:

```python
model = mg.fit(X_train, y_train, task="regression", metric="RMSE")
model = mg.fit(X_train, y_train, task="classification", metric="AUC")
```

**Increase retries for hard tasks.** If the LLM frequently generates broken code, raise `max_retries`:

```python
mg = ModelGPT(max_retries=5)
```

**Inspect what was generated.** With `verbose=True` (the default) the full generated code is printed. You can copy it, tweak it, and re-run it manually.

**Fallback is always safe.** If every retry fails, ModelGPT automatically falls back to a well-tuned `GradientBoostingRegressor` / `GradientBoostingClassifier`, so `.fit()` never crashes your pipeline.

---

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- `pandas`, `scikit-learn`, `ollama`
- Optional but recommended: `xgboost`, `lightgbm`, `catboost`

---

## License

MIT
