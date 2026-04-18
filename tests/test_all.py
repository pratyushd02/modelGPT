"""
test_all.py — Comprehensive test suite for ModelGPT

Covers:
  - Input validation & coercion (_validate_and_coerce)
  - Sandbox execution (_execute_code)
  - Self-correcting retry loop (fit with mock LLM)
  - Fallback on LLM failure
  - Happy-path regression & classification
  - Edge cases (empty data, mismatched lengths, bad targets, etc.)

Run with:
    pytest test_all.py -v
"""

import pickle
import sys
import textwrap
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from modelgpt.modelgpt import ModelGPT, _FALLBACK


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg_data():
    """Simple regression dataset — 100 rows, 3 numeric features."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((100, 3)), columns=["a", "b", "c"]
    )
    y = pd.Series(X["a"] * 2 + rng.standard_normal(100), name="target")
    return X, y


@pytest.fixture
def clf_data():
    """Simple binary classification dataset — 100 rows, 3 features."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((100, 3)), columns=["f1", "f2", "f3"]
    )
    y = pd.Series((X["f1"] > 0).astype(int), name="label")
    return X, y


@pytest.fixture
def mgpt():
    """ModelGPT instance with a dummy model string (LLM calls will be mocked)."""
    return ModelGPT(model="gpt-4o-mini", verbose=False)


# ===========================================================================
# 1. INPUT VALIDATION — _validate_and_coerce
# ===========================================================================

class TestValidateAndCoerce:

    # --- type coercion ---

    def test_numpy_array_coerced_to_dataframe(self, mgpt):
        X_np = np.random.standard_normal((50, 3))
        y_np = np.random.standard_normal(50)
        X_out, y_out = mgpt._validate_and_coerce(X_np, y_np, "regression")
        assert isinstance(X_out, pd.DataFrame)
        assert isinstance(y_out, pd.Series)

    def test_list_coerced(self, mgpt):
        X_list = [[1, 2], [3, 4], [5, 6]] * 10
        y_list = [0, 1, 0] * 10
        X_out, y_out = mgpt._validate_and_coerce(X_list, y_list, "classification")
        assert isinstance(X_out, pd.DataFrame)
        assert isinstance(y_out, pd.Series)

    def test_series_without_name_gets_default_name(self, mgpt):
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        y = pd.Series(range(20))          # no name
        _, y_out = mgpt._validate_and_coerce(X, y, "regression")
        assert y_out.name == "target"

    def test_invalid_X_type_raises(self, mgpt):
        with pytest.raises(TypeError, match="DataFrame"):
            mgpt._validate_and_coerce("not_a_dataframe", pd.Series(range(10)), "regression")

    def test_invalid_y_type_raises(self, mgpt):
        X = pd.DataFrame({"a": range(10)})
        with pytest.raises(TypeError, match="Series"):
            mgpt._validate_and_coerce(X, "not_a_series", "regression")

    # --- shape checks ---

    def test_empty_X_raises(self, mgpt):
        with pytest.raises(ValueError, match="empty"):
            mgpt._validate_and_coerce(pd.DataFrame(), pd.Series(dtype=float), "regression")

    def test_empty_y_raises(self, mgpt):
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="empty"):
            mgpt._validate_and_coerce(X, pd.Series(dtype=float), "regression")

    def test_length_mismatch_raises(self, mgpt):
        X = pd.DataFrame({"a": range(50)})
        y = pd.Series(range(100))
        with pytest.raises(ValueError, match="same number of rows"):
            mgpt._validate_and_coerce(X, y, "regression")

    def test_no_feature_columns_raises(self, mgpt):
        X = pd.DataFrame(index=range(20))   # zero columns
        y = pd.Series(range(20))
        with pytest.raises(ValueError, match="at least one feature"):
            mgpt._validate_and_coerce(X, y, "regression")

    def test_too_few_rows_raises(self, mgpt):
        X = pd.DataFrame({"a": range(5)})
        y = pd.Series(range(5))
        with pytest.raises(ValueError, match="too few"):
            mgpt._validate_and_coerce(X, y, "regression")

    # --- target quality ---

    def test_all_nan_target_raises(self, mgpt):
        X = pd.DataFrame({"a": range(20)})
        y = pd.Series([None] * 20)
        with pytest.raises(ValueError, match="all NaN"):
            mgpt._validate_and_coerce(X, y, "regression")

    def test_non_numeric_regression_target_raises(self, mgpt):
        X = pd.DataFrame({"a": range(20)})
        y = pd.Series(["cat", "dog"] * 10)
        with pytest.raises(TypeError, match="non-numeric"):
            mgpt._validate_and_coerce(X, y, "regression")

    def test_single_class_classification_raises(self, mgpt):
        X = pd.DataFrame({"a": range(20)})
        y = pd.Series([1] * 20)
        with pytest.raises(ValueError, match="at least 2 classes"):
            mgpt._validate_and_coerce(X, y, "classification")

    def test_many_classes_warns(self, mgpt):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))   # 100 unique "classes"
        with pytest.warns(UserWarning, match="unique classes"):
            mgpt._validate_and_coerce(X, y, "classification")

    def test_all_nan_column_warns(self, mgpt):
        X = pd.DataFrame({"a": range(20), "b": [None] * 20})
        y = pd.Series(range(20), dtype=float)
        with pytest.warns(UserWarning, match="all NaN"):
            mgpt._validate_and_coerce(X, y, "regression")

    def test_valid_regression_passes_through(self, reg_data, mgpt):
        X, y = reg_data
        X_out, y_out = mgpt._validate_and_coerce(X, y, "regression")
        assert X_out.shape == X.shape
        assert len(y_out) == len(y)

    def test_valid_classification_passes_through(self, clf_data, mgpt):
        X, y = clf_data
        X_out, y_out = mgpt._validate_and_coerce(X, y, "classification")
        assert X_out.shape == X.shape


# ===========================================================================
# 2. SANDBOX EXECUTION — _execute_code
# ===========================================================================

class TestExecuteCode:

    def _make_data(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((30, 2)), columns=["x1", "x2"])
        y = pd.Series(rng.standard_normal(30), name="y")
        return X, y

    def test_valid_code_returns_fitted_model(self):
        X, y = self._make_data()
        code = textwrap.dedent("""\
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
        """)
        result, err = ModelGPT._execute_code(code, X, y)
        assert err is None
        assert result is not None
        assert hasattr(result, "predict")

    def test_missing_model_variable_returns_error(self):
        X, y = self._make_data()
        code = textwrap.dedent("""\
            from sklearn.linear_model import LinearRegression
            m = LinearRegression()
            m.fit(X, y)
            # 'model' not assigned
        """)
        result, err = ModelGPT._execute_code(code, X, y)
        assert result is None
        assert "model" in err.lower()

    def test_syntax_error_returns_error(self):
        X, y = self._make_data()
        code = "this is not python !!!"
        result, err = ModelGPT._execute_code(code, X, y)
        assert result is None
        assert err is not None

    def test_runtime_error_returns_error(self):
        X, y = self._make_data()
        code = textwrap.dedent("""\
            raise RuntimeError("intentional failure")
            model = None
        """)
        result, err = ModelGPT._execute_code(code, X, y)
        assert result is None
        assert "intentional failure" in err

    def test_timeout_returns_error(self):
        X, y = self._make_data()
        # Patch timeout to 1s so the test doesn't take forever
        code = textwrap.dedent("""\
            import time
            time.sleep(999)
            model = None
        """)
        # Temporarily monkey-patch the timeout
        original = ModelGPT._execute_code.__func__ if hasattr(ModelGPT._execute_code, '__func__') else None
        import modelgpt as mg_module
        old_timeout = 120
        # We test by passing a known slow code and a short timeout
        # Since timeout is hardcoded we patch subprocess.run
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("python", 1)):
            result, err = ModelGPT._execute_code(code, X, y)
        assert result is None
        assert "timed out" in err.lower()

    def test_curly_braces_in_code_do_not_corrupt(self):
        """LLM-generated code often has dicts/f-strings — must not break runner."""
        X, y = self._make_data()
        code = textwrap.dedent("""\
            from sklearn.linear_model import Ridge
            params = {"alpha": 1.0}
            model = Ridge(**params)
            model.fit(X, y)
        """)
        result, err = ModelGPT._execute_code(code, X, y)
        assert err is None
        assert result is not None

    def test_triple_quoted_strings_in_code_do_not_corrupt(self):
        """Triple quotes in generated code must survive the file write."""
        X, y = self._make_data()
        code = textwrap.dedent('''\
            from sklearn.linear_model import Ridge
            description = """this is a docstring in generated code"""
            model = Ridge(alpha=0.5)
            model.fit(X, y)
        ''')
        result, err = ModelGPT._execute_code(code, X, y)
        assert err is None
        assert result is not None

    def test_subprocess_isolation_crash_does_not_kill_parent(self):
        """A hard crash in the subprocess must not propagate to the parent."""
        X, y = self._make_data()
        code = textwrap.dedent("""\
            import os, signal
            os.kill(os.getpid(), signal.SIGTERM)
            model = None
        """)
        # Should return an error string, not raise in the parent
        result, err = ModelGPT._execute_code(code, X, y)
        assert result is None
        assert err is not None


# ===========================================================================
# 3. SELF-CORRECTING RETRY LOOP
# ===========================================================================

class TestRetryLoop:

    def _good_code(self):
        return textwrap.dedent("""\
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
        """)

    def _bad_code(self):
        return "raise ValueError('oops')\nmodel = None"

    def test_succeeds_on_first_attempt(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False, max_retries=3)

        with patch.object(mgpt, "_call_llm", return_value=self._good_code()):
            result = mgpt.fit(X, y, task="regression")

        assert hasattr(result, "predict")
        preds = result.predict(X)
        assert len(preds) == len(y)

    def test_succeeds_on_second_attempt(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False, max_retries=3)

        responses = [self._bad_code(), self._good_code()]
        with patch.object(mgpt, "_call_llm", side_effect=responses):
            result = mgpt.fit(X, y, task="regression")

        assert hasattr(result, "predict")

    def test_falls_back_after_all_retries_exhausted(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False, max_retries=3)

        with patch.object(mgpt, "_call_llm", return_value=self._bad_code()):
            result = mgpt.fit(X, y, task="regression")

        # Should be the GradientBoosting fallback
        assert isinstance(result, GradientBoostingRegressor)
        preds = result.predict(X)
        assert len(preds) == len(y)

    def test_classification_fallback_is_correct_type(self, clf_data):
        X, y = clf_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False, max_retries=2)

        with patch.object(mgpt, "_call_llm", return_value=self._bad_code()):
            result = mgpt.fit(X, y, task="classification")

        assert isinstance(result, GradientBoostingClassifier)

    def test_error_is_fed_back_to_llm_in_retry(self, reg_data):
        """Verify the retry appends the error message to the conversation."""
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False, max_retries=2)

        call_args_list = []

        def mock_llm(messages):
            call_args_list.append(messages)
            if len(call_args_list) == 1:
                return self._bad_code()
            return self._good_code()

        with patch.object(mgpt, "_call_llm", side_effect=mock_llm):
            mgpt.fit(X, y)

        # Second call should include the error in the messages
        assert len(call_args_list) == 2
        second_call_messages = call_args_list[1]
        combined = " ".join(m["content"] for m in second_call_messages)
        assert "error" in combined.lower()

    def test_invalid_task_raises_before_llm(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)
        with pytest.raises(ValueError, match="task must be"):
            mgpt.fit(X, y, task="clustering")


# ===========================================================================
# 4. LLM CONNECTION FAILURES → IMMEDIATE FALLBACK
# ===========================================================================

class TestLLMConnectionFallback:

    def test_permission_error_falls_back(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)

        with patch.object(mgpt, "_call_llm", side_effect=PermissionError("bad key")):
            result = mgpt.fit(X, y, task="regression")

        assert isinstance(result, GradientBoostingRegressor)

    def test_connection_error_falls_back(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="ollama/qwen2.5-coder:7b", verbose=False)

        with patch.object(mgpt, "_call_llm", side_effect=ConnectionError("no ollama")):
            result = mgpt.fit(X, y, task="regression")

        assert isinstance(result, GradientBoostingRegressor)

    def test_classification_connection_error_falls_back(self, clf_data):
        X, y = clf_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)

        with patch.object(mgpt, "_call_llm", side_effect=ConnectionError("down")):
            result = mgpt.fit(X, y, task="classification")

        assert isinstance(result, GradientBoostingClassifier)


# ===========================================================================
# 5. FALLBACK MODELS — _FALLBACK dict
# ===========================================================================

class TestFallbackModels:

    def test_regression_fallback_fits_and_predicts(self, reg_data):
        X, y = reg_data
        fb = _FALLBACK["regression"]
        fb.fit(X, y)
        preds = fb.predict(X)
        assert len(preds) == len(y)

    def test_classification_fallback_fits_and_predicts(self, clf_data):
        X, y = clf_data
        fb = _FALLBACK["classification"]
        fb.fit(X, y)
        preds = fb.predict(X)
        assert set(preds).issubset({0, 1})


# ===========================================================================
# 6. HAPPY PATH — end-to-end with real (mocked) LLM response
# ===========================================================================

class TestHappyPath:

    REGRESSION_CODE = textwrap.dedent("""\
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
        model.fit(X, y)
    """)

    CLASSIFICATION_CODE = textwrap.dedent("""\
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ])
        model.fit(X, y)
    """)

    def test_regression_returns_sklearn_compatible_model(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)

        with patch.object(mgpt, "_call_llm", return_value=self.REGRESSION_CODE):
            model = mgpt.fit(X, y, task="regression")

        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert np.isfinite(preds).all()

    def test_classification_returns_sklearn_compatible_model(self, clf_data):
        X, y = clf_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)

        with patch.object(mgpt, "_call_llm", return_value=self.CLASSIFICATION_CODE):
            model = mgpt.fit(X, y, task="classification")

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_metric_hint_appears_in_prompt(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)

        captured = {}

        def mock_llm(messages):
            captured["prompt"] = messages[0]["content"]
            return self.REGRESSION_CODE

        with patch.object(mgpt, "_call_llm", side_effect=mock_llm):
            mgpt.fit(X, y, task="regression", metric="RMSE")

        assert "RMSE" in captured["prompt"]

    def test_markdown_fences_stripped_from_llm_response(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)

        fenced = f"```python\n{self.REGRESSION_CODE}\n```"
        with patch.object(mgpt, "_call_llm", return_value=fenced):
            model = mgpt.fit(X, y, task="regression")

        assert hasattr(model, "predict")

    def test_fit_with_numpy_inputs_succeeds(self):
        """Validates coercion + full fit path together."""
        rng = np.random.default_rng(7)
        X_np = rng.standard_normal((80, 4))
        y_np = rng.standard_normal(80)

        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)
        with patch.object(mgpt, "_call_llm", return_value=self.REGRESSION_CODE):
            model = mgpt.fit(X_np, y_np, task="regression")

        assert hasattr(model, "predict")


# ===========================================================================
# 7. DATASET SUMMARY — _summarize_dataset
# ===========================================================================

class TestSummarizeDataset:

    def test_regression_summary_contains_target_stats(self, reg_data):
        X, y = reg_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)
        summary = mgpt._summarize_dataset(X, y, "regression")
        assert "min=" in summary
        assert "max=" in summary
        assert "mean=" in summary

    def test_classification_summary_contains_class_dist(self, clf_data):
        X, y = clf_data
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)
        summary = mgpt._summarize_dataset(X, y, "classification")
        assert "Class dist" in summary

    def test_summary_reports_missing_values(self):
        X = pd.DataFrame({"a": [1.0, None, 3.0] * 20, "b": range(60)})
        y = pd.Series(range(60), dtype=float)
        mgpt = ModelGPT(model="gpt-4o-mini", verbose=False)
        summary = mgpt._summarize_dataset(X, y, "regression")
        assert "Missing vals" in summary


# ===========================================================================
# 8. API KEY INJECTION — _inject_api_key
# ===========================================================================

class TestApiKeyInjection:

    def test_openai_key_injected(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ModelGPT._inject_api_key("gpt-4o", "sk-test-openai")
        import os
        assert os.environ["OPENAI_API_KEY"] == "sk-test-openai"

    def test_anthropic_key_injected(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ModelGPT._inject_api_key("claude-haiku-4-5", "sk-ant-test")
        import os
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"

    def test_groq_key_injected(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        ModelGPT._inject_api_key("groq/llama3-70b-8192", "gsk-test")
        import os
        assert os.environ["GROQ_API_KEY"] == "gsk-test"