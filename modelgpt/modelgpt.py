"""
modelgpt.py — LLM-powered AutoML with multi-backend support via LiteLLM.

Supported backends (set via the `model` parameter):
    Ollama (local):  "ollama/qwen2.5-coder:7b"
    OpenAI:          "gpt-4o", "gpt-4o-mini"
    Anthropic:       "claude-opus-4-5", "claude-haiku-4-5"
    Groq:            "groq/llama3-70b-8192"
    Any LiteLLM-supported model string works.

Environment variables for API keys (set whichever you need):
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GROQ_API_KEY
    (Ollama needs no key — just `ollama serve` running locally)
"""

from __future__ import annotations

import re
import os
import textwrap
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import tempfile
import subprocess
import pickle
import json
import sys
import shutil
from pathlib import Path

try:
    import litellm
    litellm.telemetry = False          # opt out of LiteLLM telemetry
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback models used when every LLM retry is exhausted
# ---------------------------------------------------------------------------
_FALLBACK = {
    "regression":     GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "classification": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
}


class ModelGPT:
    """
    LLM-powered AutoML.  Describe your dataset → get a fitted sklearn model.

    Parameters
    ----------
    model : str
        Any LiteLLM model string.  Examples:
            "gpt-4o"                        # OpenAI
            "claude-haiku-4-5"              # Anthropic
            "groq/llama3-70b-8192"          # Groq
            "ollama/qwen2.5-coder:7b"       # local Ollama
        Default: "gpt-4o-mini"  (cheap, fast, and good enough for most tasks)
    api_key : str | None
        API key for the chosen backend.  If None, the key is read from the
        matching environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, …).
    max_retries : int
        How many times to send broken LLM-generated code back for self-correction.
    verbose : bool
        Print generated code and progress to stdout.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        if not _LITELLM_AVAILABLE:
            raise ImportError(
                "\n\n[ModelGPT] 'litellm' is not installed.\n"
                "  pip install litellm\n"
                "or install modelgpt with all extras:\n"
                "  pip install modelgpt[all]\n"
            )

        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.verbose = verbose

        # Inject the key into the environment so LiteLLM picks it up
        if api_key:
            self._inject_api_key(model, api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "regression",
        metric: str | None = None,
    ):
        """
        Select, tune, and train the best model for your dataset.

        Parameters
        ----------
        X      : pd.DataFrame  — feature matrix
        y      : pd.Series     — target vector
        task   : "regression" | "classification"
        metric : optional hint for the LLM, e.g. "RMSE", "AUC", "F1"

        Returns
        -------
        A fitted sklearn-compatible model with a .predict() method.
        """
        task = task.lower()
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got '{task}'.")

        summary = self._summarize_dataset(X, y, task)
        prompt  = self._build_prompt(summary, task, metric)

        if self.verbose:
            print(f"[ModelGPT] Backend : {self.model}")
            print(f"[ModelGPT] Task    : {task}  |  Metric: {metric or 'auto'}")
            print("[ModelGPT] Asking LLM for model code …\n")

        messages = [{"role": "user", "content": prompt}]
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            # ── call the LLM ──────────────────────────────────────────
            raw_response = self._call_llm(messages)

            if self.verbose:
                print(f"[ModelGPT] ── Attempt {attempt} ──────────────────────────")
                print(raw_response)
                print()

            code = self._extract_code(raw_response)

            # ── try to execute ─────────────────────────────────────────
            model_obj, error = self._execute_code(code, X, y)

            if model_obj is not None:
                if self.verbose:
                    print(f"[ModelGPT] ✓ Model fitted successfully on attempt {attempt}.")
                return model_obj

            # ── execution failed → self-correction loop ────────────────
            last_error = error
            if self.verbose:
                print(f"[ModelGPT] ✗ Execution error (attempt {attempt}/{self.max_retries}):\n  {error}\n")

            if attempt < self.max_retries:
                # Append the error + ask LLM to fix the code
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The code you generated raised this error when executed:\n\n"
                        f"  {error}\n\n"
                        "Please fix the code and return only the corrected raw Python — "
                        "no explanation, no markdown fences."
                    ),
                })

        # ── all retries exhausted → safe fallback ──────────────────────
        print(
            f"[ModelGPT] ⚠ All {self.max_retries} attempts failed. "
            f"Falling back to GradientBoosting{task.capitalize()}.\n"
            f"  Last error: {last_error}"
        )
        fallback = _FALLBACK[task]
        fallback.fit(X, y)
        return fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict]) -> str:
        """Send messages to the LLM and return the text response."""
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.2,       # low temp = more deterministic code
            )
            return response.choices[0].message.content
        except Exception as e:
            self._handle_llm_error(e)

    def _summarize_dataset(self, X: pd.DataFrame, y: pd.Series, task: str) -> str:
        """Build a compact dataset description to feed the LLM."""
        missing = X.isnull().sum().sum()
        lines = [
            f"Shape        : {X.shape[0]} rows × {X.shape[1]} columns",
            f"Columns      : {list(X.columns)}",
            f"Dtypes       : {X.dtypes.to_dict()}",
            f"Missing vals : {missing}",
            f"Target name  : {y.name}",
        ]
        if task == "regression":
            lines.append(f"Target stats : min={y.min():.4g}, max={y.max():.4g}, "
                         f"mean={y.mean():.4g}, std={y.std():.4g}")
        else:
            lines.append(f"Class dist   : {y.value_counts().to_dict()}")
        return "\n".join(lines)

    def _build_prompt(self, summary: str, task: str, metric: str | None) -> str:
        metric_line = f"Optimise for : {metric}" if metric else "Optimise for : the most appropriate default metric"
        return textwrap.dedent(f"""
            You are an expert ML engineer.

            Dataset summary:
            {summary}

            Task         : {task}
            {metric_line}

            Write Python code that:
            1. Imports all required libraries at the top.
            2. Uses X (a pandas DataFrame) and y (a pandas Series) — already in scope, do NOT load data.
            3. Selects the best algorithm and tunes hyperparameters with cross-validation.
            4. Fits the final model on the full (X, y).
            5. Stores the fitted model in a variable named exactly `model`.
            6. Does NOT print anything, does NOT evaluate on a test set.

            Return ONLY raw Python code — no markdown fences, no explanation.
        """).strip()

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Strip markdown code fences if the LLM ignores the instruction."""
        # Try ```python ... ``` or ``` ... ```
        match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    @staticmethod
    def _execute_code(code: str, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Sandboxed execution of LLM-generated code in an isolated subprocess.
        Returns (model, None) on success, (None, error_str) on failure.
        """
        timeout_seconds = 120  # bumped from 60 — GridSearchCV needs breathing room

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # --- Save training data ---
            X.to_pickle(tmp / "X.pkl")
            y.to_pickle(tmp / "y.pkl")

            # --- Write LLM code to its own file (NO f-string injection) ---
            # This is the core fix: never embed LLM-generated code inside an
            # f-string. Curly braces, backslashes, and triple-quotes in the
            # generated code will silently corrupt an f-string template,
            # producing a SyntaxError before the try/except even runs —
            # so neither result.pkl nor error.txt gets written.
            user_code_path = tmp / "user_code.py"
            user_code_path.write_text(code, encoding="utf-8")

            # --- Runner script (contains NO user content — safe to template) ---
            runner_path = tmp / "runner.py"
            runner_path.write_text(
                textwrap.dedent("""\
                    import pandas as pd
                    import numpy as np
                    import sklearn
                    import pickle
                    import traceback
                    import importlib.util, sys

                    X = pd.read_pickle("X.pkl")
                    y = pd.read_pickle("y.pkl")

                    try:
                        spec = importlib.util.spec_from_file_location(
                            "user_code", "user_code.py"
                        )
                        module = importlib.util.module_from_spec(spec)
                        # Expose X and y into the module's namespace before exec
                        module.X = X
                        module.y = y
                        spec.loader.exec_module(module)

                        if not hasattr(module, "model"):
                            raise ValueError(
                                "Variable 'model' was not defined in the generated code."
                            )

                        with open("result.pkl", "wb") as f:
                            pickle.dump(module.model, f)

                    except Exception:
                        with open("error.txt", "w", encoding="utf-8") as f:
                            f.write(traceback.format_exc())
                """),
                encoding="utf-8",
            )

            # --- Run the subprocess ---
            try:
                proc = subprocess.run(
                    [sys.executable, "runner.py"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return None, f"Execution timed out after {timeout_seconds}s."

            # --- Success path ---
            result_path = tmp / "result.pkl"
            if result_path.exists():
                try:
                    with open(result_path, "rb") as f:
                        model = pickle.load(f)
                    return model, None
                except Exception as e:
                    return None, f"Could not deserialize trained model: {e}"

            # --- Failure path: prefer error.txt, fall back to stderr ---
            error_path = tmp / "error.txt"
            if error_path.exists():
                return None, error_path.read_text(encoding="utf-8").strip()

            # stderr catches import errors and crashes before the try/except
            if proc.stderr.strip():
                return None, proc.stderr.strip()

            return None, "Unknown sandbox execution failure."

    # ------------------------------------------------------------------
    # API key injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_api_key(model: str, api_key: str) -> None:
        """Map the model prefix to the right environment variable."""
        model_lower = model.lower()
        if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif model_lower.startswith("claude"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif model_lower.startswith("groq/"):
            os.environ["GROQ_API_KEY"] = api_key
        elif model_lower.startswith("gemini"):
            os.environ["GEMINI_API_KEY"] = api_key
        else:
            # Generic fallback — LiteLLM will figure it out
            os.environ["OPENAI_API_KEY"] = api_key

    # ------------------------------------------------------------------
    # Friendly error messages
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_llm_error(exc: Exception) -> None:
        """Re-raise LLM errors with actionable messages."""
        msg = str(exc)
        if "ollama" in msg.lower() or "connection" in msg.lower():
            raise ConnectionError(
                "\n\n[ModelGPT] Cannot reach Ollama.\n"
                "  1. Install  : https://ollama.com/download\n"
                "  2. Pull     : ollama pull qwen2.5-coder:7b\n"
                "  3. Start    : ollama serve\n"
                "  Then retry.\n"
            ) from exc
        if "apikey" in msg.lower() or "api_key" in msg.lower() or "authentication" in msg.lower():
            raise PermissionError(
                "\n\n[ModelGPT] Missing or invalid API key.\n"
                "  Pass it directly:  ModelGPT(model='gpt-4o-mini', api_key='sk-...')\n"
                "  Or set the env var: export OPENAI_API_KEY='sk-...'\n"
            ) from exc
        raise RuntimeError(f"[ModelGPT] LLM call failed: {exc}") from exc