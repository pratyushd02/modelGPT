import re
import pandas as pd
import ollama


class ModelGPT:
    """
    LLM-powered AutoML for regression, classification, or time series tasks.
    """

    def __init__(self, llm_model: str = "qwen3-vl:235b-cloud"):
        self.llm_model = llm_model

    def fit(self, X: pd.DataFrame, y: pd.Series, task: str = "regression", metric: str = None):
        """
        Fit dataset using LLM-assisted AutoML and return the trained model.
        """
        dataset_summary = self._summarize_dataset(X, y)
        prompt = self._generate_prompt(dataset_summary, task, metric)

        # Ask LLM for model recommendation + hyperparameters + code
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        code_str = response["message"]["content"]

        # Execute the code and return trained model
        trained_model = self._execute_model_code(code_str, X, y)
        return trained_model

    def _summarize_dataset(self, X, y):
        return (
            f"Columns: {list(X.columns)}\n"
            f"Types: {X.dtypes.to_dict()}\n"
            f"Shape: {X.shape}\n"
            f"Target: {y.name}"
        )

    def _generate_prompt(self, summary, task, metric):
        return f"""
You are an expert ML engineer.
Dataset summary:
{summary}

Task: {task}, Metric: {metric}

Generate Python code to:
1. Fit the best model with the most optimized hyperparameters on the variables X (a pandas DataFrame) and y (a pandas Series).
2. Store the trained model in a variable named exactly 'model'.
3. Do not include any dataset loading code.
4. Do not include any markdown formatting or code fences.
5. Import any libraries you need at the top of the code.

Output only the raw Python code with no explanation.
"""

    def _extract_code(self, raw: str) -> str:
        """Strip markdown code fences if the LLM returns them."""
        match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _execute_model_code(self, code_str: str, X: pd.DataFrame, y: pd.Series):
        """
        Execute generated code and return the trained model object.
        """
        import numpy as np
        import sklearn

        code_str = self._extract_code(code_str)

        # Provide common libraries in globals so generated code can import them
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "sklearn": sklearn,
        }
        local_vars = {"X": X, "y": y}

        try:
            exec(code_str, exec_globals, local_vars)
        except Exception as e:
            raise RuntimeError(f"Failed to execute LLM-generated code:\n{code_str}\n\nError: {e}")

        model = local_vars.get("model")
        if model is None:
            raise ValueError("LLM-generated code did not produce a variable named 'model'.")

        return model