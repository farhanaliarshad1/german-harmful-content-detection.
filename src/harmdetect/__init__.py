"""German harmful content detection: fake news, hate speech and toxicity.

Single-task transformers, a unified multi-task BERT, a task-prefixed (m)T5 and
zero-/few-shot LLM baselines, with leakage-safe data handling and SHAP analysis.
"""

__version__ = "1.1.0"

TASKS = ("fake_news", "hate_speech", "toxicity")
