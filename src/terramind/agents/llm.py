"""Optional language-model backends for rewording a verified report.

The workflow never needs one. When one is supplied it only rewrites text that
the critic has already verified, and its output is discarded if it introduces
any number that was not in the verified text.
"""

from __future__ import annotations

import requests

PROMPT = """Rewrite the findings below as a short, plain answer to the question.
Rules: use only the facts given; keep every number and every [n] citation exactly
as written; add no new facts, numbers, places or dates; no preamble.

Question: {question}

Findings:
{findings}
"""


class OllamaLLM:
    """Local model served by Ollama (http://127.0.0.1:11434)."""

    def __init__(self, model: str = "llama3:8b", host: str = "http://127.0.0.1:11434"):
        self.model, self.host = model, host

    def rewrite(self, question: str, findings: str) -> str | None:
        try:
            res = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": PROMPT.format(question=question, findings=findings),
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=120,
            )
            res.raise_for_status()
            return res.json().get("response", "").strip() or None
        except requests.RequestException:
            return None
