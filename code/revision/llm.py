"""Minimal, cached LLM calls for the revision probes (Tasks 3, 4).

Direct provider SDK calls (no tools, no web search). Every response is appended
to a JSONL cache as it arrives, keyed by (model, prompt id), so runs resume.

Generation settings follow the pipeline (config.yaml: temperature 0.3,
max_tokens 4096) where the model accepts them:
  - GPT-5.2: gpt-5.2-2025-12-11, temperature 0.3 (same snapshot as the pipeline).
  - Claude Sonnet 4.6: claude-sonnet-4-6, temperature 0.3. The installed
    anthropic SDK (1.x) no longer exposes `temperature`; the API still accepts it
    for this model, so it is passed in the request body.
  - Claude Opus 5.5 (optional newer model, distinct label): the API rejects
    sampling parameters and thinking cannot be turned off, so it runs at the
    provider default (adaptive thinking) with effort set explicitly to "medium".
    No server-side model fallback is enabled: a refusal is recorded as such
    rather than silently answered by another model.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

MODELS = {
    "GPT-5.2": dict(provider="openai", id="gpt-5.2-2025-12-11", temperature=0.3, max_tokens=4096,
                    price_in=1.75, price_out=14.0),
    "Claude-4.6": dict(provider="anthropic", id="claude-sonnet-4-6", temperature=0.3, max_tokens=4096,
                       price_in=3.0, price_out=15.0),
    "Claude-Opus-5.5": dict(provider="anthropic", id="claude-opus-5-5", temperature=None, max_tokens=16000,
                            effort="medium", price_in=4.0, price_out=20.0),
}
EMBED_PRICE = 0.13  # text-embedding-3-large, $/M tokens

_KEY_FILES = {"OPENAI_API_KEY": "openai_api_key.txt", "ANTHROPIC_API_KEY": "anthropic_api_key.txt"}


def load_keys(key_dir: str | None = None) -> None:
    """Read keys from the environment, or from <key_dir>/<provider>_api_key.txt."""
    key_dir = key_dir or os.environ.get("SCOTUS_KEY_DIR")
    for var, fname in _KEY_FILES.items():
        if not os.environ.get(var) and key_dir and (Path(key_dir) / fname).exists():
            os.environ[var] = (Path(key_dir) / fname).read_text().strip()
    os.environ.setdefault("GOOGLE_API_KEY", "unused")  # scotus_v2.keys asks for all three


_clients: dict = {}
_lock = threading.Lock()


def _client(provider: str):
    with _lock:
        if provider not in _clients:
            if provider == "openai":
                import openai
                _clients[provider] = openai.OpenAI(max_retries=4, timeout=300)
            else:
                import anthropic
                _clients[provider] = anthropic.Anthropic(max_retries=4, timeout=600)
        return _clients[provider]


def call(model: str, prompt: str) -> dict:
    """One user-turn call. Returns dict(text, model_resolved, in_tok, out_tok, stop)."""
    m = MODELS[model]
    c = _client(m["provider"])
    for attempt in range(6):
        try:
            if m["provider"] == "openai":
                r = c.chat.completions.create(
                    model=m["id"], temperature=m["temperature"], max_completion_tokens=m["max_tokens"],
                    messages=[{"role": "user", "content": prompt}])
                return dict(text=r.choices[0].message.content or "", model_resolved=r.model,
                            in_tok=r.usage.prompt_tokens, out_tok=r.usage.completion_tokens,
                            stop=r.choices[0].finish_reason)
            kw = dict(model=m["id"], max_tokens=m["max_tokens"],
                      messages=[{"role": "user", "content": prompt}])
            if m.get("temperature") is not None:
                kw["extra_body"] = {"temperature": m["temperature"]}
            if m.get("effort"):
                kw["output_config"] = {"effort": m["effort"]}
            if m["max_tokens"] > 8000:
                with c.messages.stream(**kw) as s:
                    r = s.get_final_message()
            else:
                r = c.messages.create(**kw)
            return dict(text="".join(b.text for b in r.content if b.type == "text"),
                        model_resolved=r.model, in_tok=r.usage.input_tokens,
                        out_tok=r.usage.output_tokens, stop=r.stop_reason)
        except Exception as e:  # SDKs already retry 429/5xx; this covers the rest
            if attempt == 5:
                return dict(text="", model_resolved="", in_tok=0, out_tok=0, stop=f"error: {e}"[:300])
            time.sleep(10 * (attempt + 1))


def cost(model: str, in_tok: int, out_tok: int) -> float:
    m = MODELS[model]
    return (in_tok * m["price_in"] + out_tok * m["price_out"]) / 1e6


class Cache:
    """Append-only JSONL cache: one line per completed call."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    if not rec.get("stop", "").startswith("error"):
                        self.done[rec["key"]] = rec
                except json.JSONDecodeError:
                    pass
        self._lock = threading.Lock()

    def get(self, key):
        return self.done.get(key)

    def put(self, rec: dict):
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not rec.get("stop", "").startswith("error"):
                self.done[rec["key"]] = rec


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def parse_json(text: str) -> dict | None:
    """Full JSON object in the reply (fenced or bare); falls back to the pipeline's parser."""
    import re
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            pass
    from scotus_v2.deliberation import _parse_json
    return _parse_json(text)
