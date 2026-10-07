"""Exact-search stand-in for the `faiss` package, used only when `import faiss` fails.

On the machine that ran the revision probes, Windows Application Control blocks
faiss-cpu's compiled extension (`_swigfaiss.pyd`), which worked in September 2026.
scotus_v2.retrieval then logs "FAISS batch failed" and silently falls back to
BM25-only retrieval, which would change the pipeline's inputs.

LangChain's FAISS vector store uses `faiss.IndexFlatL2`, a brute-force exact
search over squared L2 distances. This module re-implements that index (and
read/write) in numpy, so `retrieval.hybrid_retrieve` runs unchanged and returns
the same neighbours. Rankings can differ from faiss only on exact distance ties.
"""
from __future__ import annotations

import pickle
import sys
import types

import numpy as np


class IndexFlatL2:
    metric = "l2"

    def __init__(self, d: int):
        self.d = d
        self._x = np.zeros((0, d), dtype="float32")

    @property
    def ntotal(self) -> int:
        return self._x.shape[0]

    def add(self, x) -> None:
        self._x = np.vstack([self._x, np.asarray(x, dtype="float32").reshape(-1, self.d)])

    def search(self, q, k: int):
        q = np.asarray(q, dtype="float32").reshape(-1, self.d)
        n = self.ntotal
        dist = np.empty((q.shape[0], k), dtype="float32"); idx = np.full((q.shape[0], k), -1, dtype="int64")
        dist.fill(np.inf)
        if n == 0:
            return dist, idx
        x64 = self._x.astype("float64")
        for i, v in enumerate(q.astype("float64")):
            d = ((x64 - v) ** 2).sum(1)
            kk = min(k, n)
            top = np.argsort(d, kind="stable")[:kk]
            dist[i, :kk] = d[top]; idx[i, :kk] = top
        return dist, idx

    def reconstruct(self, i: int):
        return self._x[i]


class IndexFlatIP(IndexFlatL2):
    metric = "ip"

    def search(self, q, k: int):
        q = np.asarray(q, dtype="float32").reshape(-1, self.d)
        s = q.astype("float64") @ self._x.astype("float64").T
        kk = min(k, self.ntotal)
        top = np.argsort(-s, axis=1, kind="stable")[:, :kk]
        return np.take_along_axis(s, top, 1).astype("float32"), top


def normalize_L2(x) -> None:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    x /= n


def write_index(index, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump((type(index).__name__, index.d, index._x), fh)


def read_index(path: str, io_flags: int = 0):
    with open(path, "rb") as fh:
        kind, d, x = pickle.load(fh)
    idx = {"IndexFlatL2": IndexFlatL2, "IndexFlatIP": IndexFlatIP}[kind](d)
    idx._x = x
    return idx


def install() -> bool:
    """Register the stand-in as `faiss` if the real package cannot be imported."""
    try:
        import faiss  # noqa: F401
        return False
    except Exception:
        for k in [m for m in sys.modules if m == "faiss" or m.startswith("faiss.")]:
            del sys.modules[k]
        import importlib.machinery
        mod = types.ModuleType("faiss")
        mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
        for name in ("IndexFlatL2", "IndexFlatIP", "normalize_L2", "write_index", "read_index"):
            setattr(mod, name, globals()[name])
        mod.__version__ = "numpy-shim"
        sys.modules["faiss"] = mod
        return True
