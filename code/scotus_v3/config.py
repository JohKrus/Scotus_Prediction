"""Central configuration — loads config.yaml and resolves paths."""

from __future__ import annotations

from pathlib import Path

import re

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_cfg: dict | None = None


def load(path: Path | str | None = None) -> dict:
    """Load config.yaml (cached after first call)."""
    global _cfg
    if _cfg is not None and path is None:
        return _cfg

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    return _cfg


def project_dir() -> Path:
    """Root directory of the project (parent of scotus_v3/ package)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return project_dir() / "data"


def results_dir() -> Path:
    return project_dir() / "results"


def pdf_dir_for_term(term: int) -> Path:
    """e.g. data/term_22/"""
    return data_dir() / f"term_{term}"


def pdf_dir_for_docket(term: int, docket: str) -> Path:
    """e.g. data/term_22/22-001_pdfs/"""
    return pdf_dir_for_term(term) / f"{docket}_pdfs"


def csv_path_for_docket(term: int, docket: str) -> Path:
    return pdf_dir_for_term(term) / f"{docket}_metadata.csv"


def prediction_path(docket: str, mode: str = "agentic_v3") -> Path:
    return results_dir() / "predictions" / mode / f"{docket}_predictions.json"


def syllabus_dir() -> Path:
    return results_dir() / "syllabi"


def checkpoint_path(stage: str, term: int) -> Path:
    """Per-stage, per-term checkpoint file."""
    return project_dir() / f".checkpoint_{stage}_{term}"


def parse_term_from_docket(docket: str) -> int:
    """Extract the term number from a docket string."""
    m = re.match(r"(\d+)", docket)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse term from docket: {docket}")


# --- v3 additions: opinion and justice RAG paths ---

def opinions_dir() -> Path:
    """Root directory for scraped opinion PDFs."""
    return data_dir() / "opinions"


def opinions_dir_for_term(term: int) -> Path:
    """e.g. data/opinions/term_22/"""
    return opinions_dir() / f"term_{term}"


def justice_indices_dir() -> Path:
    """Root directory for per-justice FAISS indices."""
    return data_dir() / "justice_indices"


def opinions_metadata_path() -> Path:
    """Master catalog of all scraped opinions."""
    return opinions_dir() / "opinions_metadata.json"
