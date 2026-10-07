"""Task 1: input audit (no API calls).

Re-runs the pipeline's deterministic ingestion offline -- opinion filter
(pdf.is_court_opinion), text extraction and chunking (pdf.extract_chunks),
transcript detection (March-2026 filename rule and the current content rule) --
over every docket folder under code/data/term_XX/, and writes:

  input_inventory.csv   one row per docket x file
  input_by_term.csv     per-Term summary
  leak_candidates.csv   outcome-language hits in retained text, for manual review
  leak_counts_by_term.csv
  prereg_check.csv      first-add commit of every OT2025 prediction file vs. dates

Chunk lists are cached (code/data/revision_cache/chunks/) for Task 4, which feeds
the same chunks to the unchanged retrieval code.

    python -m revision.t1_input_audit [--workers 8]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

from revision import common as C

DATA = C.CODE_DIR / "data"
CACHE = DATA / "revision_cache" / "chunks"
TERM_DIRS = {"term_22": 2022, "term_23": 2023, "term_24": 2024, "term_25": 2025}
PRED_FOLDER = {"term_22": "term_2022_2023", "term_23": "term_2023_2024",
               "term_24": "term_2024_2025", "term_25": "term_2025_2026"}

LEAK_PATTERNS = {   # name -> (regex, high_precision)
    "held_colon": (r"\bHeld:", False),
    "so_ordered": (r"it\s+is\s+so\s+ordered", True),
    "judgment_is": (r"judgment\s+of\s+the\s+(?:court\s+of\s+appeals|court\s+below|[a-z ]{1,40}court)\s+is\s+"
                    r"(?:affirmed|reversed|vacated)", True),
    "delivered_opinion": (r"JUSTICE\s+[A-Z]+\s+delivered\s+the\s+opinion", True),
    "notice_formal_revision": (r"NOTICE:\s*This\s+opinion\s+is\s+subject\s+to\s+formal\s+revision", True),
    "syllabus": (r"\bSyllabus\b", False),
    "slip_op": (r"slip\s+op\.", False),
    "judgment_issued": (r"judgment\s+issued", True),
    "mandate": (r"\bmandate\b", False),
    "petition_for_rehearing": (r"petition\s+for\s+rehearing", False),
    "costs": (r"\bcosts\b", False),
    # extensions
    "cite_as_us": (r"Cite\s+as:\s*\d+\s*U\.\s*S\.", True),
    "opinion_of_the_court": (r"opinion\s+of\s+the\s+court", False),
}
_LEAK_RE = {k: re.compile(p, re.I if k not in ("delivered_opinion", "held_colon", "syllabus") else 0)
            for k, (p, _) in LEAK_PATTERNS.items()}
_DATE_RE = re.compile(r"^\s*([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})")


def _filing_dates(term_dir: Path, docket: str) -> dict[str, pd.Timestamp]:
    f = term_dir / f"{docket}_metadata.csv"
    if not f.exists():
        return {}
    m = pd.read_csv(f, dtype=str).fillna("")
    out = {}
    for _, r in m.iterrows():
        d = _DATE_RE.match(r.get("description", ""))
        if d:
            out[r["filename"]] = (pd.to_datetime(" ".join(d.groups()), format="%b %d %Y", errors="coerce"),
                                  r["description"][:160], r.get("link_text", ""))
    return out


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def audit_docket(args):
    term_key, docket, case_name = args
    from scotus_v2 import pdf, deliberation  # noqa: import inside worker
    term_dir = DATA / term_key
    folder = term_dir / f"{docket}_pdfs"
    meta = _filing_dates(term_dir, docket)
    cap = _Capture()
    logging.getLogger("scotus_v2.pdf").addHandler(cap)
    logging.getLogger("scotus_v2.pdf").setLevel(logging.WARNING)

    rows, chunks_all, leaks = [], [], []
    own_res = [re.escape(docket)]
    first_party = re.split(r"\s+v\.?\s+", case_name or "", maxsplit=1)[0].strip()
    first_party = re.sub(r"[,.].*$", "", first_party)
    if len(first_party) >= 4 and first_party.upper() not in ("UNITED STATES", "STATE", "IN RE"):
        own_res.append(re.escape(first_party.split()[-1]) if len(first_party.split()[-1]) > 3 else re.escape(first_party))
    own_re = re.compile("|".join(own_res), re.I)

    for f in sorted(folder.glob("*.pdf")):
        cap.msgs.clear()
        excluded = pdf.is_court_opinion(f)
        reason = ""
        if excluded:
            m = re.search(r"\((.*?)\)", cap.msgs[-1]) if cap.msgs else None
            reason = "court_opinion:" + (m.group(1) if m else "?")
        try:
            with fitz.open(f) as d:
                pages = len(d)
                chars = sum(len(p.get_text()) for p in d)
                first = d[0].get_text()[:1000] if pages else ""
        except Exception:
            pages, chars, first = 0, 0, ""
        chunks = pdf.extract_chunks(f) if not excluded else []
        if not excluded and not chunks:
            reason = "no_text"
        texts = [c.page_content for c in chunks]
        is_tx_old = bool(deliberation.TRANSCRIPT_RE.search(f.name))
        is_tx_new = is_tx_old or pdf.is_oral_argument_document(texts)
        fdate, desc, link = meta.get(f.name, (pd.NaT, "", ""))
        rows.append(dict(term_dir=term_key, docket=docket, file=f.name,
                         doc_type=pdf.classify_document_type(f.name, desc),
                         description=desc, link_text=link, filing_date=fdate,
                         retained=not excluded and bool(chunks), exclusion_rule=reason,
                         is_transcript_filename=is_tx_old, is_transcript_content=is_tx_new,
                         is_amicus="amicus" in desc.lower() or "amici" in desc.lower()
                         or bool(re.search(r"amicus|amici", f.name, re.I)),
                         pages=pages, chars=chars, n_chunks=len(chunks),
                         first_page_head=re.sub(r"\s+", " ", first[:150])))
        chunks_all.extend(chunks)
        # leak scan over the chunk text the pipeline indexes
        seen = set()
        for ci, t in enumerate(texts):
            for name, rx in _LEAK_RE.items():
                for m in rx.finditer(t):
                    s0, s1 = max(0, m.start() - 90), min(len(t), m.end() + 90)
                    snip = t[s0:s1][:200]
                    key = (name, snip[60:140])
                    if key in seen:      # chunk overlap repeats hits
                        continue
                    seen.add(key)
                    ctx = t[max(0, m.start() - 300): m.end() + 300]
                    leaks.append(dict(term_dir=term_key, docket=docket, file=f.name,
                                      chunk_index=ci, offset=m.start(), pattern=name,
                                      high_precision=LEAK_PATTERNS[name][1],
                                      mentions_own_case=bool(own_re.search(ctx)), snippet=snip))
    CACHE.joinpath(term_key).mkdir(parents=True, exist_ok=True)
    with open(CACHE / term_key / f"{docket}.pkl", "wb") as fh:
        pickle.dump(chunks_all, fh)
    return rows, leaks


def prereg_check(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in sorted((C.PRED_DIR / "term_2025_2026").glob("*_predictions.json")):
        rel = f.relative_to(C.REPO_DIR).as_posix()
        log = subprocess.run(["git", "log", "--diff-filter=A", "--follow", "--format=%H %aI %cI", "--", rel],
                             cwd=C.REPO_DIR, capture_output=True, text=True).stdout.strip().splitlines()
        h, a, c = (log[-1].split() if log else ("", "", ""))
        docket = f.name.split("_")[0]
        model = "GPT-5.2" if "gpt52" in f.name else "Claude-4.6"
        rec = sc.loc[docket] if docket in sc.index else None
        arg = rec.dateArgument if rec is not None else pd.NaT
        dec = rec.dateDecision if rec is not None else pd.NaT
        cdate = pd.to_datetime(c[:10]) if c else pd.NaT
        rows.append(dict(file=rel, docket=docket, model=model, commit=h, author_date=a, committer_date=c,
                         scored=docket not in C.OT2025_EXCLUDE and rec is not None and pd.notna(rec.winner),
                         argument_date=arg, decision_date=dec,
                         committed_before_argument=bool(pd.notna(arg) and cdate < arg),
                         committed_before_decision=bool(pd.notna(dec) and cdate < dec),
                         decided_before_2026_02_01=bool(pd.notna(dec) and dec < pd.Timestamp("2026-02-01")),
                         decided_before_gpt52_snapshot_2025_12_11=bool(pd.notna(dec) and dec < pd.Timestamp("2025-12-11"))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--terms", default="term_22,term_23,term_24,term_25")
    a = ap.parse_args()
    o = C.out("t1_input_audit")
    sc = C.scdb_cases()
    runs = C.load_runs()
    predicted = set(runs.docket)

    jobs = []
    for tk in a.terms.split(","):
        d = DATA / tk
        if not d.exists():
            print(f"missing {d}")
            continue
        for folder in sorted(d.glob("*_pdfs")):
            dk = folder.name[:-5]
            jobs.append((tk, dk, sc.caseName.get(dk, "")))
    print(f"{len(jobs)} docket folders")
    rows, leaks = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (r, l) in enumerate(ex.map(audit_docket, jobs, chunksize=2)):
            rows += r
            leaks += l
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    inv = pd.DataFrame(rows)
    inv["scdb_term"] = inv.docket.map(sc.term)
    inv["argument_date"] = inv.docket.map(sc.dateArgument)
    inv["decision_date"] = inv.docket.map(sc.dateDecision)
    inv["filed_after_argument"] = inv.filing_date > inv.argument_date
    inv["filed_after_decision"] = inv.filing_date > inv.decision_date
    inv["forecast_by_pipeline"] = inv.docket.isin(predicted)
    inv.to_csv(o / "input_inventory.csv", index=False)

    lk = pd.DataFrame(leaks)
    if len(lk):
        lk["scdb_term"] = lk.docket.map(sc.term)
        lk_out = lk[lk.high_precision | lk.mentions_own_case]
        lk_out.to_csv(o / "leak_candidates.csv", index=False)
        cnt = (lk.groupby(["term_dir", "pattern"])
               .agg(hits=("snippet", "size"), dockets=("docket", "nunique"),
                    own_case_hits=("mentions_own_case", "sum")).reset_index())
        cnt.to_csv(o / "leak_counts_by_term.csv", index=False)

    # per-Term summary (prediction folder = what the pipeline was run on)
    summ = []
    for tk, g in inv.groupby("term_dir"):
        gd = g.groupby("docket")
        ret = g[g.retained]
        rd = ret.groupby("docket")
        s = dict(term_dir=tk, prediction_folder=PRED_FOLDER[tk], dockets=g.docket.nunique(),
                 dockets_forecast=g[g.forecast_by_pipeline].docket.nunique(),
                 files=len(g), files_retained=len(ret), files_excluded_opinion=int(g.exclusion_rule.str.startswith("court_opinion").sum()),
                 files_no_text=int((g.exclusion_rule == "no_text").sum()),
                 mean_files_per_docket=round(len(g) / g.docket.nunique(), 1),
                 mean_retained_per_docket=round(len(ret) / g.docket.nunique(), 1),
                 share_dockets_with_metadata=round(gd.filing_date.apply(lambda s: s.notna().any()).mean(), 3),
                 share_dockets_transcript_on_disk=round(gd.is_transcript_content.any().mean(), 3),
                 share_dockets_transcript_retained=round(g.groupby("docket").apply(
                     lambda x: (x.is_transcript_content & x.retained).any()).mean(), 3),
                 share_dockets_transcript_section_populated_march_code=round(g.groupby("docket").apply(
                     lambda x: (x.is_transcript_filename & x.retained).any()).mean(), 3),
                 share_dockets_retained_filed_after_argument=round(rd.filed_after_argument.any().reindex(gd.size().index, fill_value=False).mean(), 3),
                 share_dockets_retained_filed_after_decision=round(rd.filed_after_decision.any().reindex(gd.size().index, fill_value=False).mean(), 3),
                 median_chunks_per_docket=int(ret.groupby("docket").n_chunks.sum().median()),
                 median_amicus_files=float(g[g.is_amicus].groupby("docket").size().reindex(gd.size().index, fill_value=0).median()))
        for t in sorted(set(g.doc_type)):
            s[f"mean_files_{t.replace(' ', '_')}"] = round((g.doc_type == t).sum() / g.docket.nunique(), 2)
        summ.append(s)
    pd.DataFrame(summ).to_csv(o / "input_by_term.csv", index=False)

    pc = prereg_check(sc)
    pc.to_csv(o / "prereg_check.csv", index=False)
    print(pd.DataFrame(summ).T.to_string())


if __name__ == "__main__":
    main()
