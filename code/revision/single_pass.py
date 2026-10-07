"""Task 4: single-pass prediction (no deliberation) with the pipeline's inputs.

Ingestion, opinion filtering, chunking, embedding and retrieval are the
pipeline's own (scotus_v2.pdf.is_court_opinion, pdf.extract_chunks via the Task 1
chunk cache, retrieval.hybrid_retrieve, deliberation._research_precedent); only
the prompts after retrieval differ. Two arms:

  neutral  Step 1: the pipeline's case-analysis prompt (paper App. C.1), verbatim,
           over the same case context (hybrid_retrieve top 10, query
           "Supreme Court case legal arguments {docket}").
           Step 2: one neutral prediction call, no persona and no justices.
           Retrieval queries = the analysis's focal point, petitioner position and
           respondent position, each through the pipeline's research tool
           (_research_precedent: hybrid top 5, truncated to 3,000 characters as in
           the vote prompt).
  v1       The earlier single-pass study's prompts (v1/part2_prediction.py,
           GPT-4o / Claude 3.5 / Gemini 1.5 study), verbatim: its case-analysis
           prompt over the same top-10 context, then nine persona votes from the
           case analysis alone (no retrieval, no deliberation); majority wins.

Input parity: the March 2026 pipeline split off transcripts by filename only, so
the transcript section was empty in every saved run (has_transcripts = false);
by default this module reproduces that split. --no-transcripts additionally drops
any file the current content rule identifies as an argument transcript.

    python -m revision.single_pass run --smoke 3 [--models ...] [--arms neutral,v1]
    python -m revision.single_pass run --terms 2022,2023,2024,2025 --workers 6
    python -m revision.single_pass run --models Claude-Opus-5.5 --terms 2025
    python -m revision.single_pass score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from revision import common as C
from revision import llm

llm.load_keys()
from scotus_v2 import deliberation, retrieval, models as v2models, pdf  # noqa: E402

CHUNKS = C.CODE_DIR / "data" / "revision_cache" / "chunks"
FAISS_DIR = C.CODE_DIR / "data" / "revision_cache" / "faiss"
TERM_DIR = {2022: "term_22", 2023: "term_23", 2024: "term_24", 2025: "term_25"}
REPLICATES = 2

# Paper App. C.1 / scotus_v2.deliberation.analyze_case_node, verbatim
ANALYSIS_PROMPT = """Analyze this Supreme Court case and provide a comprehensive analysis.

Case Documents:
{case_ctx}

{transcript_section}

Provide your analysis in JSON format:
{{
  "focal_point": "The main legal question in one sentence",
  "legal_provision": "The statute or constitutional provision at issue",
  "petitioner_position": "Summary of petitioner's main argument",
  "respondent_position": "Summary of respondent's main argument",
  "key_precedents": ["List of relevant precedents mentioned"],
  "legal_complexity": "Simple/Moderate/Complex"
}}"""

NEUTRAL_PROMPT = """You are an expert on the Supreme Court of the United States. Predict how the Court will decide this case.

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}
PETITIONER'S POSITION: {petitioner_position}
RESPONDENT'S POSITION: {respondent_position}

EXCERPTS FROM THE CASE FILINGS:
{research_results}

{transcript_section}

Respond in this EXACT JSON format:
{{
  "predicted_winner": "Petitioner" or "Respondent",
  "p_petitioner_wins": 0.0 to 1.0,
  "predicted_split": "Majority-minority vote count, e.g. 6-3",
  "reasoning": "Your reasoning in at most 4 sentences"
}}
"""

# v1/part2_prediction.py, verbatim
V1_ANALYSIS_PROMPT = """
You are a Supreme Court analyst. Analyze the following case documents and provide a comprehensive analysis.

Case Documents:
{context}

Provide your analysis in the following JSON format:
{{
  "focal_point": "The main legal question in one sentence",
  "legal_provision": "The statute or constitutional provision at issue",
  "petitioner_position": "Summary of petitioner's main argument",
  "respondent_position": "Summary of respondent's main argument",
  "key_precedents": ["List of relevant precedents mentioned"],
  "legal_complexity": "Simple/Moderate/Complex"
}}
"""

V1_VOTING_PROMPT = """
You are Supreme Court Justice {judge_name}.

Your Judicial Philosophy: {philosophy}
Your Key Decision Factors: {key_factors}

Case Analysis:
{case_analysis}

THE LEGAL QUESTION: {focal_point}

PETITIONER'S POSITION: {petitioner_position}
RESPONDENT'S POSITION: {respondent_position}

Based on your judicial philosophy and the legal arguments, how would you vote?

Consider:
1. Text and original meaning of the law
2. Relevant precedents
3. Your judicial methodology
4. Practical consequences

Provide your decision in this EXACT JSON format:
{{
  "vote": "Petitioner" or "Respondent",
  "confidence": 0.0 to 1.0,
  "primary_reasoning": "Your main legal reason in one sentence",
  "methodology_applied": "Which aspect of your philosophy was most important"
}}
"""

retrieval._FAISS_CACHE_SIZE = 64   # this process only; pipeline default unchanged
_faiss_lock = threading.Lock()


def _key(docs) -> str:
    """Same key as retrieval.build_faiss_index."""
    return hashlib.sha1("\x00".join(d.page_content for d in docs).encode("utf-8")).hexdigest()


def ensure_index(docs) -> int:
    """Load a saved FAISS index into the pipeline's cache, or build (embed) and save it.

    Returns the number of chunks embedded now (0 if loaded from disk)."""
    from langchain_community.vectorstores import FAISS
    k = _key(docs)
    with _faiss_lock:
        if k in retrieval._FAISS_CACHE:
            return 0
        p = FAISS_DIR / k
        if p.exists():
            retrieval._FAISS_CACHE[k] = FAISS.load_local(str(p), v2models.get_embedding_model(),
                                                         allow_dangerous_deserialization=True)
            return 0
    store = retrieval.build_faiss_index(docs)
    if store is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(FAISS_DIR / k))
    return len(docs)


def load_docs(term: int, docket: str, no_transcripts: bool):
    with open(CHUNKS / TERM_DIR[term] / f"{docket}.pkl", "rb") as fh:
        docs = pickle.load(fh)
    tx_files = set()
    if no_transcripts:
        by = {}
        for d in docs:
            by.setdefault(d.metadata["source"], []).append(d.page_content)
        tx_files = {s for s, t in by.items()
                    if deliberation.TRANSCRIPT_RE.search(s) or pdf.is_oral_argument_document(t)}
        docs = [d for d in docs if d.metadata["source"] not in tx_files]
    return docs, tx_files


def march_split(docs):
    """_split_docs_by_type as of commit c53f23f (filename rule only)."""
    tx = [d for d in docs if deliberation.TRANSCRIPT_RE.search(d.metadata.get("source", ""))]
    other = [d for d in docs if not deliberation.TRANSCRIPT_RE.search(d.metadata.get("source", ""))]
    return tx, other


def contexts_for(term, docket, no_transcripts):
    docs, tx_dropped = load_docs(term, docket, no_transcripts)
    embedded = ensure_index(docs)
    tx, other = march_split(docs)
    if other and len(other) != len(docs):
        embedded += ensure_index(other)
    case_docs = retrieval.hybrid_retrieve(other or docs, f"Supreme Court case legal arguments {docket}")
    case_ctx = "\n\n---DOCUMENT---\n\n".join(d.page_content for d in case_docs)
    transcript_ctx = ""
    if tx:
        tdocs = retrieval.hybrid_retrieve(tx, f"Justice questions arguments skepticism agreement oral argument {docket}",
                                          top_k=min(8, len(tx)))
        transcript_ctx = "\n\n---TRANSCRIPT EXCERPT---\n\n".join(d.page_content for d in tdocs)
    # v1 retrieval: same hybrid top 10 over all documents
    v1_docs = retrieval.hybrid_retrieve(docs, f"Supreme Court case legal arguments {docket}")
    v1_ctx = "\n\n---DOCUMENT---\n\n".join(d.page_content for d in v1_docs)
    meta = dict(n_chunks=len(docs), embedded_now=embedded, transcript_files_dropped=sorted(tx_dropped),
                case_ctx_sources=[(d.metadata["source"], d.metadata["chunk_index"]) for d in case_docs],
                v1_ctx_sources=[(d.metadata["source"], d.metadata["chunk_index"]) for d in v1_docs],
                transcript_chunks_in_march_split=len(tx))
    return docs, case_ctx, transcript_ctx, v1_ctx, meta


def _cached_call(cache, key, model, prompt):
    rec = cache.get(key)
    if rec:
        return rec, False
    r = llm.call(model, prompt)
    rec = dict(key=key, model=model, prompt_sha=llm.prompt_hash(prompt), **r)
    cache.put(rec)
    return rec, True


def run_neutral(cache, model, docket, rep, docs, case_ctx, transcript_ctx, tag):
    ts = ("ORAL ARGUMENT EXCERPTS:\n" + transcript_ctx) if transcript_ctx else ""
    a_rec, _ = _cached_call(cache, f"{tag}|neutral|{docket}|{rep}|analysis", model,
                            ANALYSIS_PROMPT.format(case_ctx=case_ctx, transcript_section=ts))
    analysis = llm.parse_json(a_rec["text"]) or deliberation._fallback_analysis()
    queries = [analysis.get(k) for k in ("focal_point", "petitioner_position", "respondent_position")]
    blocks = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            blocks.append(deliberation._research_precedent(q, docs)[:3000])
    ts2 = ("ORAL ARGUMENT SIGNALS:\n" + transcript_ctx) if transcript_ctx else ""
    prompt = NEUTRAL_PROMPT.format(case_analysis=json.dumps(analysis, indent=2),
                                   focal_point=analysis.get("focal_point", "Unknown"),
                                   petitioner_position=analysis.get("petitioner_position", "Unknown"),
                                   respondent_position=analysis.get("respondent_position", "Unknown"),
                                   research_results="\n\n=====\n\n".join(blocks), transcript_section=ts2)
    p_rec, _ = _cached_call(cache, f"{tag}|neutral|{docket}|{rep}|predict", model, prompt)
    j = llm.parse_json(p_rec["text"]) or {}
    return dict(analysis_fallback=llm.parse_json(a_rec["text"]) is None,
                predicted_winner=str(j.get("predicted_winner", "")).strip().title() or None,
                p_petitioner_wins=j.get("p_petitioner_wins"), predicted_split=j.get("predicted_split"),
                reasoning=j.get("reasoning"),
                in_tok=a_rec["in_tok"] + p_rec["in_tok"], out_tok=a_rec["out_tok"] + p_rec["out_tok"])


def run_v1(cache, model, docket, rep, v1_ctx, tag, pool):
    a_rec, _ = _cached_call(cache, f"{tag}|v1|{docket}|{rep}|analysis", model,
                            V1_ANALYSIS_PROMPT.format(context=v1_ctx))
    analysis = llm.parse_json(a_rec["text"]) or {}

    def vote(item):
        name, info = item
        prompt = V1_VOTING_PROMPT.format(
            judge_name=name, philosophy=info["philosophy"], key_factors=", ".join(info["key_factors"]),
            case_analysis=json.dumps(analysis, indent=2),
            focal_point=analysis.get("focal_point", ""), petitioner_position=analysis.get("petitioner_position", ""),
            respondent_position=analysis.get("respondent_position", ""))
        rec, _ = _cached_call(cache, f"{tag}|v1|{docket}|{rep}|vote|{name}", model, prompt)
        j = llm.parse_json(rec["text"]) or {}
        return name, j, rec

    votes = list(pool.map(vote, deliberation.SUPREME_COURT_JUSTICES.items()))
    jv = {n: (j.get("vote") or ("Petitioner" if deliberation.SUPREME_COURT_JUSTICES[n]["conservative_lean"] > 0.5
                                else "Respondent")) for n, j, _ in votes}   # v1 default on parse failure
    pet = sum(v == "Petitioner" for v in jv.values())
    return dict(analysis_fallback=not analysis, predicted_winner="Petitioner" if pet > 4 else "Respondent",
                p_petitioner_wins=pet / 9, predicted_split=f"{max(pet, 9 - pet)}-{min(pet, 9 - pet)}",
                justice_votes=jv, vote_parse_failures=sum(1 for _, j, _ in votes if not j.get("vote")),
                in_tok=a_rec["in_tok"] + sum(r["in_tok"] for *_, r in votes),
                out_tok=a_rec["out_tok"] + sum(r["out_tok"] for *_, r in votes))


def case_list(terms):
    runs = C.scored_runs()
    t = C.truth_table()
    d = t.loc[sorted(runs.docket.unique())]
    d = d[d.term.isin(terms)]
    return [(int(r.term), dk) for dk, r in d.iterrows()]


def run(models, terms, arms, workers, smoke, no_transcripts):
    tag = "notx" if no_transcripts else "std"
    o = C.out("t4_single_pass")
    cases = case_list(terms)
    if smoke:
        cases = cases[:smoke]
    caches = {(a, m): llm.Cache(o / "raw" / f"{tag}_{a}_{m}.jsonl") for a in arms for m in models}
    out_path = o / f"predictions_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["arm"], r["model"], r["docket"], r["rep"]))
    ctx_path = o / f"contexts_{tag}.jsonl"
    ctx_done = set()
    if ctx_path.exists():
        ctx_done = {json.loads(l)["docket"] for l in ctx_path.open(encoding="utf-8")}
    lock = threading.Lock()
    totals = dict(in_tok={}, out_tok={}, calls=0, embedded=0)

    def do_case(tc):
        term, docket = tc
        todo = [(a, m, r) for a in arms for m in models for r in range(1, REPLICATES + 1)
                if (a, m, docket, r) not in done]
        if not todo:
            return
        docs, case_ctx, tx_ctx, v1_ctx, meta = contexts_for(term, docket, no_transcripts)
        with lock:
            totals["embedded"] += meta["embedded_now"]
            if docket not in ctx_done:
                with ctx_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict(docket=docket, term=term, **meta)) + "\n")
        with ThreadPoolExecutor(9) as inner:
            for a, m, r in todo:
                try:
                    if a == "neutral":
                        res = run_neutral(caches[(a, m)], m, docket, r, docs, case_ctx, tx_ctx, tag)
                    else:
                        res = run_v1(caches[(a, m)], m, docket, r, v1_ctx, tag, inner)
                except Exception as e:  # keep going; logged, re-run resumes
                    print(f"  {docket} {a} {m} rep{r} failed: {e}", flush=True)
                    continue
                rec = dict(arm=a, model=m, docket=docket, term=term, rep=r, inputs=tag,
                           transcript_section=bool(tx_ctx), **res)
                with lock:
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    totals["in_tok"][m] = totals["in_tok"].get(m, 0) + res["in_tok"]
                    totals["out_tok"][m] = totals["out_tok"].get(m, 0) + res["out_tok"]
                    totals["calls"] += 1
        print(f"  {term} {docket} done ({meta['n_chunks']} chunks, embedded {meta['embedded_now']})", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(do_case, cases))
    spend = sum(llm.cost(m, totals["in_tok"][m], totals["out_tok"][m]) for m in totals["in_tok"])
    emb_cost = totals["embedded"] * 400 / 1e6 * llm.EMBED_PRICE   # ~400 tokens per 1,500-char chunk
    print(f"replicate-runs: {totals['calls']}; LLM ${spend:.2f}; embeddings ~${emb_cost:.2f} "
          f"({totals['embedded']} chunks); tokens {totals}")
    if smoke and totals["calls"]:
        n_all = len(case_list(terms)) * len(models) * len(arms) * REPLICATES
        print(f"projected full run: ${spend / totals['calls'] * n_all:.2f} LLM for {n_all} replicate-runs "
              f"(+ embeddings ~${emb_cost / max(1, len(cases)) * len(case_list(terms)):.2f})")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _mcnemar(b: int, c: int) -> float:
    from scipy.stats import binomtest
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def score():
    o = C.out("t4_single_pass")
    truth = C.truth_table()
    pipe = C.scored_runs()
    pipe["p_pet"] = pipe.petitioner_votes / 9
    rows, paired = [], []
    for tag in ("std", "notx"):
        f = o / f"predictions_{tag}.jsonl"
        if not f.exists():
            continue
        sp = pd.DataFrame([json.loads(l) for l in f.open(encoding="utf-8")])
        sp = sp.drop_duplicates(["arm", "model", "docket", "rep"], keep="last")
        sp["truth"] = sp.docket.map(truth.winner)
        sp["correct"] = (sp.predicted_winner == sp.truth).astype(int)
        sp["p"] = pd.to_numeric(sp.p_petitioner_wins, errors="coerce").clip(0, 1)
        sp["brier"] = (sp.p - (sp.truth == "Petitioner")) ** 2
        sp.to_csv(o / f"predictions_{tag}_scored.csv", index=False)
        for (arm, model, term), g in sp.groupby(["arm", "model", "term"]):
            pm = "Claude-4.6" if model == "Claude-Opus-5.5" else model   # Opus compared w/ Claude pipeline
            pg = pipe[(pipe.model == pm) & pipe.docket.isin(g.docket)]
            acc = C.boot_mean(g, "correct")
            base = (truth.loc[g.docket.unique(), "winner"] == "Petitioner").mean()
            # paired by docket: single pass minus pipeline final / first-round (replicates averaged)
            m = g.groupby("docket").correct.mean().to_frame("sp").join(
                pg.groupby("docket").correct.mean().rename("final")).join(
                pg.groupby("docket").initial_correct.mean().rename("first"))
            m = m.dropna()
            d_final = C.fast_bootstrap_mean([np.array([v]) for v in (m.sp - m.final)])
            d_first = C.fast_bootstrap_mean([np.array([v]) for v in (m.sp - m.first)])
            d_base = C.fast_bootstrap_mean([np.array([v]) for v in
                                            (m.sp - (truth.loc[m.index, "winner"] == "Petitioner").astype(float))])
            # McNemar on docket consensus (ties -> Petitioner)
            cs = g.groupby("docket").predicted_winner.apply(C.modal_winner)
            cp = pg.groupby("docket").predicted_winner.apply(C.modal_winner)
            idx = cs.index.intersection(cp.index)
            tr = truth.loc[idx, "winner"]
            b = int(((cs[idx] == tr) & (cp[idx] != tr)).sum()); c = int(((cs[idx] != tr) & (cp[idx] == tr)).sum())
            rows.append(dict(inputs=tag, arm=arm, model=model, term=term, n_cases=g.docket.nunique(), n_runs=len(g),
                             acc=C.pct(acc[0]), lo=C.pct(acc[1]), hi=C.pct(acc[2]),
                             baseline=C.pct(base), pipeline_first=C.pct(pg.initial_correct.mean()),
                             pipeline_final=C.pct(pg.correct.mean()),
                             diff_vs_final=C.pct(d_final[0]), diff_vs_final_lo=C.pct(d_final[1]),
                             diff_vs_final_hi=C.pct(d_final[2]),
                             diff_vs_first=C.pct(d_first[0]), diff_vs_first_lo=C.pct(d_first[1]),
                             diff_vs_first_hi=C.pct(d_first[2]),
                             diff_vs_baseline=C.pct(d_base[0]), diff_vs_baseline_lo=C.pct(d_base[1]),
                             diff_vs_baseline_hi=C.pct(d_base[2]),
                             consensus_acc=C.pct((cs == truth.loc[cs.index, "winner"]).mean()),
                             mcnemar_sp_only=b, mcnemar_pipe_only=c, mcnemar_p=round(_mcnemar(b, c), 4),
                             brier=round(sp.loc[g.index, "brier"].mean(), 3),
                             pipeline_brier=round(((pg.p_pet - (pg.truth == "Petitioner")) ** 2).mean(), 3),
                             pred_petitioner_rate=C.pct((g.predicted_winner == "Petitioner").mean()),
                             analysis_fallbacks=int(g.analysis_fallback.sum()),
                             unparsed_winner=int((~g.predicted_winner.isin(["Petitioner", "Respondent"])).sum())))
    s = pd.DataFrame(rows)
    s.to_csv(o / "single_pass_by_term.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--models", default="GPT-5.2,Claude-4.6")
    ap.add_argument("--terms", default="2022,2023,2024,2025")
    ap.add_argument("--arms", default="neutral,v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--no-transcripts", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.models.split(","), [int(t) for t in a.terms.split(",")], a.arms.split(","),
            a.workers, a.smoke, a.no_transcripts)
    else:
        score()


if __name__ == "__main__":
    main()
