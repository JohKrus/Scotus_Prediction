"""Task 1 (part 2): which chunks reached the pipeline's case-analysis prompt.

scotus_v2.retrieval.hybrid_retrieve() concatenates the BM25 top-k and the FAISS
top-k, de-duplicates by text, and stops at k. BM25 returns k results first, so
the FAISS results enter only when the BM25 top-k contains duplicate chunk texts.
The case context (k = 10, query "Supreme Court case legal arguments {docket}",
same for every model and replicate) is therefore reproducible offline, without
embeddings, whenever its BM25 top 10 are distinct -- checked here per docket.

For every pipeline input folder this writes the case-context chunks and flags
those from files filed after argument / decision and those carrying
outcome-language hits (Task 1 leak scan).

    python -m revision.t1b_case_context
"""
from __future__ import annotations

import pickle

import pandas as pd

from revision import common as C
from revision.t1_input_audit import CACHE, _LEAK_RE, LEAK_PATTERNS, TERM_DIRS


def bm25_topk(docs, query, k):
    from langchain_community.retrievers import BM25Retriever
    bm = BM25Retriever.from_documents(docs)
    bm.k = k
    return bm.invoke(query)


def main():
    o = C.out("t1_input_audit")
    inv = pd.read_csv(o / "input_inventory.csv", low_memory=False)
    late = inv.set_index(["term_dir", "docket", "file"])[["filed_after_argument", "filed_after_decision"]]
    rows = []
    for term_key in TERM_DIRS:
        for f in sorted((CACHE / term_key).glob("*.pkl")):
            docket = f.stem
            docs = pickle.load(open(f, "rb"))
            if not docs:
                continue
            top = bm25_topk(docs, f"Supreme Court case legal arguments {docket}", 10)
            distinct = len({d.page_content for d in top}) == len(top)
            for rank, d in enumerate(top, 1):
                src = d.metadata["source"]
                key = (term_key, docket, src)
                fa, fd = late.loc[key].tolist() if key in late.index else (None, None)
                hits = [n for n, rx in _LEAK_RE.items() if LEAK_PATTERNS[n][1] and rx.search(d.page_content)]
                rows.append(dict(term_dir=term_key, docket=docket, rank=rank, source=src,
                                 chunk_index=d.metadata["chunk_index"], bm25_top10_distinct=distinct,
                                 filed_after_argument=fa, filed_after_decision=fd,
                                 high_precision_outcome_hits=";".join(hits),
                                 text_head=d.page_content[:200].replace("\n", " ")))
    cc = pd.DataFrame(rows)
    cc.to_csv(o / "case_context_chunks.csv", index=False)
    by = cc.groupby(["term_dir", "docket"]).agg(
        distinct=("bm25_top10_distinct", "first"),
        post_decision_chunks=("filed_after_decision", lambda s: int((s == True).sum())),
        post_argument_chunks=("filed_after_argument", lambda s: int((s == True).sum())),
        outcome_hit_chunks=("high_precision_outcome_hits", lambda s: int((s != "").sum()))).reset_index()
    by.to_csv(o / "case_context_by_docket.csv", index=False)
    summ = by.groupby("term_dir").agg(dockets=("docket", "size"), bm25_top10_distinct=("distinct", "mean"),
                                      dockets_with_post_decision_chunk=("post_decision_chunks", lambda s: int((s > 0).sum())),
                                      dockets_with_post_argument_chunk=("post_argument_chunks", lambda s: int((s > 0).sum())),
                                      dockets_with_outcome_hit_chunk=("outcome_hit_chunks", lambda s: int((s > 0).sum())))
    summ.to_csv(o / "case_context_summary.csv")
    print(summ.to_string())
    print(cc[(cc.filed_after_decision == True) | (cc.high_precision_outcome_hits != "")]
          [["term_dir", "docket", "rank", "source", "high_precision_outcome_hits", "text_head"]].to_string())


if __name__ == "__main__":
    main()
