"""Syllabus generation for SCOTUS cases.

Generates comprehensive Supreme Court syllabi using multiple LLMs,
leveraging the shared RAG pipeline for document retrieval and reranking.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langchain_core.documents import Document

from scotus_v2 import config, models, retrieval, pdf

log = logging.getLogger(__name__)


# ===========================================================================
# Prompts
# ===========================================================================

SYLLABUS_PROMPT = """You are generating a Supreme Court syllabus. This MUST be extremely comprehensive and detailed.

ABSOLUTE REQUIREMENTS - NO EXCEPTIONS:
- MINIMUM 2000 words (anything shorter is completely unacceptable)
- Extremely detailed factual background (400-500 words minimum)
- Comprehensive multi-part legal analysis with detailed subsections (a), (b), (c), (d), (e), etc.
- Multiple detailed paragraphs for each legal issue
- Extensive citations to precedents, statutes, and constitutional provisions
- Detailed procedural history with all court decisions
- Thorough explanation of each aspect of the Court's reasoning

Context from case documents:
{context}

Generate your extremely comprehensive Supreme Court syllabus now (minimum 2000 words):"""

EXPANSION_PROMPT = """The syllabus you generated is too short at only {word_count} words. This is completely unacceptable.

CURRENT SYLLABUS:
{syllabus}

EXPAND THIS IMMEDIATELY to at least 2000 words by:
1. Adding much more detailed factual background (triple the length)
2. Creating multiple detailed subsections (a), (b), (c), (d), (e), (f) with 200+ words each
3. Adding extensive legal reasoning and precedent analysis
4. Including comprehensive procedural history
5. Explaining every aspect of the Court's reasoning in detail

Write the fully expanded version now (minimum 2000 words):"""


# ===========================================================================
# Retrieval
# ===========================================================================

SEARCH_QUERIES = [
    "Complete comprehensive case analysis parties legal question arguments procedural history SCOTUS Docket {docket}",
    "Supreme Court case {docket} detailed factual background procedural history legal issues constitutional questions",
    "SCOTUS {docket} petitioner respondent arguments holdings decision reasoning precedents constitutional analysis",
    "Case {docket} court opinions legal reasoning precedents cited constitutional law statutory interpretation",
    "Docket {docket} lower court decisions appeals procedural posture legal standards judicial reasoning",
]


def _build_syllabus_context(all_docs: list[Document], docket: str) -> str:
    """Retrieve and rerank documents for syllabus generation."""
    queries = [q.format(docket=docket) for q in SEARCH_QUERIES]
    rerank_query = f"Supreme Court case {docket} comprehensive detailed analysis legal reasoning"

    # Build typed retrievers
    typed_docs: dict[str, list[Document]] = {}
    for doc in all_docs:
        doc_type = doc.metadata.get("document_type", "Unknown")
        typed_docs.setdefault(doc_type, []).append(doc)

    typed_retrievers = {}
    for doc_type, docs in typed_docs.items():
        store = retrieval.build_faiss_index(docs)
        if store:
            cfg = config.load()["retrieval"]
            typed_retrievers[doc_type] = store.as_retriever(
                search_kwargs={"k": cfg["top_k_per_type"]}
            )

    # Full retrieval pipeline with reranking
    excerpts = retrieval.retrieve_and_rerank(
        typed_retrievers, queries, rerank_query=rerank_query,
    )

    context = f"SCOTUS Docket {docket} - CASE MATERIALS:\n\n"
    for i, doc in enumerate(excerpts):
        doc_type = doc.metadata.get("retrieved_doc_type", doc.metadata.get("document_type", "Unknown"))
        context += (
            f"--- EXCERPT {i+1}: {doc_type} ---\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', '?')}\n\n"
            f"{doc.page_content}\n\n"
        )
    return context


# ===========================================================================
# Generation
# ===========================================================================

def _generate_single(llm_name: str, llm, context: str, docket: str) -> str:
    """Generate a syllabus with one LLM, retrying if too short."""
    cfg = config.load()["syllabus"]
    max_attempts = cfg.get("max_attempts", 3)
    min_words = cfg.get("min_words", 2000)

    prompt = SYLLABUS_PROMPT.format(context=context)
    syllabus = ""

    for attempt in range(max_attempts):
        try:
            resp = llm.invoke(prompt)
            syllabus = resp.content
            word_count = len(syllabus.split())
            log.info("    %s attempt %d: %d words", llm_name, attempt + 1, word_count)

            if word_count >= min_words - 500:  # Accept if close enough
                return syllabus

            if attempt < max_attempts - 1:
                prompt = EXPANSION_PROMPT.format(
                    word_count=word_count, syllabus=syllabus,
                )
                time.sleep(2)

        except Exception as e:
            log.error("    %s attempt %d failed: %s", llm_name, attempt + 1, e)
            if models.is_rate_limit_error(e):
                models.mark_quota_exhausted(llm_name)
                return f"Error: quota exhausted for {llm_name}"
            if attempt == max_attempts - 1:
                return f"Error generating syllabus with {llm_name}: {e}"
            time.sleep(10)

    return syllabus if syllabus else f"Failed to generate syllabus with {llm_name}"


def generate_syllabi(docket: str, pdf_dir: Path) -> dict[str, str]:
    """Generate syllabi for a case using all configured LLMs.

    Returns dict mapping LLM name to syllabus text.
    """
    # Load and chunk PDFs (including opinions for syllabus generation)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found for %s", docket)
        return {}

    all_docs: list[Document] = []
    for f in pdf_files:
        all_docs.extend(pdf.extract_chunks(f))
    if not all_docs:
        log.warning("No text extracted for %s", docket)
        return {}

    log.info("  %s: %d PDFs, %d chunks (syllabus generation)", docket, len(pdf_files), len(all_docs))

    # Build context
    context = _build_syllabus_context(all_docs, docket)

    # Generate with each LLM
    llm_models = models.get_syllabus_models()
    results: dict[str, str] = {}

    for llm_name, llm in llm_models.items():
        log.info("  Generating syllabus with %s...", llm_name)
        results[llm_name] = _generate_single(llm_name, llm, context, docket)
        time.sleep(5)

    return results


# ===========================================================================
# Public API
# ===========================================================================

def run(
    terms: list[int] | None = None,
    single_docket: str | None = None,
) -> None:
    """Generate syllabi for all cases or a single docket."""
    cfg = config.load()
    terms = terms or cfg["terms"]
    out_dir = config.syllabus_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if single_docket:
        term = int(single_docket.split("-")[0])
        pdf_dir = config.pdf_dir_for_docket(term, single_docket)
        if not pdf_dir.exists():
            log.error("PDF directory not found: %s", pdf_dir)
            return
        results = generate_syllabi(single_docket, pdf_dir)
        _save_results(single_docket, results, out_dir)
        return

    for term in terms:
        term_dir = config.pdf_dir_for_term(term)
        if not term_dir.exists():
            log.warning("Term directory not found: %s", term_dir)
            continue

        folders = sorted(
            d for d in term_dir.iterdir() if d.is_dir() and d.name.endswith("_pdfs")
        )
        log.info("Term 20%d: %d cases (syllabus generation)", term, len(folders))

        for i, folder in enumerate(folders, 1):
            docket = folder.name.replace("_pdfs", "")
            out_file = out_dir / f"{docket}_syllabi.json"

            if out_file.exists():
                log.info("  [%d/%d] %s — already done, skipping", i, len(folders), docket)
                continue

            log.info("  [%d/%d] %s", i, len(folders), docket)
            try:
                results = generate_syllabi(docket, folder)
                if results:
                    _save_results(docket, results, out_dir)
            except Exception as e:
                log.error("  Error processing %s: %s", docket, e)

            time.sleep(1)


def _save_results(docket: str, results: dict[str, str], out_dir: Path) -> None:
    """Save syllabus results as JSON and individual text files."""
    # JSON with metadata
    json_out = out_dir / f"{docket}_syllabi.json"
    json_data = {}
    for llm_name, text in results.items():
        is_error = text.startswith("Error") or text.startswith("Failed")
        json_data[llm_name] = {
            "text": text,
            "word_count": len(text.split()) if not is_error else 0,
            "success": not is_error,
        }
    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)

    # Individual text files
    for llm_name, text in results.items():
        safe_name = llm_name.replace("-", "_").replace(".", "_").replace(" ", "_")
        txt_out = out_dir / f"{docket}_{safe_name}.txt"
        with open(txt_out, "w") as f:
            f.write(text)

    log.info("  Saved syllabi for %s", docket)
