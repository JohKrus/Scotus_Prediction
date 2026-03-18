"""Shared PDF text extraction, chunking, and document classification.

Includes content-based court opinion detection to prevent data leakage,
even for PDFs with opaque filenames (e.g. 23-1137_o7jq.pdf).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scotus_v2 import config

log = logging.getLogger(__name__)

# --- Opinion detection (must be excluded from prediction inputs) ---

# Filename patterns that strongly suggest an opinion
_OPINION_FILENAME_RE = re.compile(
    r"(slip.?opinion|_opinion|_judgment|p\d+[a-z]\.pdf$)", re.I
)

# Primary marker: the "Cite as: XXX U. S. ____ (20XX)" format is unique to
# published SCOTUS opinions and orders. Briefs, petitions, and amicus filings
# never contain this citation format.
_CITE_AS_RE = re.compile(
    r"cite\s+as:\s*\d+\s*U\.?\s*S\.?\s*_{2,}", re.I
)

# Secondary markers: only used in combination with each other.
# "Slip opinion" as a header (typically "Slip Opinion" on a line by itself,
# or "(Slip Opinion)" — NOT when mentioned inside brief text like "the slip opinion held...")
_SLIP_OPINION_RE = re.compile(r"^\s*\(?slip\s*opinion\)?\s*$", re.I | re.MULTILINE)

# Syllabus header specific to published opinions (includes term reference)
_SYLLABUS_TERM_RE = re.compile(r"syllabus.*(?:october|january)\s*term", re.I)

# "Decided [Month] [Day], [Year]" in opinion header (briefs say "filed", not "decided")
_DECIDED_DATE_RE = re.compile(r"decided\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}", re.I)

# "It is so ordered" — only in actual opinions/orders
_SO_ORDERED_RE = re.compile(r"it\s+is\s+so\s+ordered", re.I)

# Justice delivering opinion — must be preceded by SCOTUS context (not lower courts).
# Requires "Cite as" or "SUPREME COURT OF THE UNITED STATES" nearby.
_DELIVERED_OPINION_RE = re.compile(
    r"justice\s+\w+\s+delivered\s+the\s+opinion\s+of\s+the\s+court", re.I
)


def is_court_opinion(file_path: Path) -> bool:
    """Detect whether a PDF is a court opinion/judgment (ground truth).

    These MUST be excluded from prediction inputs to prevent data leakage.

    Always checks content, regardless of filename. Uses high-precision
    markers that are unique to published SCOTUS opinions and never appear
    in briefs, petitions, or amicus filings:
      - "Cite as: XXX U. S. ____" (definitive)
      - "Slip opinion" header (definitive)
      - "Decided [date]" + SCOTUS context (high precision)
      - "It is so ordered" (definitive)
      - "Justice X delivered the opinion" (definitive)
      - Syllabus with term reference (definitive)
    """
    # Quick filename pre-check
    if _OPINION_FILENAME_RE.search(file_path.name):
        try:
            with fitz.open(file_path) as doc:
                if len(doc) > 0:
                    text = doc[0].get_text()[:800]
                    # Any of the old content markers suffice if filename matched
                    if re.search(r"(slip\s*opinion|opinion\s*of\s*the\s*court|it\s*is\s*so\s*ordered)", text, re.I):
                        log.warning("  Excluding court opinion (filename+content): %s", file_path.name)
                        return True
        except Exception:
            pass

    # Content-based detection for opaque filenames (e.g. 23-1137_o7jq.pdf)
    try:
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                return False

            # Read first 3 pages
            pages_to_check = min(3, len(doc))
            text = ""
            for i in range(pages_to_check):
                text += doc[i].get_text()[:1500] + "\n"

            # Definitive markers — any one of these is sufficient
            if _CITE_AS_RE.search(text):
                log.warning("  Excluding court opinion (Cite as): %s", file_path.name)
                return True

            if _SLIP_OPINION_RE.search(text):
                log.warning("  Excluding court opinion (Slip opinion): %s", file_path.name)
                return True

            if _SYLLABUS_TERM_RE.search(text):
                log.warning("  Excluding court opinion (Syllabus+Term): %s", file_path.name)
                return True

            if _SO_ORDERED_RE.search(text):
                log.warning("  Excluding court opinion (So ordered): %s", file_path.name)
                return True

            if _DELIVERED_OPINION_RE.search(text):
                # Require SCOTUS context to avoid matching lower court opinions in appendices
                if re.search(r"supreme\s+court\s+of\s+the\s+united\s+states", text, re.I):
                    log.warning("  Excluding court opinion (Delivered opinion): %s", file_path.name)
                    return True

            # Combined marker: "Decided [date]" requires BOTH "Cite as" AND
            # a justice attribution. Joint appendices and briefs can contain
            # "Decided" dates but never "Cite as: XXX U.S." format.
            if _DECIDED_DATE_RE.search(text) and _CITE_AS_RE.search(text):
                log.warning("  Excluding court opinion (Decided+Cite as): %s", file_path.name)
                return True

    except Exception:
        pass
    return False


# --- Document-type classification by filename ---
DOC_TYPE_PATTERNS = {
    "Petition for Certiorari": re.compile(r"(cert|petition|appforcert|petitionforwrit)", re.I),
    "Petitioner Brief": re.compile(r"(petitioner brief|merits brief|brief for petitioner)", re.I),
    "Respondent Brief": re.compile(r"(respondent brief|brief for respondent|unitedstates)", re.I),
    "Amicus Brief": re.compile(r"(amicus|amici)", re.I),
    "Joint Appendix": re.compile(r"(joint appendix|ja)", re.I),
    "Oral Argument Transcript": re.compile(r"(transcript|oralarg)", re.I),
    "Court Opinion": re.compile(r"(opinion|judgment)", re.I),
    "Reply Brief": re.compile(r"(reply brief|reply)", re.I),
    "Other Substantive Document": re.compile(r"(supplemental|memo|record|filing)", re.I),
}


def classify_document_type(
    filename: str,
    description: str = "",
    excerpt: str | None = None,
    llm=None,
) -> str:
    """Classify a PDF by filename pattern, falling back to LLM if needed."""
    blob = f"{filename} {description}".lower()

    for doc_type, pattern in DOC_TYPE_PATTERNS.items():
        if pattern.search(blob):
            return doc_type

    if llm and excerpt:
        try:
            prompt = (
                "Classify this document excerpt into one of these categories: "
                "Petition for Certiorari, Petitioner Brief, Respondent Brief, "
                "Amicus Brief, Joint Appendix, Oral Argument Transcript, "
                "Court Opinion, Reply Brief, Other Substantive Document. "
                "Respond with ONLY the category name.\n\n"
                f"Document excerpt: {excerpt[:1000]}"
            )
            response = llm.invoke(prompt)
            result = response.content.strip()
            if result in DOC_TYPE_PATTERNS or result == "Uncategorized Substantive Document":
                return result
        except Exception as e:
            log.warning("LLM classification failed: %s", e)

    return "Uncategorized Substantive Document"


# --- Text extraction ---

def extract_chunks(
    file_path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Extract text chunks from a PDF (simple mode for prediction)."""
    cfg = config.load()["retrieval"]
    chunk_size = chunk_size or cfg["chunk_size"]
    chunk_overlap = chunk_overlap or cfg["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    try:
        with fitz.open(file_path) as pdf:
            full_text = ""
            for page in pdf:
                page_text = page.get_text("text", sort=True)
                if page_text:
                    page_text = re.sub(r"\s+", " ", page_text)
                    page_text = re.sub(r"(\w)-\s+(\w)", r"\1\2", page_text)
                    full_text += page_text + "\n"

            for idx, chunk in enumerate(splitter.split_text(full_text)):
                if len(chunk.strip()) > 100:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": file_path.name, "chunk_index": idx},
                    ))
    except Exception as e:
        log.error("Error processing %s: %s", file_path, e)
    return docs


def extract_chunks_with_metadata(
    file_path: Path,
    doc_type: str = "Unknown",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Extract text chunks with rich metadata (for syllabus generation)."""
    cfg = config.load()["retrieval"]
    chunk_size = chunk_size or cfg["chunk_size"]
    chunk_overlap = chunk_overlap or cfg["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )

    docs: list[Document] = []
    try:
        with fitz.open(file_path) as pdf:
            first_1000 = ""
            if len(pdf) > 0:
                first_1000 = pdf[0].get_text()[:1000]

            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                for chunk in splitter.split_text(page_text):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path.name,
                            "page": page_num + 1,
                            "document_type": doc_type,
                            "first_1000_chars": first_1000,
                        },
                    ))
    except Exception as e:
        log.error("Error processing %s: %s", file_path, e)
    return docs


def read_first_page(file_path: Path, max_chars: int = 1000) -> str:
    """Read the first page of a PDF for classification."""
    try:
        with fitz.open(file_path) as pdf:
            if len(pdf) > 0:
                return pdf[0].get_text()[:max_chars]
    except Exception:
        pass
    return ""
