"""Parse SCOTUS slip opinion PDFs into per-justice segments.

A single slip opinion PDF typically contains the majority opinion,
concurrences, and dissents concatenated together. This module splits
them into individual OpinionSegments attributed to specific justices.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from scotus_v3 import config, pdf

log = logging.getLogger(__name__)


@dataclass
class OpinionSegment:
    """A single justice's opinion extracted from a slip opinion PDF."""
    justice: str            # Full name (e.g. "Clarence Thomas")
    opinion_type: str       # "majority", "concurrence", "dissent", "concurrence_in_judgment"
    joining_justices: list[str] = field(default_factory=list)
    text: str = ""
    docket: str = ""
    term: int = 0
    case_name: str = ""


# --- Justice name normalization ---

# Map last names (as they appear in opinion headers) to full names
JUSTICE_LAST_NAME_MAP: dict[str, str] = {
    "ROBERTS": "John Roberts",
    "THOMAS": "Clarence Thomas",
    "ALITO": "Samuel Alito",
    "SOTOMAYOR": "Sonia Sotomayor",
    "KAGAN": "Elena Kagan",
    "GORSUCH": "Neil Gorsuch",
    "KAVANAUGH": "Brett Kavanaugh",
    "BARRETT": "Amy Coney Barrett",
    "JACKSON": "Ketanji Brown Jackson",
    # Retired justices
    "BREYER": "Stephen Breyer",
    "GINSBURG": "Ruth Bader Ginsburg",
    "KENNEDY": "Anthony Kennedy",
    "SCALIA": "Antonin Scalia",
    "STEVENS": "John Paul Stevens",
    "SOUTER": "David Souter",
    "O'CONNOR": "Sandra Day O'Connor",
}


def _resolve_justice_name(last_name: str) -> str | None:
    """Convert a last name to a full justice name. Returns None if not recognized."""
    upper = last_name.upper().strip()
    # Strip possessives (e.g. "BARRETT'S" -> "BARRETT")
    if upper.endswith("'S"):
        upper = upper[:-2]
    if upper in JUSTICE_LAST_NAME_MAP:
        return JUSTICE_LAST_NAME_MAP[upper]
    return None


def _extract_joining_justices(text: str) -> list[str]:
    """Extract joining justices from phrases like
    'with whom JUSTICE KAGAN and JUSTICE JACKSON join'.
    """
    # Pattern: "with whom JUSTICE(S) X(,) (and) Y join"
    m = re.search(
        r"with\s+whom\s+(?:CHIEF\s+)?JUSTICES?\s+(.+?)\s+joins?",
        text, re.I
    )
    if not m:
        return []

    names_text = m.group(1)
    # Split on commas and "and"
    parts = re.split(r"\s*(?:,\s*(?:and\s+)?|and\s+)", names_text)
    justices = []
    for part in parts:
        # Remove "JUSTICE" prefix if present
        cleaned = re.sub(r"(?:CHIEF\s+)?JUSTICE\s+", "", part, flags=re.I).strip()
        if cleaned:
            resolved = _resolve_justice_name(cleaned)
            if resolved:
                justices.append(resolved)
    return justices


# --- Opinion header detection ---

# Patterns that mark the start of a new opinion section
# Justice name pattern — captures the last name.
# Matches both "JUSTICE THOMAS" (slip opinion) and "Thomas, J.," (preliminary print).
_JNAME = r"(\w+)"

# Two formats to detect justice attribution:
#   1. "JUSTICE X delivered/concurring/dissenting" (slip opinion format)
#   2. "X, J., delivered/concurring/dissenting" (preliminary print format)
#   3. "Justice X delivered" (mixed case in body text)

_OPINION_HEADERS = [
    # Majority: "JUSTICE X delivered the opinion of the Court"
    # Also: "X, J., delivered the opinion of the Court"
    (
        re.compile(
            r"(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.\s*,)"
            r"\s+delivered\s+the\s+opinion\s+of\s+the\s+Court",
            re.I,
        ),
        "majority",
    ),
    # Concurrence in judgment
    (
        re.compile(
            r"(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.\s*,?)"
            r"\s*,?\s*concurring\s+in\s+the\s+judgment",
            re.I,
        ),
        "concurrence_in_judgment",
    ),
    # Concurrence in part, dissenting in part
    (
        re.compile(
            r"(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.\s*,?)"
            r"\s*,?\s*concurring\s+in\s+part\s+and\s+dissenting\s+in\s+part",
            re.I,
        ),
        "concurrence_dissent",
    ),
    # Dissent
    (
        re.compile(
            r"(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.\s*,?)"
            r"\s*,?\s*(?:with\s+whom\s+.{5,80}\s*,?\s*)?dissenting",
            re.I,
        ),
        "dissent",
    ),
    # Concurrence (must come after more specific patterns)
    (
        re.compile(
            r"(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.\s*,?)"
            r"\s*,?\s*(?:with\s+whom\s+.{5,80}\s*,?\s*)?concurring",
            re.I,
        ),
        "concurrence",
    ),
    # "Opinion of JUSTICE X" / "Opinion of X, J."
    (
        re.compile(
            r"Opinion\s+of\s+(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.)",
            re.I,
        ),
        "concurrence",
    ),
    # "Statement of JUSTICE X"
    (
        re.compile(
            r"Statement\s+of\s+(?:(?:CHIEF\s+)?JUSTICE\s+" + _JNAME + r"|" + _JNAME + r",\s*(?:C\.\s*)?J\.)",
            re.I,
        ),
        "statement",
    ),
]


def _detect_opinion_boundaries(full_text: str) -> list[tuple[int, str, str, list[str]]]:
    """Find opinion boundaries in the full text.

    Returns list of (char_offset, justice_name, opinion_type, joining_justices).
    Sorted by char_offset.

    Distinguishes real section headers from citation references by checking
    that the match appears at or near the start of a line (not embedded in
    a citation like '(Thomas, J., dissenting)').
    """
    boundaries: list[tuple[int, str, str, list[str]]] = []
    seen_offsets: set[int] = set()

    for pattern, opinion_type in _OPINION_HEADERS:
        for m in pattern.finditer(full_text):
            offset = m.start()

            # Avoid duplicate matches at the same position
            if any(abs(offset - s) < 100 for s in seen_offsets):
                continue

            # Filter out citation references: real headers start near the
            # beginning of a line (after optional whitespace), not inside
            # parenthetical citations.
            # Look at the 30 chars before the match
            prefix = full_text[max(0, offset - 30):offset]
            # If preceded by '(' it's a citation like "(Thomas, J., dissenting)"
            stripped_prefix = prefix.rstrip()
            if stripped_prefix.endswith("("):
                continue
            # If preceded by a page number + '(' pattern, skip
            if re.search(r"\d+\s*\(?\s*$", stripped_prefix):
                # Could be citation, but check if it's really a header
                # Headers typically have a newline before them
                if "\n" not in prefix[-15:]:
                    continue

            seen_offsets.add(offset)

            # With alternation, name could be in group 1 or group 2
            last_name = m.group(1) or m.group(2)
            if not last_name:
                continue
            justice_name = _resolve_justice_name(last_name)

            # Skip unrecognized names (parser artifacts)
            if justice_name is None:
                continue

            # Extract context around the match for joining justices
            context_start = max(0, m.start() - 20)
            context_end = min(len(full_text), m.end() + 200)
            context = full_text[context_start:context_end]
            joining = _extract_joining_justices(context)

            boundaries.append((offset, justice_name, opinion_type, joining))

    boundaries.sort(key=lambda x: x[0])
    return boundaries


def parse_opinion_pdf(
    pdf_path: Path,
    docket: str = "",
    term: int = 0,
    case_name: str = "",
    author_name: str = "",
) -> list[OpinionSegment]:
    """Parse a slip opinion PDF into individual opinion segments.

    Args:
        pdf_path: Path to the opinion PDF.
        docket: Docket number (e.g. "22-340").
        term: Term number (e.g. 22).
        case_name: Case name.
        author_name: Majority author from scraper metadata (used as fallback).

    Returns:
        List of OpinionSegments, one per justice opinion section.
    """
    full_text = pdf.extract_full_text(pdf_path)
    if not full_text or len(full_text) < 500:
        log.warning("Too little text in %s, skipping", pdf_path.name)
        return []

    boundaries = _detect_opinion_boundaries(full_text)

    if not boundaries:
        # Fallback: attribute entire PDF to the majority author
        if author_name and author_name != "Per Curiam":
            log.info("  No opinion boundaries found in %s, attributing to %s", pdf_path.name, author_name)
            return [OpinionSegment(
                justice=author_name,
                opinion_type="majority",
                text=full_text,
                docket=docket,
                term=term,
                case_name=case_name,
            )]
        elif author_name == "Per Curiam":
            return [OpinionSegment(
                justice="Per Curiam",
                opinion_type="per_curiam",
                text=full_text,
                docket=docket,
                term=term,
                case_name=case_name,
            )]
        else:
            log.warning("  No boundaries and no author for %s, skipping", pdf_path.name)
            return []

    segments: list[OpinionSegment] = []

    for i, (offset, justice, op_type, joining) in enumerate(boundaries):
        # Text runs from this boundary to the next one (or end of document)
        start = offset
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        segment_text = full_text[start:end].strip()

        # Skip very short segments (likely just headers without substance)
        if len(segment_text) < 200:
            continue

        segments.append(OpinionSegment(
            justice=justice,
            opinion_type=op_type,
            joining_justices=joining,
            text=segment_text,
            docket=docket,
            term=term,
            case_name=case_name,
        ))

    if segments:
        log.info("  %s: parsed %d segments (%s)",
                 pdf_path.name, len(segments),
                 ", ".join(f"{s.justice}:{s.opinion_type}" for s in segments))

    return segments


def parse_all_opinions(terms: list[int] | None = None) -> dict[str, list[OpinionSegment]]:
    """Parse all downloaded opinion PDFs and return segments grouped by justice.

    Returns:
        Dict mapping justice name -> list of OpinionSegments from all their opinions.
    """
    from scotus_v3.opinion_scraper import load_metadata

    metadata = load_metadata()
    if not metadata:
        log.error("No opinion metadata found. Run the scraper first.")
        return {}

    cfg = config.load()
    scraper_cfg = cfg.get("opinion_scraper", {})
    terms = terms or scraper_cfg.get("terms", list(metadata.keys()))

    justice_segments: dict[str, list[OpinionSegment]] = {}
    total_parsed = 0
    total_failed = 0

    for term in sorted(terms):
        if term not in metadata:
            log.warning("No metadata for term %d", term)
            continue

        records = metadata[term]
        term_dir = config.opinions_dir_for_term(term)

        for record in records:
            pdf_path = term_dir / record.pdf_filename
            if not pdf_path.exists():
                log.debug("PDF not found: %s", pdf_path)
                total_failed += 1
                continue

            segments = parse_opinion_pdf(
                pdf_path=pdf_path,
                docket=record.docket,
                term=record.term,
                case_name=record.case_name,
                author_name=record.author_name,
            )

            for seg in segments:
                if seg.justice not in justice_segments:
                    justice_segments[seg.justice] = []
                justice_segments[seg.justice].append(seg)
                total_parsed += 1

    log.info("Parsed %d opinion segments across %d justices (%d PDFs not found)",
             total_parsed, len(justice_segments), total_failed)

    # Summary per justice
    for justice, segs in sorted(justice_segments.items()):
        types = {}
        for s in segs:
            types[s.opinion_type] = types.get(s.opinion_type, 0) + 1
        type_str = ", ".join(f"{t}={c}" for t, c in sorted(types.items()))
        log.info("  %s: %d segments (%s)", justice, len(segs), type_str)

    return justice_segments
