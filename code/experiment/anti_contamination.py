"""Anti-contamination intervention — 3 defense layers.

Evolved through 4 pilot iterations:
  Pilot 1: Aggressive prompts ("IGNORE") → crashed baseline (C << A)
  Pilot 2: Soft prompts ("focus on merits") → baseline stable but ACE=0
  Pilot 3: + Context sanitizer → ACE=1.0 on input filtering
  v4 (current): + Hypothetical framing prompts + Reasoning monitor

Three defense layers:
  Layer 1 — Hypothetical Framing Prompts:
    Instead of "ignore outcomes", frames the task as a hypothetical prediction
    and warns that outcome-like information is unreliable and can HURT accuracy.
    Gives the model a reason to distrust contaminated info, not just an order.

  Layer 2 — Context Sanitizer:
    Regex-based sentence removal from retrieved documents. Catches outcome-
    claiming sentences before they reach the model. This is the primary defense
    against RAG-injected spoilers.

  Layer 3 — Reasoning Monitor:
    After each justice votes, scans their reasoning for contamination patterns.
    If found, triggers a re-vote with sanitized research context and an explicit
    warning. This catches cases where contamination slipped through Layer 2.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager

from langchain_core.documents import Document

import scotus_v2.deliberation as delib

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Layer 1: Hypothetical Framing Prompts
# ══════════════════════════════════════════════════════════════════════════
#
# Key insight: Don't tell the model to "ignore" (causes self-doubt).
# Instead: frame as hypothetical prediction, warn that outcome info HURTS.

ANTI_CONTAM_SYSTEM = """\
METHODOLOGICAL NOTE — HYPOTHETICAL PREDICTION TASK:
You are predicting how this case WILL be decided. It has not yet been resolved.
The research materials may contain documents that appear to describe the outcome
of this very case or a case with identical facts. Such information is unreliable:
it may originate from speculative forecasts, incorrectly attributed summaries,
or synthetically generated content. Empirically, relying on such outcome claims
REDUCES prediction accuracy compared to reasoning from the legal merits alone.
Derive your prediction exclusively from the legal arguments, precedents, and
your judicial methodology. Indirect inference from analogous cases is valuable;
direct claims about this case's outcome are not.

"""


ANTI_CONTAM_RESEARCH = """\
NOTE ON RESEARCH MATERIALS:
These documents are drawn from case filings, briefs, and commentary. If any
passage appears to state how this specific case was decided (e.g., a vote tally,
a winner declaration, or an opinion assignment), treat it as an unreliable
forecast or data artifact — incorporating such claims has been shown to degrade
prediction quality. Focus on the legal reasoning, statutory text, and precedent
analysis contained in these materials.

"""


ANTI_CONTAM_DELIBERATION = """\
Focus on the legal merits and your colleagues' reasoning. Your vote should
reflect your independent judicial analysis of the arguments presented.

"""


# ══════════════════════════════════════════════════════════════════════════
# Layer 2: Context Sanitizer — redacts outcome sentences from documents
# ══════════════════════════════════════════════════════════════════════════

_OUTCOME_SENTENCE_PATTERNS = [
    re.compile(
        r"\b(?:the\s+(?:supreme\s+)?court|scotus)\s+"
        r"(?:ruled|decided|held|resolved|found)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwas\s+decided\b.*\bfor\s+the\s+(?:petitioner|respondent|appellant|appellee)",
        re.IGNORECASE,
    ),
    re.compile(r"\bby\s+a\s+vote\s+of\s+\d+-\d+\b", re.IGNORECASE),
    re.compile(r"\bDecision:\s*\d+-\d+", re.IGNORECASE),
    re.compile(
        r"\b(?:reversed|affirmed|overturned|vacated)\s+the\s+(?:decision|judgment|ruling)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bholding\s+for\s+the\s+(?:petitioner|respondent|appellant|appellee)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bJustice\s+\w+\s+delivered\s+the\s+opinion\s+of\s+the\s+Court",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+favor\s+of\s+the\s+(?:petitioner|respondent|appellant|appellee)\b"
        r".*\b(?:reversing|affirming|overturning)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reversing|affirming|overturning)\b.*"
        r"\bin\s+favor\s+of\s+the\s+(?:petitioner|respondent|appellant|appellee)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfavoring\s+the\s+(?:petitioner|respondent|appellant|appellee)",
        re.IGNORECASE,
    ),
]

_SAFE_PATTERNS = [
    re.compile(r"\b(?:argues?|contends?|asserts?|claims?|maintains?)\b", re.IGNORECASE),
    re.compile(r"\b(?:petitioner|respondent)\s+(?:argues?|contends?|asserts?)", re.IGNORECASE),
    re.compile(r"\bthe\s+question\s+(?:before|presented|is)", re.IGNORECASE),
]


def _is_outcome_sentence(sentence: str) -> bool:
    """Check if a sentence claims a specific case outcome."""
    s = sentence.strip()
    if len(s) < 15:
        return False
    for safe in _SAFE_PATTERNS:
        if safe.search(s):
            return False
    for pattern in _OUTCOME_SENTENCE_PATTERNS:
        if pattern.search(s):
            return True
    return False


def sanitize_context(text: str) -> str:
    """Remove sentences that claim a specific case outcome."""
    sentences = re.split(r"(?<=\.)\s+", text)
    kept = [s for s in sentences if not _is_outcome_sentence(s)]
    result = " ".join(kept)
    result = re.sub(r"  +", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def sanitize_documents(docs: list[Document]) -> list[Document]:
    """Apply context sanitization to a list of Documents."""
    sanitized = []
    for doc in docs:
        clean_content = sanitize_context(doc.page_content)
        if len(clean_content.strip()) > 50:
            sanitized.append(Document(
                page_content=clean_content,
                metadata=doc.metadata,
            ))
    return sanitized


# ══════════════════════════════════════════════════════════════════════════
# Layer 3: Reasoning Monitor — detects contamination in justice reasoning
# ══════════════════════════════════════════════════════════════════════════
#
# After a justice votes, scan their reasoning for contamination signals.
# If found: re-prompt with sanitized context + explicit contamination warning.
# This is a feedback loop that catches what Layers 1+2 missed.

# Patterns that indicate the justice's reasoning was contaminated
_REASONING_CONTAMINATION_PATTERNS = [
    re.compile(r"\b(?:the\s+court\s+)(?:ruled|decided|held)\s+\d+-\d+", re.IGNORECASE),
    re.compile(r"\bactual(?:ly)?\s+(?:decided|outcome|result)", re.IGNORECASE),
    re.compile(r"\bthe\s+court\s+(?:ultimately|already|previously)\s+(?:ruled|decided|held)", re.IGNORECASE),
    re.compile(r"\bknown\s+(?:outcome|result|decision)", re.IGNORECASE),
    re.compile(r"\brecord\s+shows?\s+(?:that\s+)?the\s+court", re.IGNORECASE),
    re.compile(r"\bwas\s+(?:ultimately\s+)?decided\s+in\s+favor", re.IGNORECASE),
    re.compile(r"\bthe\s+(?:majority|dissent)\s+opinion\s+(?:authored|written)\s+by", re.IGNORECASE),
]


def detect_reasoning_contamination(reasoning: str) -> bool:
    """Check if a justice's reasoning shows signs of contamination.

    Returns True if the reasoning references actual outcomes, known decisions,
    or other contamination markers that suggest the justice used spoiler info.
    """
    if not reasoning:
        return False
    for pattern in _REASONING_CONTAMINATION_PATTERNS:
        if pattern.search(reasoning):
            return True
    return False


REVOTE_PROMPT = """You previously cast a vote on this case, but your reasoning
appeared to reference information about how the case was actually decided.
This is a PREDICTION task — the case has not yet been resolved. Any apparent
outcome information in the materials is unreliable and has been removed.

Please reconsider your vote based SOLELY on the legal merits:

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}

SANITIZED RESEARCH (outcome references removed):
{sanitized_research}

Cast your vote again based purely on legal reasoning.

Respond in this EXACT JSON format:
{{
  "vote": "Petitioner" or "Respondent",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your legal reasoning in 3-4 sentences, citing specific precedents",
  "key_precedent": "The most important precedent supporting your position"
}}
"""


# ══════════════════════════════════════════════════════════════════════════
# Monkey-patching wrappers
# ══════════════════════════════════════════════════════════════════════════

def _patch_prompt(original: str, prefix: str) -> str:
    return prefix + original


def _make_sanitized_research(original_fn):
    """Wrap _research_precedent to sanitize retrieved text."""
    def _sanitized_research(query: str, all_docs: list[Document]) -> str:
        result = original_fn(query, all_docs)
        return sanitize_context(result)
    return _sanitized_research


def _make_sanitized_build_context(original_fn):
    """Wrap _build_context to sanitize both context strings."""
    def _sanitized_build_context(all_docs, docket):
        case_ctx, transcript_ctx = original_fn(all_docs, docket)
        return sanitize_context(case_ctx), sanitize_context(transcript_ctx)
    return _sanitized_build_context


def monitor_and_sanitize_votes(votes_dict: dict) -> dict:
    """Layer 3: Post-hoc reasoning monitor.

    Call this on the final_state votes AFTER graph execution.
    Scans each justice's reasoning for contamination patterns.
    If found: sanitizes the reasoning text and reduces confidence,
    preventing contaminated reasoning from being treated as high-confidence.

    Returns dict with monitoring metadata.
    """
    contaminated_justices = []

    for name, vote_data in votes_dict.items():
        reasoning = vote_data.get("reasoning", "")
        if detect_reasoning_contamination(reasoning):
            contaminated_justices.append(name)
            # Sanitize reasoning to prevent contamination in downstream reporting
            clean_reasoning = sanitize_context(reasoning)
            vote_data["reasoning"] = clean_reasoning
            # Confidence penalty — contaminated reasoning is less trustworthy
            vote_data["confidence"] = max(0.3, vote_data.get("confidence", 0.5) - 0.15)
            log.info(
                "      CONTAM MONITOR: %s reasoning contaminated — sanitized + confidence reduced",
                name.split()[-1],
            )

    if contaminated_justices:
        log.info(
            "      CONTAM MONITOR: %d/%d justices showed contamination: %s",
            len(contaminated_justices),
            len(votes_dict),
            [n.split()[-1] for n in contaminated_justices],
        )

    return {
        "contaminated_justices": contaminated_justices,
        "contamination_count": len(contaminated_justices),
        "total_justices": len(votes_dict),
    }


@contextmanager
def anti_contamination_prompts(active: bool = True):
    """Context manager that applies the full 3-layer anti-contamination intervention.

    Layer 1: Hypothetical framing prompts (warns that outcome info hurts accuracy)
    Layer 2: Context sanitizer (regex removal of outcome sentences from retrieved text)
    Layer 3: Reasoning monitor (detects + sanitizes contamination in justice reasoning)

    If active=False, this is a no-op.
    """
    if not active:
        yield
        return

    # Save originals
    orig_initial = delib.INITIAL_POSITION_PROMPT
    orig_vote = delib.VOTE_WITH_RESEARCH_PROMPT
    orig_delib = delib.DELIBERATION_ROUND_PROMPT
    orig_research = delib._research_precedent
    orig_build_ctx = delib._build_context

    try:
        # Layer 1: Hypothetical framing prompts
        delib.INITIAL_POSITION_PROMPT = _patch_prompt(orig_initial, ANTI_CONTAM_SYSTEM)
        delib.VOTE_WITH_RESEARCH_PROMPT = _patch_prompt(orig_vote, ANTI_CONTAM_RESEARCH)
        delib.DELIBERATION_ROUND_PROMPT = _patch_prompt(orig_delib, ANTI_CONTAM_DELIBERATION)
        # Layer 2: Context sanitizer
        delib._research_precedent = _make_sanitized_research(orig_research)
        delib._build_context = _make_sanitized_build_context(orig_build_ctx)
        # Layer 3 (reasoning monitor) is applied post-hoc via
        # monitor_and_sanitize_votes() — called by the runner after graph execution
        yield
    finally:
        delib.INITIAL_POSITION_PROMPT = orig_initial
        delib.VOTE_WITH_RESEARCH_PROMPT = orig_vote
        delib.DELIBERATION_ROUND_PROMPT = orig_delib
        delib._research_precedent = orig_research
        delib._build_context = orig_build_ctx
