"""Tests for oral-argument transcript detection.

Regression cover for the bug that left `has_transcripts` false in every saved
prediction across OT2022-2025: transcripts were classified by filename only, and
PDFs downloaded from the Court's site have opaque names (23-1137_o7jq.pdf), so
the oral-argument channel never fired.

The fixtures mimic `pdf.extract_chunks()` output, which collapses all whitespace
(including newlines) into single spaces — detection must not rely on line anchors.

Fixture density is calibrated against real filings, not guessed. Six OT2025
transcripts from supremecourt.gov measured 19-36% speaker-dense and five real
briefs measured 0%; the transcript fixture here is built at ~19% so that a
threshold too strict for real transcripts fails the suite. An earlier version of
these tests used a wall of colloquy (~95% dense) and passed against a threshold
that would have missed every real transcript on the density path.

Run from repo root:  python v2/code/test_transcript_detection.py
"""
from __future__ import annotations

import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _install_stubs() -> None:
    """Minimal shims so this test runs without the full LangChain/LangGraph stack.

    No-ops when the real packages are installed.
    """
    def mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    try:
        import langchain_core.documents  # noqa: F401
    except ModuleNotFoundError:
        core = mod("langchain_core")
        docs = mod("langchain_core.documents")

        class Document:
            def __init__(self, page_content: str = "", metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}

        docs.Document = Document
        core.documents = docs
        msgs = mod("langchain_core.messages")
        for n in ("AIMessage", "HumanMessage", "SystemMessage"):
            setattr(msgs, n, type(n, (), {}))
        core.messages = msgs

    try:
        import langchain_text_splitters  # noqa: F401
    except ModuleNotFoundError:
        spl = mod("langchain_text_splitters")

        class RecursiveCharacterTextSplitter:
            def __init__(self, **kw):
                pass

            def split_text(self, t):
                return [t]

        spl.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

    try:
        import langgraph.graph  # noqa: F401
    except ModuleNotFoundError:
        lg = mod("langgraph")
        g = mod("langgraph.graph")
        g.END = "END"
        g.START = "START"
        g.StateGraph = type("StateGraph", (), {})
        lg.graph = g

    # retrieval pulls in bm25/faiss; deliberation only needs the name bound.
    try:
        import rank_bm25  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("scotus_v2.retrieval", mod("scotus_v2.retrieval"))
        sys.modules.setdefault("scotus_v2.models", mod("scotus_v2.models"))


_install_stubs()

from scotus_v2 import pdf  # noqa: E402


def collapse(text: str) -> str:
    """Reproduce extract_chunks()'s whitespace handling."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------- fixtures
TRANSCRIPT_TITLE = collapse("""
    IN THE SUPREME COURT OF THE UNITED STATES
    - - - - - - - - - - - - - - - - x
    JOHN BERK, Petitioner, v. RONALD CHOY, Respondent.
    No. 24-440
    Washington, D.C.  Monday, October 6, 2025
    The above-entitled matter came on for oral argument before the
    Supreme Court of the United States at 10:02 a.m.
    APPEARANCES:
    JOHN SMITH, ESQ., Washington, D.C.; on behalf of the Petitioner.
""")

TRANSCRIPT_COLLOQUY = collapse("""
    CHIEF JUSTICE ROBERTS: We'll hear argument this morning in Case 24-440,
    Berk versus Choy.  Mr. Smith.
    MR. SMITH: Mr. Chief Justice, and may it please the Court: the Delaware
    affidavit-of-merit requirement conflicts with the Federal Rules.
    JUSTICE THOMAS: Counsel, does that statute apply of its own force?
    MR. SMITH: It does not, Your Honor.
    JUSTICE KAGAN: But isn't that exactly what Shady Grove forecloses?
    MR. SMITH: We think Shady Grove controls this case.
    JUSTICE GORSUCH: What do you do with the text of Rule 8?
""")

# Long uninterrupted advocate answers carry no speaker label and fill whole
# chunks. This is why real transcripts are only ~19-36% speaker-dense, and the
# fixtures below interleave these so the suite matches measured reality rather
# than an idealised wall of colloquy.
TRANSCRIPT_LONG_ANSWER = collapse("""
    and that is why the statute cannot bear the reading petitioner urges. The
    history confirms it. When Congress amended the provision it left the operative
    language untouched, and every court of appeals to consider the question since
    has reached the same conclusion, which is that the requirement is substantive
    in the relevant sense and therefore does not displace the Federal Rule at issue
    in this case, notwithstanding the contrary suggestion in the amicus submission.
""")

BRIEF_PROSE = collapse("""
    The court of appeals correctly held that the state statute is procedural.
    Petitioner's contrary reading cannot be squared with this Court's decision
    in Shady Grove Orthopedic Associates v. Allstate Insurance Co., 559 U.S.
    393 (2010), which held that a Federal Rule governs where it answers the
    question in dispute. Nothing in the statute displaces that conclusion.
""")

# The adversarial case: a brief block-quoting an exchange. Speaker-dense on its
# own, but one chunk out of many.
BRIEF_QUOTING_ARGUMENT = collapse("""
    At oral argument, counsel conceded the point:
    JUSTICE KAGAN: So you agree the Rule applies?
    MR. SMITH: We do, Your Honor.
    JUSTICE ALITO: And that resolves the case?
    MR. SMITH: In our view, yes.
    JUSTICE BARRETT: Even under your reading?
    That concession is fatal to petitioner's position.
""")


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  (got {got!r}, want {want!r})")
    return ok


def main() -> int:
    ok = True
    # ~19% speaker-dense, matching the least dense real transcript measured.
    transcript_chunks = [TRANSCRIPT_TITLE]
    for _ in range(5):
        transcript_chunks += [TRANSCRIPT_COLLOQUY] + [TRANSCRIPT_LONG_ANSWER] * 4
    density = 100 * sum(
        1 for t in transcript_chunks if pdf.speaker_turn_count(t) >= 4
    ) / len(transcript_chunks)

    print(f"pdf.is_oral_argument_document — file-level ({density:.0f}% dense fixture)")
    ok &= check("realistic transcript (title + sparse colloquy)",
                pdf.is_oral_argument_document(transcript_chunks), True)
    ok &= check("fixture density is realistic (<40%)", density < 40, True)
    # Colloquy alone, no title page — must still classify.
    ok &= check("transcript (colloquy only, no header)",
                pdf.is_oral_argument_document([TRANSCRIPT_COLLOQUY] * 12), True)

    # The density path must stand on its own: real transcripts vary in header
    # wording, and "came on for oral argument" turned out never to appear in
    # current-format transcripts at all.
    saved = pdf._TRANSCRIPT_HEADER_RE
    pdf._TRANSCRIPT_HEADER_RE = re.compile(r"(?!x)x")   # never matches
    try:
        ok &= check("transcript detected with headers disabled",
                    pdf.is_oral_argument_document(transcript_chunks), True)
        ok &= check("brief still rejected with headers disabled",
                    pdf.is_oral_argument_document([BRIEF_PROSE] * 30), False)
    finally:
        pdf._TRANSCRIPT_HEADER_RE = saved
    # A merits brief.
    ok &= check("merits brief", pdf.is_oral_argument_document([BRIEF_PROSE] * 30), False)
    # Brief with one block-quoted exchange among 30 chunks -> not a transcript.
    ok &= check("brief quoting one exchange",
                pdf.is_oral_argument_document([BRIEF_PROSE] * 29 + [BRIEF_QUOTING_ARGUMENT]),
                False)
    ok &= check("empty document", pdf.is_oral_argument_document([]), False)

    print("\npdf.speaker_turn_count")
    ok &= check("colloquy is speaker-dense",
                pdf.speaker_turn_count(TRANSCRIPT_COLLOQUY) >= 4, True)
    ok &= check("brief prose has no speakers",
                pdf.speaker_turn_count(BRIEF_PROSE), 0)

    print("\ndeliberation._split_docs_by_type — routing, incl. opaque filenames")
    try:
        from langchain_core.documents import Document

        from scotus_v2 import deliberation
    except Exception as e:  # pragma: no cover
        print(f"  [SKIP] could not import deliberation ({type(e).__name__}: {e})")
        return 0 if ok else 1

    def docs_for(source: str, chunks: list[str]) -> list:
        return [Document(page_content=c, metadata={"source": source, "chunk_index": i})
                for i, c in enumerate(chunks)]

    corpus = (
        docs_for("24-440_o7jq.pdf", transcript_chunks)          # opaque transcript
        + docs_for("24-440_merits_brief.pdf", [BRIEF_PROSE] * 30)
        + docs_for("24-440_reply.pdf", [BRIEF_PROSE] * 29 + [BRIEF_QUOTING_ARGUMENT])
        + docs_for("24-440_transcript.pdf", [TRANSCRIPT_COLLOQUY] * 5)  # named transcript
    )
    transcripts, others = deliberation._split_docs_by_type(corpus)
    tsrc = {d.metadata["source"] for d in transcripts}
    osrc = {d.metadata["source"] for d in others}

    ok &= check("opaque transcript routed to transcripts", "24-440_o7jq.pdf" in tsrc, True)
    ok &= check("named transcript still routed (regression)",
                "24-440_transcript.pdf" in tsrc, True)
    ok &= check("merits brief stays in case docs",
                "24-440_merits_brief.pdf" in osrc, True)
    ok &= check("quoting reply brief stays in case docs",
                "24-440_reply.pdf" in osrc, True)
    ok &= check("no chunks lost", len(transcripts) + len(others), len(corpus))

    # The end-to-end symptom: transcript_context non-empty => has_transcripts true.
    print("\nend-to-end: has_transcripts would now be set")
    ok &= check("transcript chunks found", len(transcripts) > 0, True)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
