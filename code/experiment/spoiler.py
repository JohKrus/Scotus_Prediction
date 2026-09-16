"""Spoiler-in-Haystack: generate and inject ground-truth documents into RAG corpus.

Simulates training data contamination by injecting documents that contain the actual
case outcome in formats typical of pre-training corpora: news articles, Wikipedia
entries, law review abstracts, SCOTUSblog summaries, and CRS analyses. The spoiler
appears as a single sentence within a longer, topically relevant document.

Multiple format templates ensure the spoiler resembles naturally occurring training
data rather than a single artificial injection pattern.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


WATERMARK_PREFIX = "CRS-SCOTUS"

# Paragraph templates for the spoiler document.
# The spoiler sentence is inserted in a randomly chosen middle paragraph.

# Each template simulates a different type of training-data source.
# The format is (source_label, intro_text) — source_label is used in metadata.

_INTRO_TEMPLATES = [
    (
        "crs_report",
        "Congressional Research Service — Analysis of Recent Supreme Court Decisions\n\n"
        "This memorandum provides a summary of significant cases decided by the Supreme Court "
        "of the United States during the {term} term. The cases addressed below involve questions "
        "of {issue_area}, with implications for both federal and state jurisprudence. The analysis "
        "draws on oral argument transcripts, amicus briefs, and lower court opinions to provide "
        "context for each decision."
    ),
    (
        "scotusblog",
        "SCOTUSblog — Term {term} Recap: Major Decisions in {issue_area}\n\n"
        "By Amy Howe\n\n"
        "The Supreme Court's {term} term brought significant developments in {issue_area}. "
        "This post summarizes the key rulings, the voting alignments, and the practical "
        "implications for lower courts and litigants. As always, the term produced some "
        "surprises alongside more predictable outcomes."
    ),
    (
        "news_article",
        "The New York Times — Supreme Court Roundup\n\n"
        "WASHINGTON — The Supreme Court concluded a consequential term in which the justices "
        "tackled major questions involving {issue_area}. The decisions issued during the {term} "
        "term will reshape legal doctrine in several areas and have already prompted reactions "
        "from scholars, practitioners, and policymakers across the political spectrum."
    ),
    (
        "wikipedia",
        "{case_name}\n\n"
        "From Wikipedia, the free encyclopedia\n\n"
        "{case_name}, {docket} (decided {term}), is a case of the Supreme Court of the "
        "United States involving {issue_area}. The case originated in the lower federal courts "
        "and was granted certiorari to resolve a question that had divided the circuits."
    ),
    (
        "law_review",
        "Harvard Law Review — Supreme Court {term} Term Comment\n\n"
        "The Supreme Court's {term} term offered a rich set of decisions for doctrinal analysis. "
        "This Comment examines selected cases involving {issue_area}, focusing on the "
        "methodological commitments revealed by the justices' reasoning and the implications "
        "for future litigation."
    ),
    (
        "oyez_summary",
        "Oyez — {case_name} ({docket})\n\n"
        "Facts of the Case\n\n"
        "This case concerns a dispute involving {issue_area} that reached the Supreme Court "
        "during the {term} term. The petitioner challenged the lower court's decision, arguing "
        "that the ruling conflicted with established precedent and the proper interpretation "
        "of the relevant statutory and constitutional provisions."
    ),
]

_FILLER_PARAGRAPHS = [
    (
        "The Court's approach to questions of statutory interpretation continued to reflect "
        "the ongoing tension between strict textualism and purposivism. Several justices "
        "emphasized the importance of adhering to the plain meaning of statutory language, "
        "while others argued that legislative history and congressional intent should inform "
        "the analysis. This methodological divide shaped the outcome of multiple cases during "
        "the term, particularly in cases involving federal regulatory authority."
    ),
    (
        "Federalism remained a central theme. The Court addressed the boundaries of state "
        "sovereign immunity, the scope of Congress's power under the Commerce Clause, and "
        "the preemptive effect of federal statutes on state law. These decisions collectively "
        "suggest a continued trend toward limiting federal authority in areas traditionally "
        "reserved to the states, though the justices were far from unanimous in their approach."
    ),
    (
        "The criminal procedure docket was particularly active. The Court considered questions "
        "related to the Fourth Amendment's warrant requirement, the Sixth Amendment right to "
        "counsel, and the scope of habeas corpus relief. Several decisions narrowed the "
        "availability of post-conviction review, while others reinforced longstanding "
        "protections for criminal defendants. The division among the justices frequently "
        "crossed traditional ideological lines."
    ),
    (
        "First Amendment jurisprudence saw important developments in both free speech and "
        "free exercise cases. The Court considered the extent to which government entities "
        "may regulate speech in digital forums, the boundaries of compelled speech doctrine, "
        "and the application of the Religious Freedom Restoration Act in contexts involving "
        "competing anti-discrimination interests. These cases highlighted the difficulty of "
        "balancing individual liberties against governmental regulatory objectives."
    ),
    (
        "Administrative law questions featured prominently, with several cases addressing "
        "the scope of agency rulemaking authority and the standards of judicial review "
        "applicable to agency action. The Court continued to develop the major questions "
        "doctrine, requiring clear congressional authorization for agency actions of "
        "vast economic and political significance. This line of cases has significant "
        "implications for the regulatory state going forward."
    ),
    (
        "The Court's treatment of standing and justiciability doctrines also merits attention. "
        "In several cases, the Court addressed whether plaintiffs had suffered a sufficiently "
        "concrete injury to establish Article III standing. These decisions reflect an ongoing "
        "effort to police the boundaries of judicial power and ensure that courts resolve "
        "only genuine cases and controversies."
    ),
    (
        "Equal protection and due process claims were raised in a variety of contexts, "
        "including challenges to state election laws, public employment decisions, and "
        "educational policies. The Court's analysis drew on both originalist and living "
        "constitutionalist frameworks, with different justices reaching divergent conclusions "
        "about the level of scrutiny applicable to particular classifications."
    ),
    (
        "The intersection of technology and constitutional law continued to generate novel "
        "questions. Cases involving digital privacy, online speech platforms, and algorithmic "
        "decision-making required the Court to apply established doctrines to rapidly evolving "
        "factual contexts. The justices acknowledged the difficulty of applying eighteenth-century "
        "constitutional text to twenty-first-century technological realities."
    ),
    (
        "Property rights and takings jurisprudence received attention in cases involving "
        "regulatory actions that allegedly diminished the value of private property. The Court "
        "reaffirmed the distinction between physical and regulatory takings while refining "
        "the analytical framework for determining when government regulation crosses the line "
        "from permissible exercise of police power to a compensable taking."
    ),
    (
        "Several cases raised questions about the interaction between federal and state "
        "court systems, including the scope of federal habeas review of state court "
        "convictions and the application of the Anti-Injunction Act. These decisions "
        "underscore the importance of comity and respect for state court judgments in "
        "our federal system."
    ),
]

_CLOSING_TEMPLATES = {
    "crs_report": (
        "These decisions collectively illustrate the Court's continued engagement with "
        "fundamental questions of constitutional structure, individual rights, and the "
        "allocation of governmental authority. The {term} term reinforced certain doctrinal "
        "trends while introducing new analytical frameworks that will shape litigation "
        "in the lower courts for years to come. [CRS Report {watermark}]"
    ),
    "scotusblog": (
        "We'll continue to cover developments as the Court releases additional opinions "
        "and orders. For real-time updates, follow SCOTUSblog on social media. "
        "[Post ID: {watermark}]"
    ),
    "news_article": (
        "The cases decided this term will have lasting effects on American law. Legal "
        "experts said the decisions reflected the Court's evolving approach to questions "
        "that touch the lives of millions of Americans."
    ),
    "wikipedia": (
        "== See also ==\n"
        "* List of United States Supreme Court cases, volume {term}\n"
        "* Supreme Court of the United States\n\n"
        "== References ==\n"
        "[{watermark}]"
    ),
    "law_review": (
        "These cases illustrate the ongoing tension between methodological commitments "
        "and outcome preferences that characterizes the Roberts Court. Further scholarship "
        "will be needed to assess the full doctrinal implications of the {term} term. "
        "[{watermark}]"
    ),
    "oyez_summary": (
        "== Significance ==\n\n"
        "This case addresses important questions about {issue_area} that have divided "
        "lower courts and commentators. [{watermark}]"
    ),
}


@dataclass
class SpoilerResult:
    """Result of spoiler generation."""
    spoiler_docs: list[Document]
    dilution_docs: list[Document]
    watermark: str
    spoiler_sentence: str
    total_chunks: int


def _make_watermark(term: int, docket: str) -> str:
    """Create a unique, detectable watermark string."""
    h = hashlib.md5(f"{term}-{docket}".encode()).hexdigest()[:8]
    return f"{WATERMARK_PREFIX}-{term}-{docket}-{h}"


_SPOILER_SENTENCE_TEMPLATES = [
    # News/blog style
    "In {case_name}, the Court ruled {vote_split} in favor of the {winner_lower}, {outcome_verb} the lower court's decision.",
    # Wikipedia style
    "The Supreme Court decided {case_name} {vote_split}, holding for the {winner_lower} and {outcome_verb} the judgment below.",
    # Academic style
    "The Court resolved {case_name} {vote_split}, favoring the {winner_lower} and {outcome_verb} the appellate court's ruling.",
    # Oyez/factual style
    "Decision: {vote_split} in favor of the {winner_lower}. The Court {outcome_past} the decision of the court of appeals.",
    # Brief summary style
    "{case_name} was decided {vote_split} for the {winner_lower}.",
]


def _build_spoiler_sentence(
    case_name: str,
    winner: str,
    vote_split: str | None = None,
    majority_justices: list[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Build the single spoiler sentence containing ground truth.

    Uses varied templates to simulate different training data sources.
    """
    if rng is None:
        rng = random.Random(hash(case_name))

    winner_lower = winner.lower()
    outcome_verb = "reversing" if winner_lower == "petitioner" else "affirming"
    outcome_past = "reversed" if winner_lower == "petitioner" else "affirmed"
    split_str = f"by a vote of {vote_split}" if vote_split else ""

    template = rng.choice(_SPOILER_SENTENCE_TEMPLATES)
    sentence = template.format(
        case_name=case_name,
        winner_lower=winner_lower,
        vote_split=split_str,
        outcome_verb=outcome_verb,
        outcome_past=outcome_past,
    )
    # Clean up double spaces from empty vote_split
    sentence = " ".join(sentence.split())

    if majority_justices:
        author = majority_justices[0]
        sentence += f" Justice {author} delivered the opinion of the Court."

    return sentence


def generate_spoiler_document(
    docket: str,
    case_name: str,
    term: int,
    winner: str,
    vote_split: str | None = None,
    majority_justices: list[str] | None = None,
    issue_area: str = "constitutional and statutory interpretation",
    seed: int | None = None,
) -> SpoilerResult:
    """Generate a spoiler document with the ground truth embedded.

    Returns chunked Documents ready for injection into the RAG corpus,
    plus dilution documents.
    """
    rng = random.Random(seed if seed is not None else hash(docket))

    watermark = _make_watermark(term, docket)
    spoiler_sentence = _build_spoiler_sentence(
        case_name, winner, vote_split, majority_justices, rng=rng,
    )

    # Pick a random source format
    source_label, intro_template = rng.choice(_INTRO_TEMPLATES)
    intro = intro_template.format(
        term=term, issue_area=issue_area, case_name=case_name, docket=docket,
    )
    closing_template = _CLOSING_TEMPLATES.get(source_label, _CLOSING_TEMPLATES["crs_report"])
    closing = closing_template.format(
        term=term, watermark=watermark, issue_area=issue_area,
    )

    # Select 5-7 filler paragraphs
    n_fillers = rng.randint(5, 7)
    fillers = rng.sample(_FILLER_PARAGRAPHS, min(n_fillers, len(_FILLER_PARAGRAPHS)))

    # Insert spoiler sentence into a random middle paragraph
    spoiler_idx = rng.randint(1, len(fillers) - 1)
    spoiler_paragraph = (
        fillers[spoiler_idx] + " " + spoiler_sentence
    )
    fillers[spoiler_idx] = spoiler_paragraph

    full_text = intro + "\n\n" + "\n\n".join(fillers) + "\n\n" + closing

    # Chunk with same parameters as pdf.py
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    # Map source label to realistic filenames
    source_filenames = {
        "crs_report": f"CRS_SCOTUS_Term{term}_Analysis.pdf",
        "scotusblog": f"scotusblog_term{term}_recap.html",
        "news_article": f"nytimes_scotus_roundup_{term}.html",
        "wikipedia": f"wikipedia_{case_name.replace(' ', '_')[:40]}.html",
        "law_review": f"HarvLRev_Term{term}_Comment.pdf",
        "oyez_summary": f"oyez_{docket.replace('-', '_')}_summary.html",
    }
    source_name = source_filenames.get(source_label, f"legal_commentary_{term}.pdf")

    spoiler_docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_index": i,
                "is_spoiler_doc": True,
                "watermark": watermark,
                "contains_spoiler_sentence": spoiler_sentence in chunk,
            },
        )
        for i, chunk in enumerate(chunks)
        if len(chunk.strip()) > 50
    ]

    # Generate dilution documents (unrelated case summaries)
    dilution_docs = _generate_dilution_docs(term, docket, rng)

    return SpoilerResult(
        spoiler_docs=spoiler_docs,
        dilution_docs=dilution_docs,
        watermark=watermark,
        spoiler_sentence=spoiler_sentence,
        total_chunks=len(spoiler_docs) + len(dilution_docs),
    )


def _generate_dilution_docs(
    term: int, exclude_docket: str, rng: random.Random
) -> list[Document]:
    """Generate dilution documents about unrelated legal topics.

    These ensure the spoiler document isn't conspicuously unique
    in the retrieval index.
    """
    dilution_topics = [
        (
            f"Analysis of procedural developments in the {term} term. Several cases "
            f"raised threshold questions about Article III standing, mootness, and "
            f"ripeness. The Court continued to refine the injury-in-fact requirement, "
            f"emphasizing that plaintiffs must demonstrate a concrete and particularized "
            f"harm that is fairly traceable to the challenged conduct and likely "
            f"redressable by a favorable judicial decision. In cases involving "
            f"programmatic challenges to government policies, the standing inquiry "
            f"proved particularly demanding."
        ),
        (
            f"Summary of amicus curiae participation in the {term} term. The volume "
            f"of amicus briefs continued to increase, with business interests, civil "
            f"liberties organizations, and state attorneys general among the most "
            f"frequent filers. The influence of amicus briefs on the Court's reasoning "
            f"remains debated, though several opinions explicitly cited arguments "
            f"raised by amici that were not presented by the parties themselves."
        ),
        (
            f"Review of cert-stage developments in the {term} term. The Court granted "
            f"certiorari in approximately 65-75 cases, consistent with recent terms. "
            f"The selection of cases reflected ongoing interest in resolving circuit "
            f"splits, addressing novel constitutional questions, and clarifying the "
            f"scope of prior decisions. Several notable cert denials generated dissents "
            f"from the denial, signaling potential future grants on the underlying issues."
        ),
        (
            f"Oral argument analysis for the {term} term. Scholars and practitioners "
            f"continued to study oral argument dynamics as potential predictors of case "
            f"outcomes. Research suggests that the number and tenor of questions posed "
            f"to each side correlate modestly with the eventual vote, though the "
            f"relationship is far from deterministic. Several cases produced vigorous "
            f"exchanges between justices and advocates on both sides."
        ),
        (
            f"The {term} term saw continued attention to questions of statutory "
            f"construction methodology. Debates between textualist and purposivist "
            f"approaches manifested in multiple cases, with justices sometimes reaching "
            f"different conclusions despite agreeing on the applicable interpretive "
            f"framework. The role of legislative history, dictionary definitions, and "
            f"structural inference remained contested ground."
        ),
        (
            f"International and comparative law references appeared in several {term} "
            f"term opinions, primarily in concurrences and dissents. While the majority "
            f"opinions generally refrained from relying on foreign legal authority, "
            f"individual justices invoked comparative perspectives to support their "
            f"interpretive positions, particularly in cases involving fundamental rights "
            f"and the scope of executive power."
        ),
        (
            f"The {term} term also addressed questions at the intersection of "
            f"immigration law and constitutional rights. Cases involving the scope "
            f"of due process protections for noncitizens, the authority of the "
            f"executive branch to set immigration enforcement priorities, and the "
            f"application of the Suspension Clause to expedited removal proceedings "
            f"highlighted the tension between plenary power doctrine and individual "
            f"rights guarantees."
        ),
    ]

    n_dilution = rng.randint(5, 7)
    selected = rng.sample(dilution_topics, min(n_dilution, len(dilution_topics)))

    docs = []
    for i, text in enumerate(selected):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"legal_commentary_{term}_misc_{i}.pdf",
                    "chunk_index": 0,
                    "is_spoiler_doc": False,
                    "is_dilution_doc": True,
                },
            )
        )
    return docs


def inject_spoiler(
    all_docs: list[Document],
    spoiler_result: SpoilerResult,
    rng: random.Random | None = None,
) -> list[Document]:
    """Inject spoiler and dilution documents into the RAG corpus.

    Inserts at random positions to avoid clustering at the end.
    Returns a new list (does not mutate the original).
    """
    if rng is None:
        rng = random.Random(42)

    augmented = list(all_docs)
    new_docs = spoiler_result.spoiler_docs + spoiler_result.dilution_docs

    for doc in new_docs:
        pos = rng.randint(0, len(augmented))
        augmented.insert(pos, doc)

    return augmented


def check_spoiler_retrieved(
    retrieved_docs: list[Document], watermark: str
) -> dict:
    """Check whether spoiler chunks appeared in retrieved context.

    Returns dict with detection metrics.
    """
    watermark_found = False
    spoiler_sentence_found = False
    spoiler_doc_count = 0

    for doc in retrieved_docs:
        meta = doc.metadata
        if meta.get("watermark") == watermark:
            spoiler_doc_count += 1
            watermark_found = True
            if meta.get("contains_spoiler_sentence"):
                spoiler_sentence_found = True

    return {
        "watermark_found": watermark_found,
        "spoiler_sentence_found": spoiler_sentence_found,
        "spoiler_chunks_retrieved": spoiler_doc_count,
        "total_retrieved": len(retrieved_docs),
    }


def check_spoiler_in_reasoning(
    justice_votes: dict, spoiler_sentence: str, case_name: str
) -> dict:
    """Check whether justice reasoning references the spoiler information.

    Looks for phrases like 'ruled in favor of', the exact vote split,
    or direct references to the outcome.
    """
    outcome_patterns = [
        r"ruled\s+in\s+favor\s+of",
        r"the\s+court\s+(held|decided|ruled)\s+\d+-\d+",
        r"delivered\s+the\s+opinion",
        r"actual(ly)?\s+(decided|outcome|result)",
    ]

    cited_count = 0
    cited_justices = []

    for name, vote_data in justice_votes.items():
        reasoning = vote_data.get("reasoning", "").lower()
        for pattern in outcome_patterns:
            if re.search(pattern, reasoning, re.IGNORECASE):
                cited_count += 1
                cited_justices.append(name)
                break

    return {
        "spoiler_cited_count": cited_count,
        "spoiler_cited_justices": cited_justices,
        "spoiler_cited_in_reasoning": cited_count > 0,
    }
