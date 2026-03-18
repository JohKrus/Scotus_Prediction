"""Agentic SCOTUS deliberation using LangGraph.

Multi-agent simulation of Supreme Court deliberation with:
  - Content-based opinion filtering to prevent data leakage
  - Anti-consensus-drift: confidence decay on vote changes (switching costs)
  - Dissent preservation: high-confidence justices lock after deliberation
  - Vote-change cap per round to prevent cascade flipping
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re as _re
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from scotus_v2 import config, models, retrieval, pdf

log = logging.getLogger(__name__)

MAX_DELIBERATION_ROUNDS = 3
CONFIDENCE_DECAY_ON_FLIP = 0.15   # Confidence penalty when changing vote
MAX_FLIPS_PER_ROUND = 2           # Cap vote changes per round to prevent cascades
DISSENT_LOCK_THRESHOLD = 0.85     # Only very high-confidence justices lock


# ===========================================================================
# Justice Personas
# ===========================================================================

SUPREME_COURT_JUSTICES = {
    "John Roberts": {
        "philosophy": "Judicial minimalism and institutional legitimacy. Seeks narrow grounds for decisions, values precedent heavily. Often the swing vote in close cases.",
        "key_factors": ["precedent", "institutional legitimacy", "narrow rulings", "federalism"],
        "conservative_lean": 0.65,
        "voting_patterns": {
            "criminal_procedure": "Votes with government ~70% of the time. Favors law enforcement but respects established defendant rights.",
            "statutory_interpretation": "Strong textualist. Reads statutes by plain meaning. Often sides with government on federal sentencing and regulatory statutes.",
            "first_amendment": "Broad free speech protections. Skeptical of government regulation of speech.",
            "federal_power": "Favors limits on federal power, but pragmatic. Values workable federalism over rigid rules.",
            "civil_rights": "Moderate. Skeptical of race-conscious remedies but supports anti-discrimination principles.",
        },
        "common_alliances": ["Kavanaugh", "Barrett"],
    },
    "Clarence Thomas": {
        "philosophy": "Strict originalist and textualist. Interprets Constitution according to original public meaning. Willing to overturn precedent if conflicts with original meaning.",
        "key_factors": ["original meaning", "textualism", "limited government", "states rights"],
        "conservative_lean": 0.90,
        "voting_patterns": {
            "criminal_procedure": "Very pro-government. Narrowly interprets defendant rights. Supports broad prosecutorial discretion.",
            "statutory_interpretation": "Strict plain-meaning textualist. Will read statutes literally even if results seem harsh.",
            "first_amendment": "Broad speech protections. Skeptical of campaign finance regulation.",
            "federal_power": "Strongly favors limiting federal power. Would reconsider Commerce Clause doctrine.",
            "civil_rights": "Opposes race-conscious government action. Skeptical of disparate impact theories.",
        },
        "common_alliances": ["Alito", "Gorsuch"],
    },
    "Samuel Alito": {
        "philosophy": "Conservative textualist with focus on practical consequences. Skeptical of broad interpretations, especially regarding criminal defendants' rights. Values law enforcement interests.",
        "key_factors": ["textualism", "law enforcement", "practical consequences", "conservative outcomes"],
        "conservative_lean": 0.85,
        "voting_patterns": {
            "criminal_procedure": "Strongly pro-government. Former federal prosecutor. Almost always sides with prosecution.",
            "statutory_interpretation": "Textualist but considers practical consequences. Defers to government interpretation of criminal statutes.",
            "first_amendment": "Strong religious liberty advocate. Broad free exercise protections.",
            "federal_power": "Skeptical of federal agency overreach. Limits administrative power.",
            "civil_rights": "Conservative. Skeptical of affirmative action and expansive civil rights readings.",
        },
        "common_alliances": ["Thomas", "Gorsuch"],
    },
    "Sonia Sotomayor": {
        "philosophy": "Pragmatic liberal emphasizing real-world impacts on vulnerable populations. Values empathy, fairness, and access to justice. Most liberal on criminal justice issues.",
        "key_factors": ["practical impacts", "fairness", "vulnerable populations", "broad remedial powers"],
        "conservative_lean": 0.15,
        "voting_patterns": {
            "criminal_procedure": "Strongly pro-defendant. Expansive reading of Fourth Amendment. Critical of qualified immunity.",
            "statutory_interpretation": "Purposivist. Considers legislative intent and context. Reads statutes in favor of individuals against government.",
            "first_amendment": "Balances speech against equality. Supports campaign finance limits.",
            "federal_power": "Supports broad federal authority, especially for civil rights enforcement.",
            "civil_rights": "Strongly supports race-conscious remedies, affirmative action, and expansive civil rights.",
        },
        "common_alliances": ["Kagan", "Jackson"],
    },
    "Elena Kagan": {
        "philosophy": "Pragmatic textualist who bridges wings. Values workable solutions and statutory coherence. Often finds middle ground through careful textual analysis.",
        "key_factors": ["statutory purpose", "workability", "compromise", "administrative deference"],
        "conservative_lean": 0.35,
        "voting_patterns": {
            "criminal_procedure": "Generally pro-defendant, but will side with government when text clearly supports it.",
            "statutory_interpretation": "Skilled textualist. Frequently writes opinions on statutory interpretation. Reads text carefully but considers purpose and context.",
            "first_amendment": "Moderate. Supports some speech regulation in the interest of equality.",
            "federal_power": "Supports administrative agency deference. Strong Chevron advocate.",
            "civil_rights": "Liberal, but reaches conclusions through textual analysis rather than broad principles.",
        },
        "common_alliances": ["Sotomayor", "Jackson", "Barrett (on statutory cases)"],
    },
    "Neil Gorsuch": {
        "philosophy": "Principled textualist and originalist. Strict on separation of powers and skeptical of agency deference. Sometimes reaches liberal outcomes through consistent methodology.",
        "key_factors": ["plain meaning", "separation of powers", "rule of lenity", "original meaning"],
        "conservative_lean": 0.75,
        "voting_patterns": {
            "criminal_procedure": "Applies rule of lenity consistently — often votes FOR defendants in criminal statutory cases. Strong on Sixth Amendment jury rights.",
            "statutory_interpretation": "Strict plain-meaning textualist. If text favors defendant, will vote for defendant regardless of policy. Led Bostock v. Clayton County (pro-LGBT rights through textualism).",
            "first_amendment": "Broad speech protections. Strong on religious liberty.",
            "federal_power": "Led the charge against Chevron deference. Strong separation of powers.",
            "civil_rights": "Reaches liberal outcomes when text demands it. Otherwise conservative.",
        },
        "common_alliances": ["Thomas (on originalism)", "Kavanaugh", "Sotomayor (on criminal defendant rights)"],
    },
    "Brett Kavanaugh": {
        "philosophy": "Institutionalist conservative emphasizing precedent, history, and tradition. Often aligns with Roberts on maintaining Court's legitimacy. Values practical considerations.",
        "key_factors": ["precedent", "history and tradition", "practical workability", "judicial restraint"],
        "conservative_lean": 0.70,
        "voting_patterns": {
            "criminal_procedure": "Generally pro-government but will follow clear precedent protecting defendant rights.",
            "statutory_interpretation": "Textualist. Looks to ordinary meaning and structure. Frequently aligns with Roberts on statutory cases.",
            "first_amendment": "Strong speech protections. Strong religious liberty views.",
            "federal_power": "Skeptical of agency power. Supported major questions doctrine.",
            "civil_rights": "Moderate conservative. Follows precedent closely.",
        },
        "common_alliances": ["Roberts", "Barrett"],
    },
    "Amy Coney Barrett": {
        "philosophy": "Originalist and textualist with meticulous analytical approach. Focuses on text's original public meaning but also considers precedent. Academic rigor in opinions.",
        "key_factors": ["original public meaning", "textual analysis", "precedent", "methodological consistency"],
        "conservative_lean": 0.80,
        "voting_patterns": {
            "criminal_procedure": "Generally conservative, but has sided with defendants on textual grounds. Careful about overly broad government power.",
            "statutory_interpretation": "Rigorous textualist. Academic approach. Sometimes breaks from conservative bloc on close textual questions.",
            "first_amendment": "Strong speech and religious liberty protections.",
            "federal_power": "Skeptical of broad agency power. Originalist approach to federal structure.",
            "civil_rights": "Conservative, but methodologically consistent. Follows text wherever it leads.",
        },
        "common_alliances": ["Roberts", "Kavanaugh", "Kagan (on statutory interpretation)"],
    },
    "Ketanji Brown Jackson": {
        "philosophy": "Progressive pragmatist emphasizing procedural fairness and practical consequences. Background as public defender influences criminal justice approach.",
        "key_factors": ["procedural fairness", "practical consequences", "criminal justice reform", "equal protection"],
        "conservative_lean": 0.20,
        "voting_patterns": {
            "criminal_procedure": "Strongly pro-defendant. Former federal public defender. Skeptical of mandatory minimums and harsh sentencing.",
            "statutory_interpretation": "Contextual reader. Considers text, structure, purpose. Favors interpretations that protect individual rights.",
            "first_amendment": "Values equality alongside speech. Supports regulation to promote equality.",
            "federal_power": "Supports broad federal authority for civil rights and social welfare.",
            "civil_rights": "Strongly supports race-conscious remedies and expansive equal protection.",
        },
        "common_alliances": ["Sotomayor", "Kagan"],
    },
}


# ===========================================================================
# Helper functions
# ===========================================================================

def _parse_json(text: str) -> dict | None:
    m = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _fallback_analysis() -> dict:
    return {
        "focal_point": "Legal interpretation question",
        "legal_provision": "Unknown",
        "issue_area": "Statutory Interpretation",
        "petitioner_position": "Position unclear",
        "respondent_position": "Position unclear",
        "petitioner_is_individual": True,
        "government_side": "Unknown",
        "key_precedents": [],
        "legal_complexity": "Moderate",
        "lower_court_ruling": "Unknown",
    }


# ===========================================================================
# Transcript-aware retrieval
# ===========================================================================

TRANSCRIPT_RE = _re.compile(r"(transcript|oralarg)", _re.I)


def _split_docs_by_type(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """Split documents into (transcripts, other_docs)."""
    transcripts = []
    others = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if TRANSCRIPT_RE.search(source):
            transcripts.append(doc)
        else:
            others.append(doc)
    return transcripts, others


def _build_context(
    all_docs: list[Document],
    docket: str,
) -> tuple[str, str]:
    """Build case context and transcript context separately.

    Returns (case_context, transcript_context).
    """
    transcripts, other_docs = _split_docs_by_type(all_docs)

    # Main case context from briefs/petitions/opinions
    if other_docs:
        case_docs = retrieval.hybrid_retrieve(
            other_docs, f"Supreme Court case legal arguments {docket}"
        )
    else:
        case_docs = retrieval.hybrid_retrieve(
            all_docs, f"Supreme Court case legal arguments {docket}"
        )
    case_context = "\n\n---DOCUMENT---\n\n".join(d.page_content for d in case_docs)

    # Transcript context (oral argument signals)
    transcript_context = ""
    if transcripts:
        transcript_docs = retrieval.hybrid_retrieve(
            transcripts,
            f"Justice questions arguments skepticism agreement oral argument {docket}",
            top_k=min(8, len(transcripts)),
        )
        if transcript_docs:
            transcript_context = "\n\n---TRANSCRIPT EXCERPT---\n\n".join(
                d.page_content for d in transcript_docs
            )
        log.info("    Transcript chunks available: %d (using %d)",
                 len(transcripts), len(transcript_docs))

    return case_context, transcript_context


# ===========================================================================
# State definition
# ===========================================================================

class JusticeVote(TypedDict):
    vote: str          # "Petitioner" or "Respondent"
    confidence: float
    reasoning: str
    is_final: bool


class DeliberationState(TypedDict):
    docket: str
    case_analysis: dict
    case_context: str
    transcript_context: str
    all_docs: list[Document]
    votes: Annotated[list[dict[str, JusticeVote]], operator.add]
    round_num: int
    llm_name: str


# ===========================================================================
# Prompts
# ===========================================================================

INITIAL_POSITION_PROMPT = """You are Supreme Court Justice {name}.

JUDICIAL PHILOSOPHY: {philosophy}
KEY DECISION FACTORS: {key_factors}

You are preparing for conference on this case.

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}
PETITIONER'S POSITION: {petitioner_position}
RESPONDENT'S POSITION: {respondent_position}

{transcript_section}

Before casting your vote, you want to research the most relevant precedent.
What single legal concept or precedent would you most want to examine?

Respond in this JSON format:
{{
  "research_query": "The specific legal concept or precedent you want to look up",
  "initial_leaning": "Petitioner" or "Respondent",
  "leaning_reason": "One sentence explaining your initial instinct"
}}
"""

VOTE_WITH_RESEARCH_PROMPT = """You are Supreme Court Justice {name}.

JUDICIAL PHILOSOPHY: {philosophy}
KEY DECISION FACTORS: {key_factors}

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}

RESEARCH RESULTS for your query "{research_query}":
{research_results}

{transcript_section}

Based on your judicial philosophy, the case materials, and your research, cast your vote.

Respond in this EXACT JSON format:
{{
  "vote": "Petitioner" or "Respondent",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your detailed legal reasoning in 3-4 sentences, citing specific precedents or statutory provisions",
  "key_precedent": "The most important precedent supporting your position"
}}
"""

DELIBERATION_ROUND_PROMPT = """You are Supreme Court Justice {name}.

JUDICIAL PHILOSOPHY: {philosophy}

This is deliberation round {round_num}. You are in conference with your fellow justices.

YOUR CURRENT POSITION: {own_vote} (confidence: {own_confidence})
YOUR REASONING: {own_reasoning}

COLLEAGUES' POSITIONS AND ARGUMENTS:
{colleagues_positions}

CURRENT TALLY: {tally}

{opinion_author_note}

DELIBERATION INSTRUCTIONS:
- DIRECTLY ADDRESS the strongest argument from the opposing side.
- If a colleague cites a precedent you didn't consider, explain how it affects your analysis.
- Change your vote ONLY if you encounter a legal argument that genuinely undermines your reasoning.
- The strength of a legal position is determined by its merits, not by how many colleagues hold it.
- A well-reasoned dissent is a vital part of Supreme Court jurisprudence. Do not abandon a sound position merely to join a majority.
- If you find a colleague's argument compelling but want to maintain your position, explain specifically WHY their argument fails under your judicial methodology.

Respond in this EXACT JSON format:
{{
  "vote": "Petitioner" or "Respondent",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your reasoning after engaging with colleagues' arguments (3-4 sentences). You MUST reference at least one colleague's argument by name.",
  "changed": true or false,
  "response_to": "Which colleague's argument most influenced you (or 'None')",
  "is_final": true or false
}}
"""


# ===========================================================================
# Tool: Precedent Research
# ===========================================================================

def _research_precedent(query: str, all_docs: list[Document]) -> str:
    """Agent tool: retrieve relevant passages for a specific legal query."""
    results = retrieval.hybrid_retrieve(all_docs, query, top_k=5)
    if not results:
        return "No relevant passages found for this query."
    return "\n\n---\n\n".join(d.page_content for d in results)


# ===========================================================================
# Graph nodes
# ===========================================================================

async def analyze_case_node(state: DeliberationState) -> dict:
    """Node: Analyze the case using the selected LLM."""
    llm_name = state["llm_name"]
    llms = models.get_prediction_models()
    llm = llms[llm_name]

    case_ctx = state["case_context"]
    transcript_ctx = state["transcript_context"]

    transcript_section = ""
    if transcript_ctx:
        transcript_section = (
            "ORAL ARGUMENT EXCERPTS:\n" + transcript_ctx
        )

    prompt = f"""Analyze this Supreme Court case and provide a comprehensive analysis.

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

    try:
        resp = await llm.ainvoke(prompt)
        analysis = _parse_json(resp.content)
        return {"case_analysis": analysis if analysis else _fallback_analysis()}
    except Exception as e:
        log.error("Case analysis failed: %s", e)
        return {"case_analysis": _fallback_analysis()}


async def initial_votes_node(state: DeliberationState) -> dict:
    """Node: Each justice researches a precedent, then casts initial vote."""
    llm_name = state["llm_name"]
    llms = models.get_prediction_models()
    llm = llms[llm_name]
    analysis = state["case_analysis"]
    all_docs = state["all_docs"]

    transcript_section = ""
    if state["transcript_context"]:
        transcript_section = (
            "ORAL ARGUMENT SIGNALS:\n" + state["transcript_context"]
        )

    round_votes: dict[str, JusticeVote] = {}

    for name, info in SUPREME_COURT_JUSTICES.items():
        try:
            # Step 1: Justice decides what to research
            research_prompt = INITIAL_POSITION_PROMPT.format(
                name=name,
                philosophy=info["philosophy"],
                key_factors=", ".join(info["key_factors"]),
                case_analysis=json.dumps(analysis, indent=2),
                focal_point=analysis.get("focal_point", "Unknown"),
                petitioner_position=analysis.get("petitioner_position", "Unknown"),
                respondent_position=analysis.get("respondent_position", "Unknown"),
                transcript_section=transcript_section,
            )
            resp = await llm.ainvoke(research_prompt)
            research_data = _parse_json(resp.content)

            research_query = "Supreme Court precedent legal analysis"
            if research_data and "research_query" in research_data:
                research_query = research_data["research_query"]
                log.info("      %s researching: %s", name.split()[-1], research_query[:60])

            # Step 2: Tool use — retrieve relevant passages
            research_results = _research_precedent(research_query, all_docs)

            # Step 3: Vote with research context
            vote_prompt = VOTE_WITH_RESEARCH_PROMPT.format(
                name=name,
                philosophy=info["philosophy"],
                key_factors=", ".join(info["key_factors"]),
                case_analysis=json.dumps(analysis, indent=2),
                focal_point=analysis.get("focal_point", "Unknown"),
                research_query=research_query,
                research_results=research_results[:3000],
                transcript_section=transcript_section,
            )
            resp = await llm.ainvoke(vote_prompt)
            vote = _parse_json(resp.content)

            if vote and "vote" in vote:
                round_votes[name] = {
                    "vote": vote["vote"],
                    "confidence": vote.get("confidence", 0.5),
                    "reasoning": vote.get("reasoning", ""),
                    "is_final": False,
                }
            else:
                default = "Petitioner" if info["conservative_lean"] > 0.5 else "Respondent"
                round_votes[name] = {"vote": default, "confidence": 0.3, "reasoning": "Fallback", "is_final": False}

            await asyncio.sleep(0.3)

        except Exception as e:
            log.error("Initial vote error %s: %s", name, e)
            default = "Petitioner" if info["conservative_lean"] > 0.5 else "Respondent"
            round_votes[name] = {"vote": default, "confidence": 0.2, "reasoning": str(e), "is_final": False}

    _log_tally("Initial votes", round_votes)
    return {"votes": [round_votes], "round_num": 1}


async def deliberation_node(state: DeliberationState) -> dict:
    """Node: One round of deliberation with anti-consensus-drift mechanisms."""
    llm_name = state["llm_name"]
    llms = models.get_prediction_models()
    llm = llms[llm_name]
    analysis = state["case_analysis"]
    round_num = state["round_num"]
    prev_votes = state["votes"][-1]  # Latest round

    pet = sum(1 for v in prev_votes.values() if v["vote"] == "Petitioner")
    res = len(prev_votes) - pet
    tally = f"Petitioner {pet} – Respondent {res}"

    # Determine likely opinion author
    majority_side = "Petitioner" if pet > res else "Respondent"

    # Collect proposed changes, then cap them
    proposed_votes: dict[str, JusticeVote] = {}
    change_proposals: list[tuple[str, JusticeVote]] = []  # (name, new_vote)

    for name, info in SUPREME_COURT_JUSTICES.items():
        own = prev_votes[name]

        # Dissent preservation — lock high-confidence justices after round 1
        if own["is_final"]:
            proposed_votes[name] = {**own}
            continue

        if round_num >= 2 and own["confidence"] >= DISSENT_LOCK_THRESHOLD:
            log.info("      %s locked (confidence %.2f >= %.2f)",
                     name.split()[-1], own["confidence"], DISSENT_LOCK_THRESHOLD)
            proposed_votes[name] = {**own, "is_final": True}
            continue

        # Build colleagues' positions
        colleagues_lines = []
        for other_name, other_vote in prev_votes.items():
            if other_name == name:
                continue
            colleagues_lines.append(
                f"  Justice {other_name} ({other_vote['vote']}, "
                f"confidence {other_vote['confidence']:.1f}): {other_vote['reasoning']}"
            )

        # Opinion assignment note for Chief Justice
        opinion_note = ""
        if name == "John Roberts" and prev_votes[name]["vote"] == majority_side:
            opinion_note = (
                "As Chief Justice, you would assign the majority opinion. "
                "Consider whether a narrower holding might attract more justices."
            )

        try:
            prompt = DELIBERATION_ROUND_PROMPT.format(
                name=name,
                philosophy=info["philosophy"],
                round_num=round_num,
                own_vote=own["vote"],
                own_confidence=own["confidence"],
                own_reasoning=own["reasoning"],
                colleagues_positions="\n".join(colleagues_lines),
                tally=tally,
                opinion_author_note=opinion_note,
            )
            resp = await llm.ainvoke(prompt)
            vote = _parse_json(resp.content)

            if vote and "vote" in vote:
                actually_changed = own["vote"] != vote["vote"]

                new_confidence = vote.get("confidence", own["confidence"])

                # Confidence decay on vote change
                if actually_changed:
                    new_confidence = max(0.1, new_confidence - CONFIDENCE_DECAY_ON_FLIP)
                    log.info(
                        "      Round %d: %s wants to change %s → %s (conf %.2f → %.2f)",
                        round_num, name.split()[-1], own["vote"], vote["vote"],
                        own["confidence"], new_confidence,
                    )

                new_vote: JusticeVote = {
                    "vote": vote["vote"],
                    "confidence": new_confidence,
                    "reasoning": vote.get("reasoning", own["reasoning"]),
                    "is_final": vote.get("is_final", False),
                }

                if actually_changed:
                    change_proposals.append((name, new_vote))
                else:
                    proposed_votes[name] = new_vote
            else:
                proposed_votes[name] = {**own, "is_final": True}

            await asyncio.sleep(0.3)

        except Exception as e:
            log.error("Deliberation error %s round %d: %s", name, round_num, e)
            proposed_votes[name] = {**own, "is_final": True}

    # Vote-change cap — only allow MAX_FLIPS_PER_ROUND changes per round.
    # Prioritize changes by those with highest post-change confidence (most justified).
    change_proposals.sort(key=lambda x: x[1]["confidence"], reverse=True)
    accepted_changes = 0
    for name, new_vote in change_proposals:
        if accepted_changes < MAX_FLIPS_PER_ROUND:
            proposed_votes[name] = new_vote
            accepted_changes += 1
            log.info("      Round %d: ACCEPTED change for %s", round_num, name.split()[-1])
        else:
            # Reject the change — keep previous vote
            proposed_votes[name] = {**prev_votes[name], "is_final": False}
            log.info("      Round %d: REJECTED change for %s (cap reached)", round_num, name.split()[-1])

    _log_tally(f"Round {round_num}", proposed_votes)
    return {"votes": [proposed_votes], "round_num": round_num + 1}


# ===========================================================================
# Routing logic
# ===========================================================================

def should_continue_deliberating(state: DeliberationState) -> str:
    """Decide whether another deliberation round is needed."""
    round_num = state["round_num"]
    votes_history = state["votes"]

    if round_num > MAX_DELIBERATION_ROUNDS:
        log.info("      Max deliberation rounds reached")
        return "finalize"

    # Always do at least 2 deliberation rounds
    if round_num <= 2:
        return "continue"

    # Force extra round if result is close (5-4 or 6-3)
    latest = votes_history[-1]
    pet = sum(1 for v in latest.values() if v["vote"] == "Petitioner")
    margin = abs(pet - (len(latest) - pet))
    if margin <= 3 and round_num <= MAX_DELIBERATION_ROUNDS:
        log.info("      Close vote (%d margin) — continuing deliberation", margin)
        return "continue"

    # No vote changes in last round — consensus reached
    if len(votes_history) >= 2:
        prev = votes_history[-2]
        curr = votes_history[-1]
        changes = sum(1 for n in curr if curr[n]["vote"] != prev[n]["vote"])
        if changes == 0:
            log.info("      No changes in last round — deliberation converged")
            return "finalize"

    return "continue"


def finalize_node(state: DeliberationState) -> dict:
    """Terminal node — just passes through, results are in state."""
    return {}


# ===========================================================================
# Graph construction
# ===========================================================================

def build_deliberation_graph() -> StateGraph:
    """Build the LangGraph for justice deliberation."""
    graph = StateGraph(DeliberationState)

    graph.add_node("analyze_case", analyze_case_node)
    graph.add_node("initial_votes", initial_votes_node)
    graph.add_node("deliberate", deliberation_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "analyze_case")
    graph.add_edge("analyze_case", "initial_votes")
    graph.add_edge("initial_votes", "deliberate")

    graph.add_conditional_edges(
        "deliberate",
        should_continue_deliberating,
        {"continue": "deliberate", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# ===========================================================================
# Helper
# ===========================================================================

def _log_tally(label: str, votes: dict[str, JusticeVote]) -> None:
    pet = sum(1 for v in votes.values() if v["vote"] == "Petitioner")
    res = len(votes) - pet
    locked = sum(1 for v in votes.values() if v.get("is_final"))
    log.info("      %s: Petitioner %d – Respondent %d (locked: %d)", label, pet, res, locked)


# ===========================================================================
# Public API
# ===========================================================================

async def predict_case_agentic(
    docket: str, pdf_dir: Path, llm_name: str | None = None,
) -> list[dict]:
    """Run agentic deliberation for a single case."""
    cfg = config.load()
    replicates = cfg["prediction"]["replicates"]

    # Content-based opinion filtering to prevent data leakage
    pdf_files = [f for f in pdf_dir.glob("*.pdf") if not pdf.is_court_opinion(f)]
    if not pdf_files:
        log.warning("No PDFs found for %s", docket)
        return []

    all_docs: list[Document] = []
    for f in pdf_files:
        all_docs.extend(pdf.extract_chunks(f))
    if not all_docs:
        log.warning("No text extracted for %s", docket)
        return []

    case_context, transcript_context = _build_context(all_docs, docket)
    log.info("  %s: %d PDFs, %d chunks (mode=agentic_v2)", docket, len(pdf_files), len(all_docs))

    graph = build_deliberation_graph()
    llm_models = models.get_prediction_models()
    llm_names = [llm_name] if llm_name else list(llm_models.keys())

    results: list[dict] = []

    for ln in llm_names:
        for r in range(1, replicates + 1):
            log.info("    %s [rep %d] — agentic_v2 deliberation", ln, r)
            try:
                initial_state: DeliberationState = {
                    "docket": docket,
                    "case_analysis": {},
                    "case_context": case_context,
                    "transcript_context": transcript_context,
                    "all_docs": all_docs,
                    "votes": [],
                    "round_num": 0,
                    "llm_name": ln,
                }

                final_state = await graph.ainvoke(initial_state)

                # Extract results
                all_rounds = final_state["votes"]
                final_votes = all_rounds[-1]
                initial_votes = all_rounds[0] if all_rounds else final_votes

                pet = sum(1 for v in final_votes.values() if v["vote"] == "Petitioner")
                res = len(final_votes) - pet
                winner = "Petitioner" if pet > res else "Respondent"
                split = f"{max(pet, res)}-{min(pet, res)}"
                avg_conf = sum(v["confidence"] for v in final_votes.values()) / len(final_votes)

                # Track changes across rounds
                total_changes = 0
                for round_idx in range(1, len(all_rounds)):
                    prev_r = all_rounds[round_idx - 1]
                    curr_r = all_rounds[round_idx]
                    total_changes += sum(
                        1 for n in curr_r if curr_r[n]["vote"] != prev_r[n]["vote"]
                    )

                result = {
                    "docket": docket,
                    "llm_model": ln,
                    "mode": "agentic_v2",
                    "replicate": r,
                    "has_transcripts": bool(transcript_context),
                    "case_analysis": final_state["case_analysis"],
                    "justice_votes": final_votes,
                    "initial_votes": initial_votes,
                    "petitioner_votes": pet,
                    "respondent_votes": res,
                    "predicted_winner": winner,
                    "vote_split": split,
                    "average_confidence": round(avg_conf, 3),
                    "deliberation_rounds": len(all_rounds) - 1,
                    "total_vote_changes": total_changes,
                    "votes_per_round": [
                        {n: {"vote": v["vote"], "confidence": v["confidence"],
                             "is_final": v.get("is_final", False)}
                         for n, v in rd.items()}
                        for rd in all_rounds
                    ],
                }

                # Initial tally for comparison
                init_pet = sum(1 for v in initial_votes.values() if v["vote"] == "Petitioner")
                init_res = len(initial_votes) - init_pet
                result["initial_winner"] = "Petitioner" if init_pet > init_res else "Respondent"
                result["initial_vote_split"] = f"{max(init_pet, init_res)}-{min(init_pet, init_res)}"

                results.append(result)

                msg = f"    {ln} [rep {r}]: {winner} ({split})"
                if total_changes > 0:
                    msg += f" [{total_changes} vote changes over {len(all_rounds)-1} rounds, was {result['initial_vote_split']}]"
                else:
                    msg += f" [stable after {len(all_rounds)-1} deliberation rounds]"
                log.info(msg)

                await asyncio.sleep(2)

            except Exception as e:
                log.error("    %s [rep %d] agentic_v2 failed: %s", ln, r, e)

    return results


async def run(
    terms: list[int] | None = None,
    single_docket: str | None = None,
) -> None:
    """Run agentic_v2 predictions for all cases or a single docket."""
    from scotus_v2.deliberation import predict_case_agentic

    cfg = config.load()
    terms = terms or cfg["terms"]

    def save_results(docket: str, results: list[dict], mode: str = "agentic_v2") -> None:
        out = config.prediction_path(docket, mode)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info("  Saved: %s", out)

    if single_docket:
        term = int(single_docket.split("-")[0])
        pdf_dir = config.pdf_dir_for_docket(term, single_docket)
        if not pdf_dir.exists():
            log.error("PDF directory not found: %s", pdf_dir)
            return
        results = await predict_case_agentic(single_docket, pdf_dir)
        if results:
            save_results(single_docket, results, mode="agentic_v2")
        return

    for term in terms:
        term_dir = config.pdf_dir_for_term(term)
        if not term_dir.exists():
            log.warning("Term directory not found: %s", term_dir)
            continue

        folders = sorted(
            d for d in term_dir.iterdir() if d.is_dir() and d.name.endswith("_pdfs")
        )
        log.info("Term 20%d: %d cases (mode=agentic_v2)", term, len(folders))

        for i, folder in enumerate(folders, 1):
            docket = folder.name.replace("_pdfs", "")

            if config.prediction_path(docket, "agentic_v2").exists():
                log.info("  [%d/%d] %s — already predicted, skipping", i, len(folders), docket)
                continue

            log.info("  [%d/%d] %s", i, len(folders), docket)
            try:
                results = await predict_case_agentic(docket, folder)
                if results:
                    save_results(docket, results, mode="agentic_v2")
            except Exception as e:
                log.error("  Error processing %s: %s", docket, e)

            await asyncio.sleep(1)
