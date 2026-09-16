"""Agentic SCOTUS deliberation v3 — Per-Justice Opinion RAG with Profile Synthesis.

Two-phase approach:
  Phase 1 (justice_profile_node): LLM reads retrieved opinion excerpts and synthesizes
          a case-specific profile for each justice — replacing static personas entirely.
  Phase 2 (initial_votes_node): Each justice votes using their synthesized profile.

Graph: START -> analyze_case -> justice_profile_synthesis -> initial_votes -> deliberate -> [continue/finalize]
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

from scotus_v3 import config, models, retrieval, pdf
from scotus_v3.justice_rag import JusticeRAGManager

log = logging.getLogger(__name__)

MAX_DELIBERATION_ROUNDS = 3
CONFIDENCE_DECAY_ON_FLIP = 0.15
MAX_FLIPS_PER_ROUND = 2
DISSENT_LOCK_THRESHOLD = 0.85


# ===========================================================================
# Justice Personas (static fallback — used when RAG context is thin)
# ===========================================================================

SUPREME_COURT_JUSTICES = {
    "John Roberts": {
        "philosophy": "Judicial minimalism and institutional legitimacy. Seeks narrow grounds for decisions, values precedent heavily.",
        "key_factors": ["precedent", "institutional legitimacy", "narrow rulings", "federalism"],
        "conservative_lean": 0.65,
    },
    "Clarence Thomas": {
        "philosophy": "Strict originalist and textualist. Interprets Constitution according to original public meaning. Willing to overturn precedent.",
        "key_factors": ["original meaning", "textualism", "limited government", "states rights"],
        "conservative_lean": 0.90,
    },
    "Samuel Alito": {
        "philosophy": "Conservative textualist focused on practical consequences. Skeptical of broad interpretations, especially for criminal defendants.",
        "key_factors": ["textualism", "law enforcement", "practical consequences", "conservative outcomes"],
        "conservative_lean": 0.85,
    },
    "Sonia Sotomayor": {
        "philosophy": "Pragmatic liberal emphasizing real-world impacts on vulnerable populations. Values empathy, fairness, and access to justice.",
        "key_factors": ["practical impacts", "fairness", "vulnerable populations", "broad remedial powers"],
        "conservative_lean": 0.15,
    },
    "Elena Kagan": {
        "philosophy": "Pragmatic textualist who bridges wings. Values workable solutions and statutory coherence.",
        "key_factors": ["statutory purpose", "workability", "compromise", "administrative deference"],
        "conservative_lean": 0.35,
    },
    "Neil Gorsuch": {
        "philosophy": "Principled textualist and originalist. Strict on separation of powers. Sometimes reaches liberal outcomes through consistent methodology.",
        "key_factors": ["plain meaning", "separation of powers", "rule of lenity", "original meaning"],
        "conservative_lean": 0.75,
    },
    "Brett Kavanaugh": {
        "philosophy": "Institutionalist conservative emphasizing precedent, history, and tradition. Values practical considerations.",
        "key_factors": ["precedent", "history and tradition", "practical workability", "judicial restraint"],
        "conservative_lean": 0.70,
    },
    "Amy Coney Barrett": {
        "philosophy": "Originalist and textualist with meticulous analytical approach. Focuses on original public meaning but also considers precedent.",
        "key_factors": ["original public meaning", "textual analysis", "precedent", "methodological consistency"],
        "conservative_lean": 0.80,
    },
    "Ketanji Brown Jackson": {
        "philosophy": "Progressive pragmatist emphasizing procedural fairness and practical consequences. Background as public defender.",
        "key_factors": ["procedural fairness", "practical consequences", "criminal justice reform", "equal protection"],
        "conservative_lean": 0.20,
    },
}

JUSTICE_NAMES = list(SUPREME_COURT_JUSTICES.keys())


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
# Transcript-aware retrieval (same as v2)
# ===========================================================================

TRANSCRIPT_RE = _re.compile(r"(transcript|oralarg)", _re.I)


def _split_docs_by_type(docs: list[Document]) -> tuple[list[Document], list[Document]]:
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
    transcripts, other_docs = _split_docs_by_type(all_docs)

    if other_docs:
        case_docs = retrieval.hybrid_retrieve(
            other_docs, f"Supreme Court case legal arguments {docket}"
        )
    else:
        case_docs = retrieval.hybrid_retrieve(
            all_docs, f"Supreme Court case legal arguments {docket}"
        )
    case_context = "\n\n---DOCUMENT---\n\n".join(d.page_content for d in case_docs)

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
# State definition (extended for v3)
# ===========================================================================

class JusticeVote(TypedDict):
    vote: str
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
    # v3 additions
    predict_term: int
    justice_contexts: dict[str, str]  # per-justice RAG results


# ===========================================================================
# Prompts — Phase 1: Profile Synthesis (the key v3 innovation)
# ===========================================================================

PROFILE_SYNTHESIS_PROMPT = """You are an expert Supreme Court analyst. Below are excerpts from opinions
that Justice {name} has authored in prior terms on legal issues related to the current case.

CURRENT CASE LEGAL QUESTION: {focal_point}
CURRENT CASE LEGAL PROVISION: {legal_provision}

EXCERPTS FROM JUSTICE {name_upper}'S PRIOR OPINIONS:
{opinion_excerpts}

Based ONLY on these actual prior opinions, extract this justice's analytical methodology
for this type of legal issue. Do NOT predict which side they would favor — focus entirely
on HOW they reason, what tests they apply, and what principles they prioritize.

Respond in this JSON format:
{{
  "methodology": "3-4 sentences describing how this justice actually reasons about this type of legal issue. Be specific — cite the doctrines, tests, and analytical frameworks from the excerpts.",
  "key_doctrines": ["list of 3-5 specific legal doctrines/tests/principles this justice applies, extracted from the excerpts"],
  "analytical_questions": ["list of 2-3 questions this justice would ask when analyzing a case like this, based on their past reasoning patterns"],
  "notable_quotes": "1-2 direct quotes from the excerpts that best capture this justice's analytical approach"
}}
"""

# ===========================================================================
# Prompts — Phase 2: Voting with Synthesized Profile
# ===========================================================================

INITIAL_POSITION_PROMPT = """You are Supreme Court Justice {name}.

YOUR JUDICIAL METHODOLOGY (extracted from your actual prior opinions on similar legal issues):
{synthesized_profile}

You are preparing for conference on this case.

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}
PETITIONER'S POSITION: {petitioner_position}
RESPONDENT'S POSITION: {respondent_position}

{transcript_section}

Apply YOUR methodology and analytical framework (described above) to THIS specific case.
Consider: which side's arguments better align with the doctrines and tests you have
consistently applied in your prior opinions?

Before casting your vote, identify the most relevant precedent you want to examine.

Respond in this JSON format:
{{
  "research_query": "The specific legal concept or precedent you want to look up",
  "initial_leaning": "Petitioner" or "Respondent",
  "leaning_reason": "One sentence explaining your initial instinct based on applying your methodology to this case's specific facts"
}}
"""

VOTE_WITH_RESEARCH_PROMPT = """You are Supreme Court Justice {name}.

YOUR JUDICIAL METHODOLOGY (from your actual prior opinions):
{synthesized_profile}

CASE ANALYSIS:
{case_analysis}

LEGAL QUESTION: {focal_point}

RESEARCH RESULTS for your query "{research_query}":
{research_results}

{transcript_section}

Apply your methodology and the doctrines you have consistently used in prior opinions
to the SPECIFIC FACTS of this case. Your vote should follow from applying your
established analytical framework to this case's particular legal questions.

Respond in this EXACT JSON format:
{{
  "vote": "Petitioner" or "Respondent",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your detailed legal reasoning in 3-4 sentences. Explain how your established doctrines and tests apply to THIS case's specific facts.",
  "key_precedent": "The most important precedent supporting your position"
}}
"""

DELIBERATION_ROUND_PROMPT = """You are Supreme Court Justice {name}.

YOUR BEHAVIORAL PROFILE (from your actual prior opinions):
{synthesized_profile}

This is deliberation round {round_num}. You are in conference with your fellow justices.

YOUR CURRENT POSITION: {own_vote} (confidence: {own_confidence})
YOUR REASONING: {own_reasoning}

COLLEAGUES' POSITIONS AND ARGUMENTS:
{colleagues_positions}

CURRENT TALLY: {tally}

{opinion_author_note}

DELIBERATION INSTRUCTIONS:
- DIRECTLY ADDRESS the strongest argument from the opposing side.
- Your behavioral profile reflects how you have ACTUALLY ruled in similar cases.
  Changing your vote means departing from your established pattern — justify this strongly.
- Change your vote ONLY if a colleague presents a legal argument that genuinely
  distinguishes this case from the precedents in your profile.
- A well-reasoned dissent is a vital part of Supreme Court jurisprudence.
  Do not abandon a sound position merely to join a majority.

Respond in this EXACT JSON format:
{{
  "vote": "Petitioner" or "Respondent",
  "confidence": 0.0 to 1.0,
  "reasoning": "Your reasoning after engaging with colleagues' arguments (3-4 sentences). Reference at least one colleague's argument by name.",
  "changed": true or false,
  "response_to": "Which colleague's argument most influenced you (or 'None')",
  "is_final": true or false
}}
"""


# ===========================================================================
# Tool: Precedent Research (same as v2)
# ===========================================================================

def _research_precedent(query: str, all_docs: list[Document]) -> str:
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
        transcript_section = "ORAL ARGUMENT EXCERPTS:\n" + transcript_ctx

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


async def justice_profile_node(state: DeliberationState) -> dict:
    """Node: Synthesize a case-specific behavioral profile for each justice.

    Two-phase approach:
    1. Retrieve relevant opinion excerpts from each justice's RAG
    2. Use LLM to synthesize a behavioral profile from those excerpts

    The synthesized profile REPLACES the static persona — this is the key
    difference from v3's first approach where excerpts were just passive context.
    """
    llm_name = state["llm_name"]
    llms = models.get_prediction_models()
    llm = llms[llm_name]
    analysis = state["case_analysis"]
    predict_term = state["predict_term"]

    rag = _rag_manager

    if rag is None:
        log.warning("No JusticeRAGManager available — using static personas only")
        # Fallback: use static personas as profiles
        profiles = {}
        for name, info in SUPREME_COURT_JUSTICES.items():
            profiles[name] = json.dumps({
                "methodology": info["philosophy"],
                "likely_side": "Petitioner" if info["conservative_lean"] > 0.5 else "Respondent",
                "side_probability": info["conservative_lean"],
                "key_doctrines": info["key_factors"],
                "dissent_triggers": "When the legal text clearly contradicts the majority position.",
                "notable_quotes": "",
            })
        return {"justice_contexts": profiles}

    focal = analysis.get("focal_point", "legal question")
    provision = analysis.get("legal_provision", "")
    issue_area = analysis.get("issue_area", "")
    precedents = analysis.get("key_precedents", [])

    # Multi-query: cast a wider net to find relevant opinions
    queries = [
        f"{focal} {provision}".strip(),
    ]
    if issue_area:
        queries.append(f"Supreme Court {issue_area} {provision}")
    if precedents:
        queries.append(f"{' '.join(precedents[:3])} legal doctrine")
    # Always add a broad doctrinal query
    queries.append(f"{focal} constitutional interpretation statutory analysis")

    cfg = config.load()
    rag_cfg = cfg.get("justice_rag", {})
    top_k = rag_cfg.get("top_k", 8)

    justice_contexts: dict[str, str] = {}

    for name in JUSTICE_NAMES:
        info = SUPREME_COURT_JUSTICES[name]

        # Phase 1: Multi-query retrieval of relevant opinion excerpts
        excerpts = rag.format_justice_context(
            justice_name=name,
            query=queries,  # Pass list for multi-query
            predict_term=predict_term,
            top_k=top_k,
        )

        if not excerpts or "No relevant" in excerpts:
            # Fallback to static persona
            log.info("      %s: no RAG results, using static persona", name.split()[-1])
            justice_contexts[name] = json.dumps({
                "methodology": info["philosophy"],
                "key_doctrines": info["key_factors"],
                "analytical_questions": [],
                "notable_quotes": "",
            })
            continue

        # Phase 2: LLM synthesizes a case-specific behavioral profile
        try:
            synthesis_prompt = PROFILE_SYNTHESIS_PROMPT.format(
                name=name,
                name_upper=name.upper(),
                focal_point=focal,
                legal_provision=provision,
                opinion_excerpts=excerpts[:6000],  # More generous limit for synthesis
            )
            resp = await llm.ainvoke(synthesis_prompt)
            profile = _parse_json(resp.content)

            if profile and "methodology" in profile:
                justice_contexts[name] = json.dumps(profile)
                log.info("      %s: synthesized profile (likely %s, p=%.2f)",
                         name.split()[-1],
                         profile.get("likely_side", "?"),
                         profile.get("side_probability", 0.5))
            else:
                # Synthesis failed, use excerpts directly
                justice_contexts[name] = json.dumps({
                    "methodology": info["philosophy"],
                    "likely_side": "Petitioner" if info["conservative_lean"] > 0.5 else "Respondent",
                    "side_probability": info["conservative_lean"],
                    "key_doctrines": info["key_factors"],
                    "dissent_triggers": "Unknown",
                    "notable_quotes": "",
                })

            await asyncio.sleep(0.3)

        except Exception as e:
            log.error("Profile synthesis failed for %s: %s", name, e)
            justice_contexts[name] = json.dumps({
                "methodology": info["philosophy"],
                "key_doctrines": info["key_factors"],
                "analytical_questions": [],
                "notable_quotes": "",
            })

    return {"justice_contexts": justice_contexts}


async def initial_votes_node(state: DeliberationState) -> dict:
    """Node: Each justice researches a precedent, then casts initial vote.

    Uses synthesized behavioral profiles instead of static personas.
    """
    llm_name = state["llm_name"]
    llms = models.get_prediction_models()
    llm = llms[llm_name]
    analysis = state["case_analysis"]
    all_docs = state["all_docs"]
    justice_contexts = state.get("justice_contexts", {})

    transcript_section = ""
    if state["transcript_context"]:
        transcript_section = "ORAL ARGUMENT SIGNALS:\n" + state["transcript_context"]

    round_votes: dict[str, JusticeVote] = {}

    for name, info in SUPREME_COURT_JUSTICES.items():
        # Parse synthesized profile
        profile_json = justice_contexts.get(name, "{}")
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError:
            profile = {"methodology": info["philosophy"], "key_doctrines": info["key_factors"]}

        # Format profile — methodology only, NO directional prediction
        doctrines = profile.get("key_doctrines", info["key_factors"])
        if isinstance(doctrines, list):
            doctrines_str = ", ".join(doctrines)
        else:
            doctrines_str = str(doctrines)

        profile_text = f"METHODOLOGY: {profile.get('methodology', info['philosophy'])}\n"
        profile_text += f"KEY DOCTRINES YOU APPLY: {doctrines_str}\n"

        questions = profile.get("analytical_questions", [])
        if questions:
            profile_text += "QUESTIONS YOU TYPICALLY ASK: " + "; ".join(questions) + "\n"

        notable = profile.get("notable_quotes", "")
        if notable:
            profile_text += f'FROM YOUR PRIOR OPINIONS: "{notable}"'

        try:
            # Step 1: Justice decides what to research
            research_prompt = INITIAL_POSITION_PROMPT.format(
                name=name,
                synthesized_profile=profile_text,
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

            # Step 2: Tool use — retrieve relevant passages from case docs
            research_results = _research_precedent(research_query, all_docs)

            # Step 3: Vote with methodology profile + research
            vote_prompt = VOTE_WITH_RESEARCH_PROMPT.format(
                name=name,
                synthesized_profile=profile_text,
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
                round_votes[name] = {"vote": default, "confidence": 0.3,
                                     "reasoning": "Fallback", "is_final": False}

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
    prev_votes = state["votes"][-1]
    justice_contexts = state.get("justice_contexts", {})

    pet = sum(1 for v in prev_votes.values() if v["vote"] == "Petitioner")
    res = len(prev_votes) - pet
    tally = f"Petitioner {pet} – Respondent {res}"
    majority_side = "Petitioner" if pet > res else "Respondent"

    proposed_votes: dict[str, JusticeVote] = {}
    change_proposals: list[tuple[str, JusticeVote]] = []

    for name, info in SUPREME_COURT_JUSTICES.items():
        own = prev_votes[name]

        if own["is_final"]:
            proposed_votes[name] = {**own}
            continue

        if round_num >= 2 and own["confidence"] >= DISSENT_LOCK_THRESHOLD:
            log.info("      %s locked (confidence %.2f >= %.2f)",
                     name.split()[-1], own["confidence"], DISSENT_LOCK_THRESHOLD)
            proposed_votes[name] = {**own, "is_final": True}
            continue

        colleagues_lines = []
        for other_name, other_vote in prev_votes.items():
            if other_name == name:
                continue
            colleagues_lines.append(
                f"  Justice {other_name} ({other_vote['vote']}, "
                f"confidence {other_vote['confidence']:.1f}): {other_vote['reasoning']}"
            )

        opinion_note = ""
        if name == "John Roberts" and prev_votes[name]["vote"] == majority_side:
            opinion_note = (
                "As Chief Justice, you would assign the majority opinion. "
                "Consider whether a narrower holding might attract more justices."
            )

        # Parse synthesized profile for deliberation context
        profile_json = justice_contexts.get(name, "{}")
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError:
            profile = {"methodology": info["philosophy"]}

        profile_text = (
            f"METHODOLOGY: {profile.get('methodology', info['philosophy'])}\n"
            f"KEY DOCTRINES: {', '.join(profile.get('key_doctrines', info['key_factors']))}"
        )

        try:
            prompt = DELIBERATION_ROUND_PROMPT.format(
                name=name,
                synthesized_profile=profile_text,
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

                if actually_changed:
                    new_confidence = max(0.1, new_confidence - CONFIDENCE_DECAY_ON_FLIP)
                    log.info(
                        "      Round %d: %s wants to change %s -> %s (conf %.2f -> %.2f)",
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

    # Vote-change cap
    change_proposals.sort(key=lambda x: x[1]["confidence"], reverse=True)
    accepted_changes = 0
    for name, new_vote in change_proposals:
        if accepted_changes < MAX_FLIPS_PER_ROUND:
            proposed_votes[name] = new_vote
            accepted_changes += 1
            log.info("      Round %d: ACCEPTED change for %s", round_num, name.split()[-1])
        else:
            proposed_votes[name] = {**prev_votes[name], "is_final": False}
            log.info("      Round %d: REJECTED change for %s (cap reached)", round_num, name.split()[-1])

    _log_tally(f"Round {round_num}", proposed_votes)
    return {"votes": [proposed_votes], "round_num": round_num + 1}


# ===========================================================================
# Routing logic (same as v2)
# ===========================================================================

def should_continue_deliberating(state: DeliberationState) -> str:
    round_num = state["round_num"]
    votes_history = state["votes"]

    if round_num > MAX_DELIBERATION_ROUNDS:
        log.info("      Max deliberation rounds reached")
        return "finalize"

    if round_num <= 2:
        return "continue"

    latest = votes_history[-1]
    pet = sum(1 for v in latest.values() if v["vote"] == "Petitioner")
    margin = abs(pet - (len(latest) - pet))
    if margin <= 3 and round_num <= MAX_DELIBERATION_ROUNDS:
        log.info("      Close vote (%d margin) — continuing deliberation", margin)
        return "continue"

    if len(votes_history) >= 2:
        prev = votes_history[-2]
        curr = votes_history[-1]
        changes = sum(1 for n in curr if curr[n]["vote"] != prev[n]["vote"])
        if changes == 0:
            log.info("      No changes in last round — deliberation converged")
            return "finalize"

    return "continue"


def finalize_node(state: DeliberationState) -> dict:
    return {}


# ===========================================================================
# Graph construction (v3 — includes justice_research node)
# ===========================================================================

def build_deliberation_graph() -> StateGraph:
    """Build the LangGraph for v3 justice deliberation with opinion RAG."""
    graph = StateGraph(DeliberationState)

    graph.add_node("analyze_case", analyze_case_node)
    graph.add_node("justice_profile_synthesis", justice_profile_node)
    graph.add_node("initial_votes", initial_votes_node)
    graph.add_node("deliberate", deliberation_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "analyze_case")
    graph.add_edge("analyze_case", "justice_profile_synthesis")
    graph.add_edge("justice_profile_synthesis", "initial_votes")
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
# Module-level RAG manager (set before graph invocation)
# ===========================================================================

_rag_manager: JusticeRAGManager | None = None


def set_rag_manager(manager: JusticeRAGManager) -> None:
    """Set the module-level RAG manager for use by graph nodes."""
    global _rag_manager
    _rag_manager = manager


# ===========================================================================
# Public API
# ===========================================================================

async def predict_case_agentic(
    docket: str,
    pdf_dir: Path,
    llm_name: str | None = None,
    rag_manager: JusticeRAGManager | None = None,
    predict_term: int | None = None,
) -> list[dict]:
    """Run agentic v3 deliberation for a single case.

    Args:
        docket: Case docket number.
        pdf_dir: Directory containing case PDFs (briefs, petitions, etc.).
        llm_name: Specific LLM to use (or None for all configured models).
        rag_manager: Pre-loaded JusticeRAGManager with per-justice indices.
        predict_term: The term being predicted (for temporal filtering).
    """
    cfg = config.load()
    replicates = cfg["prediction"]["replicates"]

    if predict_term is None:
        predict_term = config.parse_term_from_docket(docket)

    # Set module-level RAG manager for graph nodes
    if rag_manager:
        set_rag_manager(rag_manager)

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
    log.info("  %s: %d PDFs, %d chunks (mode=agentic_v3, term=%d)",
             docket, len(pdf_files), len(all_docs), predict_term)

    graph = build_deliberation_graph()
    llm_models = models.get_prediction_models()
    llm_names = [llm_name] if llm_name else list(llm_models.keys())

    results: list[dict] = []

    for ln in llm_names:
        for r in range(1, replicates + 1):
            log.info("    %s [rep %d] — agentic_v3 deliberation", ln, r)
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
                    "predict_term": predict_term,
                    "justice_contexts": {},
                }

                final_state = await graph.ainvoke(initial_state)

                all_rounds = final_state["votes"]
                final_votes = all_rounds[-1]
                initial_votes = all_rounds[0] if all_rounds else final_votes

                pet = sum(1 for v in final_votes.values() if v["vote"] == "Petitioner")
                res = len(final_votes) - pet
                winner = "Petitioner" if pet > res else "Respondent"
                split = f"{max(pet, res)}-{min(pet, res)}"
                avg_conf = sum(v["confidence"] for v in final_votes.values()) / len(final_votes)

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
                    "mode": "agentic_v3",
                    "replicate": r,
                    "has_transcripts": bool(transcript_context),
                    "has_opinion_rag": rag_manager is not None,
                    "predict_term": predict_term,
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
                log.error("    %s [rep %d] agentic_v3 failed: %s", ln, r, e)

    return results


async def run(
    terms: list[int] | None = None,
    single_docket: str | None = None,
    llm_name: str | None = None,
) -> None:
    """Run agentic_v3 predictions for all cases or a single docket."""
    cfg = config.load()
    terms = terms or cfg["terms"]

    # Load per-justice RAG indices
    log.info("Loading per-justice opinion RAG indices...")
    rag_manager = JusticeRAGManager()
    rag_manager.load_all()

    if not rag_manager.stores:
        log.warning("No justice RAG indices found! Run opinion scraper + index builder first.")
        log.warning("Proceeding without opinion RAG (fallback to static personas only).")

    for name in rag_manager.available_justices():
        log.info("  %s", rag_manager.get_justice_summary(name, predict_term=99))

    def save_results(docket: str, results: list[dict], mode: str = "agentic_v3") -> None:
        out = config.prediction_path(docket, mode)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info("  Saved: %s", out)

    if single_docket:
        term = config.parse_term_from_docket(single_docket)
        pdf_dir = config.pdf_dir_for_docket(term, single_docket)
        if not pdf_dir.exists():
            log.error("PDF directory not found: %s", pdf_dir)
            return
        results = await predict_case_agentic(
            single_docket, pdf_dir,
            llm_name=llm_name,
            rag_manager=rag_manager,
            predict_term=term,
        )
        if results:
            save_results(single_docket, results)
        return

    for term in terms:
        term_dir = config.pdf_dir_for_term(term)
        if not term_dir.exists():
            log.warning("Term directory not found: %s", term_dir)
            continue

        folders = sorted(
            d for d in term_dir.iterdir() if d.is_dir() and d.name.endswith("_pdfs")
        )
        log.info("Term 20%d: %d cases (mode=agentic_v3)", term, len(folders))

        for i, folder in enumerate(folders, 1):
            docket = folder.name.replace("_pdfs", "")

            if config.prediction_path(docket, "agentic_v3").exists():
                log.info("  [%d/%d] %s — already predicted, skipping", i, len(folders), docket)
                continue

            log.info("  [%d/%d] %s", i, len(folders), docket)
            try:
                results = await predict_case_agentic(
                    docket, folder,
                    llm_name=llm_name,
                    rag_manager=rag_manager,
                    predict_term=term,
                )
                if results:
                    save_results(docket, results)
            except Exception as e:
                log.error("  Error processing %s: %s", docket, e)

            await asyncio.sleep(1)
