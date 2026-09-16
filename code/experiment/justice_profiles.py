"""Enhanced justice knowledge base for RAG augmentation.

Builds rich, retrievable documents about each Supreme Court justice from:
1. Voting history statistics (computed from ground truth CSVs, prior terms only)
2. Detailed biographical and philosophical profiles (curated text)

These documents are used across ALL experimental conditions equally,
so they do not confound the contamination measurement.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scotus_v2 import config
from scotus_v2.evaluate import JUSTICE_NAME_MAP, ABBREV_TO_NAME, load_ground_truth

log = logging.getLogger(__name__)

PROFILES_DIR = config.data_dir() / "justice_profiles"

# Detailed justice profiles — biographical, methodological, and strategic.
# These expand the short persona strings in deliberation.py with retrievable depth.

JUSTICE_PROFILES: dict[str, list[str]] = {
    "John Roberts": [
        (
            "Chief Justice John G. Roberts Jr. — Biographical Background\n\n"
            "Born January 27, 1955, in Buffalo, New York. A.B. from Harvard College (1976), "
            "J.D. from Harvard Law School (1979). Clerked for Judge Henry Friendly on the "
            "Second Circuit and then for Justice William Rehnquist. Served in the Reagan "
            "administration as Special Assistant to the Attorney General and Associate Counsel "
            "to the President. Practiced at Hogan & Hartson (now Hogan Lovells), argued 39 "
            "cases before the Supreme Court. Nominated to the D.C. Circuit by President "
            "George W. Bush in 2003, then elevated to Chief Justice in 2005 after the death "
            "of Chief Justice Rehnquist."
        ),
        (
            "Chief Justice Roberts — Judicial Methodology\n\n"
            "Roberts is a judicial minimalist who favors narrow, case-specific holdings over "
            "broad pronouncements. He has articulated a vision of the Chief Justice as an "
            "institutionalist whose primary concern is the legitimacy and public perception "
            "of the Court. He frequently seeks to build consensus, sometimes joining opinions "
            "he might not fully agree with to avoid fragmented decisions. His approach to "
            "statutory interpretation is primarily textualist, though he is pragmatic about "
            "considering purpose when text is ambiguous. On constitutional questions, he "
            "tends toward incrementalism — moving the law in a conservative direction "
            "through a series of small steps rather than dramatic shifts."
        ),
        (
            "Chief Justice Roberts — Notable Opinions and Voting Patterns\n\n"
            "Key majority opinions: National Federation of Independent Business v. Sebelius "
            "(2012, upholding ACA individual mandate as a tax), Shelby County v. Holder "
            "(2013, striking preclearance formula of VRA), Department of Commerce v. New York "
            "(2019, blocking citizenship question on census), Trump v. Hawaii (2018, upholding "
            "travel ban). Roberts frequently forms a bloc with Kavanaugh and Barrett on close "
            "cases. He has crossed ideological lines on major cases where he perceives "
            "institutional legitimacy at stake. His role as Chief Justice gives him the power "
            "to assign opinions when he is in the majority, which he uses strategically to "
            "control the scope of holdings."
        ),
    ],
    "Clarence Thomas": [
        (
            "Justice Clarence Thomas — Biographical Background\n\n"
            "Born June 23, 1948, in Pin Point, Georgia. A.B. from College of the Holy Cross "
            "(1971), J.D. from Yale Law School (1974). Served as an Assistant Attorney General "
            "in Missouri, then at Monsanto Company. Appointed Chairman of the EEOC by President "
            "Reagan in 1982. Nominated to the D.C. Circuit by President George H.W. Bush in "
            "1990, then to the Supreme Court in 1991. Confirmed after contentious hearings "
            "by a 52-48 vote."
        ),
        (
            "Justice Thomas — Judicial Methodology\n\n"
            "Thomas is the Court's most consistent originalist. He interprets the Constitution "
            "according to its original public meaning at the time of ratification, and is "
            "willing to overturn longstanding precedent when he believes it conflicts with "
            "original meaning. He regularly writes solo concurrences calling for reconsideration "
            "of established doctrines. On statutory interpretation, he is a strict textualist "
            "who reads statutes by their plain meaning. He has a distinctive approach to "
            "the Commerce Clause, arguing it should be limited to actual interstate commerce. "
            "He is skeptical of substantive due process and the incorporation doctrine."
        ),
        (
            "Justice Thomas — Notable Opinions and Alliances\n\n"
            "Key opinions: Dobbs v. Jackson Women's Health Organization (2022, concurrence "
            "calling for reconsideration of Griswold, Lawrence, Obergefell), New York State "
            "Rifle & Pistol Association v. Bruen (2022, expanding Second Amendment rights), "
            "Gonzales v. Raich (2005, dissent arguing Commerce Clause limits). Thomas "
            "frequently aligns with Alito and Gorsuch, forming the Court's most conservative "
            "bloc. He is the longest-serving current justice."
        ),
    ],
    "Samuel Alito": [
        (
            "Justice Samuel A. Alito Jr. — Biographical Background\n\n"
            "Born April 1, 1950, in Trenton, New Jersey. A.B. from Princeton University "
            "(1972), J.D. from Yale Law School (1975). Served as Assistant U.S. Attorney "
            "for the District of New Jersey, then in the Office of Legal Counsel and as "
            "U.S. Attorney for New Jersey. Appointed to the Third Circuit by President "
            "George H.W. Bush in 1990, nominated to the Supreme Court by President "
            "George W. Bush in 2005."
        ),
        (
            "Justice Alito — Judicial Methodology and Key Opinions\n\n"
            "Alito is a conservative textualist with a prosecutorial background that strongly "
            "influences his criminal procedure jurisprudence. He almost invariably sides with "
            "law enforcement and is skeptical of expanding defendant protections. On religious "
            "liberty, he is the Court's strongest advocate for broad Free Exercise protections, "
            "authoring Burwell v. Hobby Lobby (2014). He authored the majority in Dobbs v. "
            "Jackson Women's Health Organization (2022), overturning Roe v. Wade. He is "
            "skeptical of administrative agency power and regulatory overreach. Common "
            "alliances with Thomas and Gorsuch."
        ),
    ],
    "Sonia Sotomayor": [
        (
            "Justice Sonia Sotomayor — Biographical Background\n\n"
            "Born June 25, 1954, in the Bronx, New York. A.B. from Princeton University "
            "(1976), J.D. from Yale Law School (1979). Served as Assistant District Attorney "
            "in New York County, then in private practice. Appointed to the District Court "
            "by President George H.W. Bush in 1992, elevated to the Second Circuit by "
            "President Clinton in 1998. Nominated to the Supreme Court by President Obama "
            "in 2009. First Hispanic justice."
        ),
        (
            "Justice Sotomayor — Judicial Methodology and Key Opinions\n\n"
            "Sotomayor is the Court's most liberal justice on criminal procedure, reflecting "
            "her background as a prosecutor who saw the system's impact on communities. She "
            "writes frequent dissents in Fourth Amendment and qualified immunity cases, "
            "advocating for broader protections. She is a purposivist in statutory interpretation, "
            "emphasizing legislative intent and real-world consequences. Notable dissents in "
            "Dobbs, Students for Fair Admissions v. Harvard, and Trump v. Hawaii. She commonly "
            "aligns with Kagan and Jackson."
        ),
    ],
    "Elena Kagan": [
        (
            "Justice Elena Kagan — Biographical Background\n\n"
            "Born April 28, 1960, in New York City. A.B. from Princeton (1981), M.Phil. from "
            "Oxford (1983), J.D. from Harvard Law School (1986). Clerked for Judge Abner Mikva "
            "and Justice Thurgood Marshall. Professor at University of Chicago Law School and "
            "Harvard Law School, where she became the first female Dean. Served as Solicitor "
            "General under President Obama before nomination to the Supreme Court in 2010."
        ),
        (
            "Justice Kagan — Judicial Methodology and Key Opinions\n\n"
            "Kagan is widely regarded as the Court's most skilled writer and pragmatic bridge-builder. "
            "She is a textualist by methodology but frequently reaches liberal outcomes through "
            "careful textual analysis. She is particularly influential in statutory interpretation "
            "cases, where her academic precision often wins over swing justices. She authored "
            "key opinions on administrative law and is a strong defender of Chevron deference. "
            "She bridges ideological lines, occasionally aligning with Barrett and Gorsuch on "
            "statutory cases. Her dissents are known for their clarity and rhetorical force."
        ),
    ],
    "Neil Gorsuch": [
        (
            "Justice Neil M. Gorsuch — Biographical Background\n\n"
            "Born August 29, 1967, in Denver, Colorado. A.B. from Columbia University (1988), "
            "J.D. from Harvard Law School (1991), D.Phil. from Oxford (2004). Clerked for "
            "Judges David Sentelle and Anthony Kennedy. Practiced at Kellogg Huber, then served "
            "in the DOJ. Appointed to the Tenth Circuit by President George W. Bush in 2006, "
            "nominated to the Supreme Court by President Trump in 2017."
        ),
        (
            "Justice Gorsuch — Judicial Methodology and Key Opinions\n\n"
            "Gorsuch is a principled textualist and originalist who sometimes reaches liberal "
            "outcomes through consistent methodology. He authored Bostock v. Clayton County "
            "(2020), ruling that Title VII prohibits discrimination based on sexual orientation "
            "and gender identity — a textualist analysis that produced a liberal result. He "
            "consistently applies the rule of lenity in criminal cases, often voting for "
            "defendants. He led the charge against Chevron deference in Loper Bright (2024). "
            "He is strong on separation of powers and Fourth Amendment protections for digital "
            "privacy. He sometimes aligns with Sotomayor on criminal defendant rights."
        ),
    ],
    "Brett Kavanaugh": [
        (
            "Justice Brett M. Kavanaugh — Biographical Background\n\n"
            "Born February 12, 1965, in Washington, D.C. A.B. from Yale University (1987), "
            "J.D. from Yale Law School (1990). Clerked for Judges Walter Stapleton and Alex "
            "Kozinski, then for Justice Anthony Kennedy. Served in the Starr investigation, "
            "then as Staff Secretary in the George W. Bush White House. Appointed to the D.C. "
            "Circuit in 2006, nominated to the Supreme Court by President Trump in 2018."
        ),
        (
            "Justice Kavanaugh — Judicial Methodology and Key Opinions\n\n"
            "Kavanaugh is an institutionalist conservative who values precedent, history, and "
            "practical workability. He frequently aligns with Roberts, forming the center-right "
            "of the Court. He co-authored the framework for the major questions doctrine and "
            "is skeptical of broad agency power. In criminal cases, he generally sides with "
            "the government but follows clear precedent. He is a strong advocate for religious "
            "liberty and free speech. His opinions tend to be measured and narrow, avoiding "
            "sweeping constitutional pronouncements."
        ),
    ],
    "Amy Coney Barrett": [
        (
            "Justice Amy Coney Barrett — Biographical Background\n\n"
            "Born January 28, 1972, in New Orleans, Louisiana. B.A. from Rhodes College (1994), "
            "J.D. from Notre Dame Law School (1997, first in class). Clerked for Judge Laurence "
            "Silberman and Justice Antonin Scalia. Professor at Notre Dame Law School for 15 "
            "years. Appointed to the Seventh Circuit by President Trump in 2017, nominated to "
            "the Supreme Court in 2020."
        ),
        (
            "Justice Barrett — Judicial Methodology and Key Opinions\n\n"
            "Barrett is an originalist and textualist with meticulous academic rigor. She is "
            "methodologically close to Scalia but more measured in tone. She has shown "
            "willingness to break from the conservative bloc on close textual questions, "
            "particularly aligning with Kagan on statutory interpretation. She authored "
            "significant opinions on standing and First Amendment questions. She is cautious "
            "about overly broad holdings and values methodological consistency over reaching "
            "preferred outcomes. She frequently forms a bloc with Roberts and Kavanaugh."
        ),
    ],
    "Ketanji Brown Jackson": [
        (
            "Justice Ketanji Brown Jackson — Biographical Background\n\n"
            "Born September 14, 1970, in Washington, D.C. A.B. from Harvard University (1992), "
            "J.D. from Harvard Law School (1996). Clerked for Judge Patti Saris and Justice "
            "Stephen Breyer. Served as Assistant Special Counsel on the U.S. Sentencing "
            "Commission, Assistant Federal Public Defender in D.C., and in private practice. "
            "Vice Chair of the U.S. Sentencing Commission. Appointed to the District Court "
            "by President Obama in 2013, elevated to the D.C. Circuit by President Biden in "
            "2021, then to the Supreme Court in 2022. First Black female justice."
        ),
        (
            "Justice Jackson — Judicial Methodology and Key Opinions\n\n"
            "Jackson is a progressive pragmatist whose background as a public defender and "
            "sentencing commissioner deeply informs her criminal justice approach. She is "
            "strongly pro-defendant in criminal cases, skeptical of mandatory minimums, and "
            "critical of prosecutorial overreach. In statutory interpretation, she favors "
            "contextual analysis considering text, structure, and purpose. She is a strong "
            "advocate for race-conscious remedies and expansive equal protection. Her early "
            "tenure has shown a willingness to write lengthy separate opinions establishing "
            "her jurisprudential framework. She aligns most closely with Sotomayor and Kagan."
        ),
    ],
}


def _compute_voting_stats(
    gt: dict[str, dict], max_term: int
) -> dict[str, dict]:
    """Compute voting statistics from ground truth for terms <= max_term.

    Returns per-justice stats: vote counts, petitioner rate, agreement matrix.
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "petitioner": 0, "respondent": 0,
        "agreements": defaultdict(int), "disagreements": defaultdict(int),
    })

    # Group votes by case
    cases: dict[str, dict[str, str]] = {}
    for docket, case in gt.items():
        term = int(docket.split("-")[0])
        if term > max_term:
            continue
        cases[docket] = case["justice_votes"]

    for docket, votes in cases.items():
        for abbrev, vote in votes.items():
            name = ABBREV_TO_NAME.get(abbrev, abbrev)
            stats[name]["total"] += 1
            if vote == "Petitioner":
                stats[name]["petitioner"] += 1
            else:
                stats[name]["respondent"] += 1

            # Pairwise agreement
            for other_abbrev, other_vote in votes.items():
                if other_abbrev == abbrev:
                    continue
                other_name = ABBREV_TO_NAME.get(other_abbrev, other_abbrev)
                if vote == other_vote:
                    stats[name]["agreements"][other_name] += 1
                else:
                    stats[name]["disagreements"][other_name] += 1

    return dict(stats)


def _format_voting_stats_doc(name: str, stats: dict) -> str:
    """Format voting statistics as a readable document."""
    total = stats["total"]
    if total == 0:
        return ""

    pet_rate = stats["petitioner"] / total * 100
    lines = [
        f"Justice {name} — Voting Statistics (Historical)\n",
        f"Total cases decided: {total}",
        f"Voted for Petitioner: {stats['petitioner']} ({pet_rate:.1f}%)",
        f"Voted for Respondent: {stats['respondent']} ({100 - pet_rate:.1f}%)",
        "",
        "Agreement rates with other justices:",
    ]

    for other in sorted(stats["agreements"].keys()):
        agree = stats["agreements"][other]
        disagree = stats["disagreements"].get(other, 0)
        pair_total = agree + disagree
        if pair_total > 0:
            rate = agree / pair_total * 100
            lines.append(f"  {other}: {rate:.1f}% agreement ({agree}/{pair_total})")

    return "\n".join(lines)


def build_justice_documents(
    predict_term: int,
) -> dict[str, list[Document]]:
    """Build retrievable justice profile documents.

    Args:
        predict_term: The term being predicted. Voting stats only use
                      prior terms to prevent data leakage.

    Returns:
        Dict mapping justice name to list of Documents.
    """
    gt = load_ground_truth()

    # Compute voting stats from prior terms only
    max_term = predict_term - 1
    voting_stats = _compute_voting_stats(gt, max_term)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    justice_docs: dict[str, list[Document]] = {}

    for name, profiles in JUSTICE_PROFILES.items():
        docs = []

        # Add curated profile documents
        for profile_text in profiles:
            for chunk in splitter.split_text(profile_text):
                if len(chunk.strip()) > 50:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": f"justice_profile_{name.replace(' ', '_')}.txt",
                            "justice": name,
                            "doc_type": "profile",
                        },
                    ))

        # Add voting statistics document
        if name in voting_stats:
            stats_text = _format_voting_stats_doc(name, voting_stats[name])
            if stats_text:
                docs.append(Document(
                    page_content=stats_text,
                    metadata={
                        "source": f"justice_stats_{name.replace(' ', '_')}.txt",
                        "justice": name,
                        "doc_type": "voting_stats",
                        "max_term": max_term,
                    },
                ))

        justice_docs[name] = docs
        log.info("  %s: %d profile documents", name, len(docs))

    return justice_docs


def get_justice_context(
    justice_name: str,
    all_justice_docs: dict[str, list[Document]],
    case_analysis: dict | None = None,
) -> str:
    """Get relevant profile context for a specific justice.

    Returns formatted text ready to insert into prompts.
    """
    docs = all_justice_docs.get(justice_name, [])
    if not docs:
        return ""

    # For now, return all documents for the justice (they're small enough).
    # A future enhancement could use retrieval to select the most relevant
    # profile chunks based on the case analysis.
    texts = [doc.page_content for doc in docs]
    return "\n\n---\n\n".join(texts)
