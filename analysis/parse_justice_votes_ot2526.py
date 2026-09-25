"""Parse per-justice OT2025-26 votes from the SCOTUSblog-derived results sheet.

OT2025-26 is not in the SCDB yet, so justice-level scoring has no authoritative
source. The results spreadsheet does carry the reporter's attribution line
("Barrett, J., delivered the opinion of the Court, in which ... joined."), which
identifies the majority and any dissenters. This module parses that line into a
per-justice majority/dissent map and validates the parse against the independent
`Vote split` column; cases whose parse disagrees with the stated split are
dropped rather than guessed at.

A justice's vote for a *party* is then majority XOR petitioner-won, the same
convention score_predictions.py applies to the SCDB.

Import `load_justice_truth()` from the analysis scripts; run directly to see
parse coverage and the cases that failed validation.
"""
from __future__ import annotations

import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import DATA_DIR  # noqa: E402
XLSX = os.path.join(DATA_DIR, "scotus_ot2025_cases.xlsx")

JUSTICES = [
    "Roberts", "Thomas", "Alito", "Sotomayor", "Kagan",
    "Gorsuch", "Kavanaugh", "Barrett", "Jackson",
]

# prediction persona name -> surname used in the attribution line
PERSONA_TO_SURNAME = {
    "John Roberts": "Roberts", "Clarence Thomas": "Thomas", "Samuel Alito": "Alito",
    "Sonia Sotomayor": "Sotomayor", "Elena Kagan": "Kagan", "Neil Gorsuch": "Gorsuch",
    "Brett Kavanaugh": "Kavanaugh", "Amy Coney Barrett": "Barrett",
    "Ketanji Brown Jackson": "Jackson",
}


def _names_in(text: str) -> list[str]:
    """Surnames of justices mentioned in a fragment ("The Chief Justice" -> Roberts)."""
    names = [j for j in JUSTICES if re.search(rf"\b{j}\b", text)]
    if "Roberts" not in names and re.search(r"\bChief Justice\b", text):
        names.append("Roberts")
    return names


# Clause starts in the two styles the sheet uses. Reporter style:
#   "Barrett, J., delivered ...", "Roberts, C. J., announced ...",
#   "Alito, J., and Gorsuch, J., filed ...", "Alito, J., took no part ..."
_REPORTER_MARK = (r"\b[A-Z][a-z]+,\s*(?:C\.\s*)?J\.,?\s*"
                  r"(?:and\s+[A-Z][a-z]+,\s*JJ?\.,?\s*)?"
                  r"(?:filed|delivered|announced|took\s+no\s+part)")
# Per curiam style, one sentence per writing:
#   "Opinion per curiam. Justice Alito, with whom ..., dissenting."
#   "Justice Jackson would deny the petition for a writ of certiorari."
_SENTENCE_MARK = (r"(?:^|(?<=\.\s))(?:Opinion\s+per\s+curiam"
                  r"|(?:The\s+Chief\s+Justice|Justice\s+[A-Z][a-z]+)(?=,|\s+would\b))")


def _clauses(text: str) -> list[str]:
    """Split the attribution line into one clause per opinion or separate statement."""
    marks = sorted({0} | {m.start() for m in re.finditer(f"{_REPORTER_MARK}|{_SENTENCE_MARK}", text)})
    marks.append(len(text))
    return [text[marks[i]:marks[i + 1]] for i in range(len(marks) - 1)]


def parse_attribution(text: str) -> dict[str, str] | None:
    """Map surname -> 'majority' | 'dissent' | 'recused' for one attribution line.

    Every justice named in a clause takes that clause's role. Coding rules
    (agreed with the authors for OT2025-26):
      - "concurring in the judgment" (whole judgment) agrees on the outcome, even
        with "and dissenting in part" appended (Kavanaugh in Trump v. Barbara);
      - "concurring in part and dissenting in part", "concurring in the judgment
        in part and dissenting in part", and plain dissents count as dissent;
      - "would deny the petition for a writ of certiorari" counts as dissent.
    Returns None when the line carries no majority or per curiam clause.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    role: dict[str, str] = {}
    saw_majority = False

    for cl in _clauses(text):
        low = cl.lower()
        names = _names_in(cl)
        if "took no part" in low:
            for j in names:
                role[j] = "recused"
            continue
        if re.search(r"delivered the opinion|announced the judgment|per curiam", low):
            saw_majority = True
            for j in names:
                role.setdefault(j, "majority")
        elif "would deny" in low:
            for j in names:
                role[j] = "dissent"
        elif re.search(r"concurring in the judgment(?! in part)", low):
            for j in names:
                role.setdefault(j, "majority")
        elif "dissent" in low:
            for j in names:
                role[j] = "dissent"
        else:
            # plain concurrence -> agrees on outcome
            for j in names:
                role.setdefault(j, "majority")

    if not saw_majority:
        return None
    # Anyone unmentioned joined the majority silently.
    for j in JUSTICES:
        role.setdefault(j, "majority")
    return role


_ROLE_RANK = {"join": 0, "concur": 1, "concur_judgment": 2, "dissent": 3}


def parse_writings(text: str) -> dict | None:
    """Who wrote what, for authorship and separate-writing scoring.

    Returns {"author": surname or None (per curiam), "roles": {surname: role},
    "wrote": {surname: bool}} with role in author | join | concur |
    concur_judgment | dissent | recused, using the same clause rules as
    parse_attribution(). "wrote" marks authoring a separate opinion (a
    concurrence or dissent, not the opinion of the Court); a bare "would deny
    the petition" statement is a dissenting vote but not a writing.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    author, roles, wrote = None, {}, {j: False for j in JUSTICES}
    saw_majority = False

    def assign(j, role):
        if roles.get(j) in ("author", "recused"):
            return
        if _ROLE_RANK[role] >= _ROLE_RANK.get(roles.get(j), -1):
            roles[j] = role

    for cl in _clauses(text):
        low = cl.lower()
        names = _names_in(cl)
        sentence_style = re.match(r"\W*(?:The\s+Chief\s+Justice|Justice\s)", cl) is not None
        if sentence_style:
            writers = [] if "would deny" in low else _names_in(cl.split(",", 1)[0])
        else:
            head = re.split(r"\b(?:filed|delivered|announced|took\s+no\s+part)\b", cl, 1)[0]
            writers = _names_in(head)

        if "took no part" in low:
            for j in names:
                roles[j] = "recused"
            continue
        if re.search(r"delivered the opinion|announced the judgment|per curiam", low):
            saw_majority = True
            if "per curiam" not in low and writers:
                author = writers[0]
                roles[author] = "author"
            for j in names:
                assign(j, "join")
            continue
        if "would deny" in low or ("dissent" in low and not re.search(r"concurring in the judgment(?! in part)", low)):
            role = "dissent"
        elif re.search(r"concurring in the judgment(?! in part)", low):
            role = "concur_judgment"
        else:
            role = "concur"
        for j in names:
            assign(j, role)
        for j in writers:
            wrote[j] = True

    if not saw_majority:
        return None
    for j in JUSTICES:
        roles.setdefault(j, "join")
    return {"author": author, "roles": roles, "wrote": wrote}


def load_writings_truth() -> dict[str, dict]:
    """docket -> parse_writings() result, for OT2025-26 cases whose dissent count
    matches the sheet's `Vote split` (the same validation as load_justice_truth)."""
    x = pd.read_excel(XLSX)
    out = {}
    for _, r in x.iterrows():
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(r["Vote split"]))
        w = parse_writings(r["Justices split by name"])
        if w is None or m is None:
            continue
        if sum(v == "dissent" for v in w["roles"].values()) == int(m.group(2)):
            out[str(r["Docket number"]).strip()] = w
    return out


def load_justice_truth(verbose: bool = False):
    """Return (truth, dropped) where truth maps docket -> {surname: 'Petitioner'|'Respondent'}.

    Only cases whose parsed majority size matches the sheet's `Vote split` are kept.
    """
    x = pd.read_excel(XLSX)
    truth, dropped = {}, []

    for _, r in x.iterrows():
        docket = str(r["Docket number"]).strip()
        winner = r["Winner (petitioner/respondent)"]
        split = r["Vote split"]
        if not isinstance(winner, str) or not isinstance(split, str):
            continue
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", split)
        role = parse_attribution(r["Justices split by name"])
        if role is None or m is None:
            dropped.append((docket, "unparseable", split))
            continue

        maj_n, min_n = int(m.group(1)), int(m.group(2))
        got_maj = sum(v == "majority" for v in role.values())
        got_min = sum(v == "dissent" for v in role.values())

        if (got_maj, got_min) != (maj_n, min_n):
            dropped.append((docket, f"parsed {got_maj}-{got_min} != stated", split))
            continue

        # A recused justice has no vote and is left out of the case's map.
        pet_won = winner.strip() == "Petitioner"
        truth[docket] = {
            j: ("Petitioner" if (role[j] == "majority") == pet_won else "Respondent")
            for j in JUSTICES if role[j] != "recused"
        }

    if verbose:
        print(f"justice-level truth parsed for {len(truth)} cases; {len(dropped)} dropped")
        for d, why, split in dropped:
            print(f"   drop {d:9s} {why:28s} (sheet split: {split})")
    return truth, dropped


if __name__ == "__main__":
    truth, _ = load_justice_truth(verbose=True)
    lean = {j: 0 for j in JUSTICES}
    seated = {j: 0 for j in JUSTICES}
    for votes in truth.values():
        for j, v in votes.items():
            lean[j] += v == "Petitioner"
            seated[j] += 1
    print(f"\nPetitioner-vote rate by justice (n={len(truth)} cases):")
    for j in JUSTICES:
        print(f"   {j:11s} {100*lean[j]/seated[j]:5.1f}%")
