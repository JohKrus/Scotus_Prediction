"""Scrape SCOTUS slip opinions from supremecourt.gov.

Downloads all opinion PDFs organized by term, with author attribution
from the slip opinions table. Supports checkpointing and rate limiting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from scotus_v3 import config

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (scotus-opinion-scraper/1.0)"}

SLIP_OPINIONS_URL = "https://www.supremecourt.gov/opinions/slipopinion/{term}"

# Justice initials as they appear in the "J." column of the slip opinions table.
# Mapping covers current justices and recent retirees.
JUSTICE_INITIALS: dict[str, str] = {
    # Current justices (as of term 25)
    "R": "John Roberts",
    "JR": "John Roberts",
    "JGR": "John Roberts",
    "T": "Clarence Thomas",
    "CT": "Clarence Thomas",
    "A": "Samuel Alito",
    "SA": "Samuel Alito",
    "SS": "Sonia Sotomayor",
    "EK": "Elena Kagan",
    "NG": "Neil Gorsuch",
    "NMG": "Neil Gorsuch",
    "BK": "Brett Kavanaugh",
    "BMK": "Brett Kavanaugh",
    "AB": "Amy Coney Barrett",
    "ACB": "Amy Coney Barrett",
    "KJ": "Ketanji Brown Jackson",
    "KBJ": "Ketanji Brown Jackson",
    "PC": "Per Curiam",
    # Retired justices
    "SB": "Stephen Breyer",
    "SGB": "Stephen Breyer",
    "B": "Stephen Breyer",
    "RBG": "Ruth Bader Ginsburg",
    "AK": "Anthony Kennedy",
    "AMK": "Anthony Kennedy",
    "AS": "Antonin Scalia",
    "JP": "John Paul Stevens",
    "DS": "David Souter",
    "G": "Neil Gorsuch",
    "K": "Elena Kagan",
    "S": "Sonia Sotomayor",
}


@dataclass
class OpinionRecord:
    """Metadata for a single opinion entry from the slip opinions table."""
    term: int
    docket: str
    case_name: str
    date: str
    author_initials: str
    author_name: str
    pdf_url: str
    pdf_filename: str


def _fetch(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed HTML."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def resolve_author(initials: str) -> str:
    """Map author initials to full justice name."""
    initials = initials.strip().upper()
    if initials in JUSTICE_INITIALS:
        return JUSTICE_INITIALS[initials]
    # Try without periods
    clean = initials.replace(".", "")
    if clean in JUSTICE_INITIALS:
        return JUSTICE_INITIALS[clean]
    log.warning("Unknown justice initials: '%s'", initials)
    return f"Unknown ({initials})"


def scrape_term_opinions(term: int) -> list[OpinionRecord]:
    """Scrape the slip opinions page for a single term.

    The page at supremecourt.gov/opinions/slipopinion/{term} contains a table
    with columns: R- (sequence), Date, Docket, Name (with PDF link), J., Pt.
    """
    url = SLIP_OPINIONS_URL.format(term=term)
    log.info("Scraping opinions for term %d: %s", term, url)

    try:
        soup = _fetch(url)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            log.warning("No slip opinions page for term %d (404)", term)
            return []
        raise
    except requests.exceptions.RequestException as e:
        log.error("Error fetching term %d: %s", term, e)
        return []

    records: list[OpinionRecord] = []

    # The page may contain multiple tables (split by volume/date range).
    # Collect ALL table-bordered tables and iterate over all their rows.
    all_tables = soup.find_all("table", class_="table-bordered")
    if not all_tables:
        # Fallback: try any table with rows containing docket numbers
        for t in soup.find_all("table"):
            if len(t.find_all("tr")) > 2:
                all_tables.append(t)

    if not all_tables:
        log.warning("No opinions table found for term %d", term)
        return []

    all_rows = []
    for table in all_tables:
        all_rows.extend(table.find_all("tr"))

    for row in all_rows:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        try:
            # Typical column order: R-, Date, Docket, Name (with link), J., Pt.
            date = cells[1].get_text(strip=True)
            docket = cells[2].get_text(strip=True)
            name_cell = cells[3]
            author_cell = cells[4]

            # Extract PDF link from the case name cell
            pdf_link = name_cell.find("a", href=True)
            if not pdf_link:
                continue

            href = pdf_link["href"]
            # Accept .pdf links, including those with #page= anchors
            # (older terms use preliminaryprint URLs with page anchors)
            href_clean = href.split("#")[0]  # Remove anchor
            if not href_clean.lower().endswith(".pdf"):
                continue

            # Build absolute URL (keep the clean version without anchor)
            if href_clean.startswith("/"):
                pdf_url = f"https://www.supremecourt.gov{href_clean}"
            elif href_clean.startswith("http"):
                pdf_url = href_clean
            else:
                pdf_url = f"https://www.supremecourt.gov/opinions/{href_clean}"

            case_name = name_cell.get_text(strip=True)
            author_initials = author_cell.get_text(strip=True)
            author_name = resolve_author(author_initials)

            # Derive filename from URL
            pdf_filename = pdf_url.split("/")[-1]

            records.append(OpinionRecord(
                term=term,
                docket=docket,
                case_name=case_name,
                date=date,
                author_initials=author_initials,
                author_name=author_name,
                pdf_url=pdf_url,
                pdf_filename=pdf_filename,
            ))

        except (IndexError, AttributeError) as e:
            log.debug("Skipping row in term %d: %s", term, e)
            continue

    log.info("  Term %d: found %d opinions", term, len(records))
    return records


def download_opinion_pdf(record: OpinionRecord, dest_dir: Path) -> Path | None:
    """Download an opinion PDF. Returns path to saved file, or None on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / record.pdf_filename

    if dest.exists():
        return dest

    try:
        resp = requests.get(record.pdf_url, headers=HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(65536):
                if chunk:
                    f.write(chunk)
        return dest
    except Exception as e:
        log.error("Download failed %s: %s", record.pdf_filename, e)
        return None


def _read_checkpoint(term: int) -> bool:
    """Check if a term has already been fully scraped."""
    cp = config.checkpoint_path("opinions", term)
    return cp.exists()


def _write_checkpoint(term: int) -> None:
    """Mark a term as fully scraped."""
    cp = config.checkpoint_path("opinions", term)
    cp.write_text("done")


def scrape_all_opinions(
    terms: list[int] | None = None,
    resume: bool = True,
    rate_limit: float = 1.0,
) -> dict[int, list[OpinionRecord]]:
    """Scrape all opinion PDFs for the given terms.

    Returns a dict mapping term -> list of OpinionRecords.
    Also saves the master metadata catalog.
    """
    cfg = config.load()
    scraper_cfg = cfg.get("opinion_scraper", {})
    terms = terms or scraper_cfg.get("terms", [18, 19, 20, 21, 22, 23, 24, 25])
    rate_limit = scraper_cfg.get("rate_limit_seconds", rate_limit)

    all_records: dict[int, list[OpinionRecord]] = {}

    for term in terms:
        if resume and _read_checkpoint(term):
            log.info("Term %d already scraped, skipping (use resume=False to re-scrape)", term)
            # Load existing records from metadata
            continue

        records = scrape_term_opinions(term)
        if not records:
            continue

        term_dir = config.opinions_dir_for_term(term)
        downloaded = 0

        for record in records:
            path = download_opinion_pdf(record, term_dir)
            if path:
                downloaded += 1
            time.sleep(rate_limit)

        log.info("  Term %d: downloaded %d/%d opinion PDFs", term, downloaded, len(records))
        all_records[term] = records
        _write_checkpoint(term)

    # Save/update master metadata catalog
    _save_metadata(all_records)

    return all_records


def _save_metadata(records_by_term: dict[int, list[OpinionRecord]]) -> None:
    """Save the master metadata catalog as JSON."""
    metadata_path = config.opinions_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if present
    existing: dict = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            existing = json.load(f)

    # Merge new records
    for term, records in records_by_term.items():
        existing[str(term)] = [asdict(r) for r in records]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    log.info("Metadata catalog saved: %s", metadata_path)


def load_metadata() -> dict[int, list[OpinionRecord]]:
    """Load the master metadata catalog."""
    metadata_path = config.opinions_metadata_path()
    if not metadata_path.exists():
        return {}

    with open(metadata_path, encoding="utf-8") as f:
        raw = json.load(f)

    result: dict[int, list[OpinionRecord]] = {}
    for term_str, records in raw.items():
        term = int(term_str)
        result[term] = [OpinionRecord(**r) for r in records]

    return result


# --- CLI entry point ---

def main() -> None:
    """Run the opinion scraper from the command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape SCOTUS slip opinions")
    parser.add_argument("--terms", type=str, default=None,
                        help="Comma-separated terms, e.g. '18,19,20,21,22,23,24'")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-scrape even if checkpoint exists")
    args = parser.parse_args()

    terms = None
    if args.terms:
        terms = [int(t.strip()) for t in args.terms.split(",")]

    scrape_all_opinions(terms=terms, resume=not args.no_resume)


if __name__ == "__main__":
    main()
