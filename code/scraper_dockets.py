"""Part 1 — Scrape SCOTUS docket pages and download substantive PDFs."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from scotus import config

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (pdf-collector/4.0)"}

# Regex for substantive vs. non-substantive document filtering
KEEP = re.compile(
    r"(petition|brief|appendix|merits|amicus|amici|reply|supplemental|"
    r"joint appendix|transcript)",
    re.I,
)
SKIP = re.compile(
    r"(motion|order|extend|extension|certificate|proof of service|"
    r"rehearing|waiver|distributed|record|compliance)",
    re.I,
)
DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{2} \d{4}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _sanitize_filename(name: str, max_length: int = 100) -> str:
    name = re.sub(r"[<>:\"/\\|?*]", "_", name)
    name = re.sub(r"%[0-9A-Fa-f]{2}", "_", name)
    name = re.sub(r"[^\w\-_.]", "_", name)
    name = re.sub(r"_+", "_", name)

    if "." in name:
        stem, ext = name.rsplit(".", 1)
        ext = "." + ext
    else:
        stem, ext = name, ".pdf"

    if len(stem) > max_length:
        h = hashlib.md5(stem.encode()).hexdigest()[:8]
        stem = stem[: max_length - 10] + "_" + h

    return stem + ext


def _is_substantive(desc: str, link_text: str) -> bool:
    blob = f"{desc} {link_text}".lower()
    return bool(KEEP.search(blob)) and not SKIP.search(blob)


def _extract_date(table) -> str:
    text = table.get_text()
    m = DATE_RE.search(text)
    return m.group() if m else ""


def _extract_description(table) -> str:
    text = table.get_text(" ", strip=True)
    text = re.sub(r"(Main Document|Proof of Service|Certificate of Word Count)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def _nearest_date_row(tr) -> tuple[str, str]:
    prev = tr.previous_sibling
    while prev and prev.name != "tr":
        prev = prev.previous_sibling
    if prev:
        tds = prev.find_all("td")
        if len(tds) >= 2 and DATE_RE.match(tds[0].get_text(strip=True)):
            return tds[0].get_text(strip=True), tds[1].get_text(" ", strip=True)
    return "", ""


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------

def gather_items(docket_num: str) -> list[dict]:
    """Collect PDF links from a single docket page."""
    url = f"https://www.supremecourt.gov/docket/docketfiles/html/public/{docket_num}.html"

    try:
        soup = _fetch(url)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return []
        log.warning("Error fetching %s: %s", docket_num, e)
        return []
    except requests.exceptions.RequestException as e:
        log.warning("Error fetching %s: %s", docket_num, e)
        return []

    title = soup.title.string if soup.title else ""
    if "Search - Supreme Court" in title:
        return []
    if "proceedings" not in soup.get_text().lower() and f"Docket for {docket_num}" not in title:
        return []

    # Try new ProceedingItem tables first, fall back to old format
    proceeding_tables = soup.find_all("table", class_=lambda x: x and "ProceedingItem" in x)
    old_pdf_links = None

    if not proceeding_tables:
        proceedings_table = soup.find("table", id="proceedings")
        if proceedings_table:
            old_pdf_links = proceedings_table.find_all("a", href=re.compile(r"\.pdf$", re.I))
        else:
            return []

    items: list[dict] = []

    if proceeding_tables:
        for table in proceeding_tables:
            pdf_links = table.find_all("a", href=re.compile(r"\.pdf$", re.I))
            if not pdf_links:
                continue
            date = _extract_date(table)
            desc = _extract_description(table)
            for link_tag in pdf_links:
                link_text = link_tag.get_text(strip=True) or "Main Document"
                if not _is_substantive(desc, link_text):
                    continue
                href = urljoin(url, link_tag["href"])
                orig = os.path.basename(urlparse(href).path)
                items.append({
                    "date": date,
                    "description": desc,
                    "link_text": link_text,
                    "url": href,
                    "filename": _sanitize_filename(orig),
                    "original_filename": orig,
                })
    elif old_pdf_links:
        for link_tag in old_pdf_links:
            date, desc = _nearest_date_row(link_tag.parent.parent)
            if not date:
                continue
            if not _is_substantive(desc, link_tag.get_text(strip=True)):
                continue
            href = urljoin(url, link_tag["href"])
            orig = os.path.basename(urlparse(href).path)
            items.append({
                "date": date,
                "description": desc,
                "link_text": link_tag.get_text(strip=True) or "Main Document",
                "url": href,
                "filename": _sanitize_filename(orig),
                "original_filename": orig,
            })

    return items


def download_pdf(item: dict, pdf_dir: Path) -> bool:
    """Download a single PDF. Returns True on success."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / item["filename"]

    if len(str(dest)) > 250:
        short = f"doc_{hashlib.md5(item['url'].encode()).hexdigest()[:12]}.pdf"
        dest = pdf_dir / short
        item["filename"] = short

    if dest.exists():
        return True

    try:
        resp = requests.get(item["url"], headers=HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(65536):
                if chunk:
                    f.write(chunk)
        time.sleep(0.5)
        return True
    except Exception as e:
        log.error("Download failed %s: %s", item["filename"], e)
        return False


def save_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["date", "description", "link_text", "filename", "original_filename", "url"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _read_checkpoint(term: int) -> int:
    """Return the last successfully processed docket index (0 if none)."""
    cp = config.checkpoint_path("scrape", term)
    if cp.exists():
        try:
            return int(cp.read_text().strip())
        except ValueError:
            pass
    return 0


def _write_checkpoint(term: int, index: int) -> None:
    cp = config.checkpoint_path("scrape", term)
    cp.write_text(str(index))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(
    terms: list[int] | None = None,
    docket_range: tuple[int, int] | None = None,
    single_docket: str | None = None,
    resume: bool = False,
) -> None:
    """Scrape PDFs for the given terms and docket range."""
    cfg = config.load()
    terms = terms or cfg["terms"]
    lo, hi = docket_range or tuple(cfg["docket_range"])
    rate = cfg.get("rate_limit_seconds", 1)

    if single_docket:
        _process_docket(single_docket, int(single_docket.split("-")[0]))
        return

    for term in terms:
        log.info("=== Term 20%d ===", term)
        start_at = _read_checkpoint(term) + 1 if resume else lo
        if start_at > lo:
            log.info("Resuming from docket index %d", start_at)

        processed = 0
        downloaded = 0

        for i in range(start_at, hi + 1):
            docket = f"{term}-{i:03d}"
            ok = _process_docket(docket, term)
            if ok:
                processed += 1
                downloaded += ok  # ok is the count of PDFs
            _write_checkpoint(term, i)

            if processed % 50 == 0 and processed > 0:
                log.info("  Progress: %d dockets processed for Term 20%d", processed, term)

            time.sleep(rate)

        log.info("Term 20%d done — %d dockets, %d PDFs", term, processed, downloaded)


MIN_DOCS = 5  # dockets with fewer items were not decided on the merits


def _process_docket(docket: str, term: int) -> int:
    """Process one docket. Returns number of downloaded PDFs."""
    items = gather_items(docket)
    if len(items) < MIN_DOCS:
        return 0

    pdf_dir = config.pdf_dir_for_docket(term, docket)
    csv_path = config.csv_path_for_docket(term, docket)

    ok = sum(1 for item in items if download_pdf(item, pdf_dir))
    save_csv(items, csv_path)
    log.info("  %s: %d/%d PDFs downloaded", docket, ok, len(items))
    return ok
