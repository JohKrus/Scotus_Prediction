#!/usr/bin/env python3
"""SCOTUS v3 Setup — Scrape opinions and build per-justice RAG indices.

Run this once before running predictions:
    python run_v3_setup.py [--terms 18,19,20,21,22,23,24]
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("v3_setup.log"),
    ],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SCOTUS v3: Scrape opinions + build justice RAG")
    parser.add_argument("--terms", type=str, default=None,
                        help="Comma-separated terms to scrape (default: 18-25)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping, only build indices from existing PDFs")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-scrape even if checkpoints exist")
    args = parser.parse_args()

    terms = None
    if args.terms:
        terms = [int(t.strip()) for t in args.terms.split(",")]

    # Step 1: Scrape opinions
    if not args.skip_scrape:
        log.info("=" * 60)
        log.info("STEP 1: Scraping SCOTUS slip opinions")
        log.info("=" * 60)

        from scotus_v3.opinion_scraper import scrape_all_opinions
        records = scrape_all_opinions(terms=terms, resume=not args.no_resume)

        total = sum(len(r) for r in records.values())
        log.info("Scraped %d opinions across %d terms", total, len(records))
    else:
        log.info("Skipping scrape (--skip-scrape)")

    # Step 2: Build per-justice FAISS indices
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: Building per-justice FAISS indices")
    log.info("=" * 60)

    from scotus_v3.justice_rag import JusticeRAGManager
    manager = JusticeRAGManager()
    manager.build_all(terms=terms)

    log.info("")
    log.info("=" * 60)
    log.info("SETUP COMPLETE")
    log.info("=" * 60)
    log.info("Built indices for %d justices:", len(manager.stores))
    for name in manager.available_justices():
        log.info("  %s", manager.get_justice_summary(name, predict_term=99))
    log.info("")
    log.info("You can now run predictions with: python run_term22_v3.py")


if __name__ == "__main__":
    main()
