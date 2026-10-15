#!/usr/bin/env python3
"""One-time script to fetch S&P 500 sector (GICS) data from yfinance.

Generates data/raw/sp500_sectors.csv with columns: ticker,sector

Usage:
    python scripts/generate_sectors.py

The output file is committed to the repository so it does not need to be
regenerated on every run. Re-run only when the S&P 500 membership changes
significantly and you want updated sector assignments.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    import yfinance as yf

    constituent_path = Path("data/raw/sp500_history.csv")
    if not constituent_path.exists():
        print(f"ERROR: {constituent_path} not found.")
        print("Run: python -m src.cli universe fetch-constituents")
        sys.exit(1)

    from src.config import load_config
    from src.universe.sp500 import SP500Universe
    from datetime import date

    cfg = load_config(Path("config"))
    sp500 = SP500Universe.from_csv(constituent_path)
    tickers = sorted(sp500.ever_members(cfg.data.start_date, cfg.data.end_date))

    output_path = Path("data/raw/sp500_sectors.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str]] = []
    total = len(tickers)

    print(f"Fetching sector data for {total} tickers...")
    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"

        results.append((ticker, sector))

        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] last: {ticker} → {sector}")

        time.sleep(0.3)  # throttle to avoid rate-limiting

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "sector"])
        writer.writerows(results)

    print(f"\nWritten: {output_path} ({len(results)} rows)")
    unknown = sum(1 for _, s in results if s == "Unknown")
    print(f"Unknown sectors: {unknown}/{len(results)}")


if __name__ == "__main__":
    main()
