#!/usr/bin/env python3
"""
Fetches completed FIFA World Cup 2026 group stage results from the
martj42 dataset and appends any new ones to data/manual_results.json.

Run manually:  python scripts/fetch_results.py
Run via CI:    called by .github/workflows/sync_results.yml every hour.
"""

import json
import os
import sys

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_FILE = os.path.join(ROOT, "data", "manual_results.json")

# Normalise CSV team names → names used in app.py / GROUPS
NAME_MAP = {
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Cape Verde Is.": "Cape Verde",
    "United States": "United States",
}

def normalise(name):
    return NAME_MAP.get(name, name)

def load_existing():
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def fetch_from_csv():
    print(f"Downloading {DATA_URL} ...")
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"])

    # Group stage: June 11–27 2026, FIFA World Cup, both scores present
    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2026-06-11")
        & (df["date"] <= "2026-06-27")
        & df["home_score"].notna()
        & df["away_score"].notna()
    )
    completed = df[mask].copy()
    print(f"  Found {len(completed)} completed WC2026 group stage match(es) in CSV.")

    results = []
    for _, row in completed.iterrows():
        results.append({
            "home":       normalise(row["home_team"]),
            "away":       normalise(row["away_team"]),
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "date":       row["date"].strftime("%Y-%m-%d"),
        })
    return results

def main():
    existing = load_existing()
    existing_pairs = {(r["home"], r["away"]) for r in existing}

    try:
        fetched = fetch_from_csv()
    except Exception as exc:
        print(f"ERROR fetching CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    new_results = [r for r in fetched if (r["home"], r["away"]) not in existing_pairs]

    if new_results:
        for r in new_results:
            print(f"  + {r['home']} {r['home_score']}–{r['away_score']} {r['away']}  ({r['date']})")
        updated = existing + new_results
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(updated, f, indent=2)
        print(f"Saved {len(updated)} total result(s) to {RESULTS_FILE}")
    else:
        print("No new results found — nothing to update.")

if __name__ == "__main__":
    main()
