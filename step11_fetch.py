"""
Step 11a - Fetch 2022 World Cup injury data (run this FIRST)
-------------------------------------------------------------
Downloads the official injury/absence list for every 2022 World
Cup match from API-Football (your free plan covers season 2022)
and saves it to a file: wc2022_injuries.json

Your API key stays on YOUR computer - never share it or paste it
into chats or screenshots.

Uses 1-3 of your 100 daily requests.

Run:  venv/bin/python step11_fetch.py
"""

import json
import requests

API_KEY = "8cff0f9812c1efb9e40bc575089ff090"

# API-Football sometimes names teams differently from our results
# dataset. This translates them. If the backtest reports unmatched
# names, add them here.
NAME_FIX = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
}


def fetch_all_pages():
    records, page = [], 1
    while True:
        r = requests.get(
            "https://v3.football.api-sports.io/injuries",
            headers={"x-apisports-key": API_KEY},
            params={"league": 1, "season": 2022},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            print("API reported errors:", data["errors"])
            break
        records += data.get("response", [])
        paging = data.get("paging", {})
        if page >= paging.get("total", 1):
            break
        page += 1
    return records


def main():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("Open this file and paste your API key into API_KEY first.")
        return

    print("Fetching 2022 World Cup injury records...")
    records = fetch_all_pages()
    print(f"Got {len(records)} raw records.")

    # Reshape into: {"TeamName|YYYY-MM-DD": [player names missing]}
    # Only count players actually OUT, not just doubtful.
    absences = {}
    skipped_questionable = 0
    for item in records:
        status = (item["player"].get("type") or "").lower()
        if "missing" not in status:          # e.g. "Questionable" - might still play
            skipped_questionable += 1
            continue
        team = item["team"]["name"]
        team = NAME_FIX.get(team, team)
        date = item["fixture"]["date"][:10]   # YYYY-MM-DD
        key = f"{team}|{date}"
        absences.setdefault(key, [])
        player = item["player"]["name"]
        if player not in absences[key]:
            absences[key].append(player)

    with open("wc2022_injuries.json", "w") as f:
        json.dump(absences, f, indent=2)

    n_players = sum(len(v) for v in absences.values())
    print(f"Saved wc2022_injuries.json:")
    print(f"  {len(absences)} team-match entries, {n_players} confirmed absences")
    print(f"  (skipped {skipped_questionable} merely 'questionable' records)")
    print("\nNow run:  venv/bin/python step11_backtest.py")


if __name__ == "__main__":
    main()
