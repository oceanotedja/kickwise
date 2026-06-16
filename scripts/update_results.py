#!/usr/bin/env python3
"""
Fetch finished World Cup 2026 results and update MANUAL_RESULTS in app.py.
Run from the repo root. Uses the martj42 international results CSV.
"""

import csv
import io
import re
import sys
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

APP_PY = "app.py"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# CSV team name → app.py team name
NAME_MAP = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Czechia": "Czech Republic",
}

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAM_TO_GROUP = {t: g for g, teams in GROUPS.items() for t in teams}


def normalize(name):
    return NAME_MAP.get(name, name)


def parse_existing_results(content):
    m = re.search(r"MANUAL_RESULTS\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not m:
        return set()
    pairs = set()
    for match in re.finditer(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,', m.group(1)):
        pairs.add((match.group(1), match.group(2)))
    return pairs


def parse_match_times(content):
    m = re.search(r"MATCH_TIMES_UTC\s*=\s*\{(.*?)\}", content, re.DOTALL)
    if not m:
        return {}
    times = {}
    pat = r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*:\s*"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})"'
    for match in re.finditer(pat, m.group(1)):
        times[(match.group(1), match.group(2))] = match.group(3)
    return times


def fetch_csv_results():
    req = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    results = {}
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        if "FIFA World Cup" not in row.get("tournament", ""):
            continue
        if not row.get("date", "").startswith("2026"):
            continue
        hs = row.get("home_score", "").strip()
        as_ = row.get("away_score", "").strip()
        if not hs or not as_:
            continue
        try:
            hs_i, as_i = int(float(hs)), int(float(as_))
        except ValueError:
            continue
        h, a = normalize(row["home_team"]), normalize(row["away_team"])
        results[(h, a)] = (hs_i, as_i)
    return results


def find_new_results(match_times, existing, csv_results):
    now_utc = datetime.now(timezone.utc)
    new = []
    for (h, a), utc_str in match_times.items():
        if (h, a) in existing:
            continue
        kickoff = datetime.strptime(utc_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if now_utc < kickoff + timedelta(hours=2):
            continue
        if (h, a) not in csv_results:
            print(f"  Awaiting CSV: {h} vs {a} (kicked off {utc_str} UTC)")
            continue
        hs, as_i = csv_results[(h, a)]
        utc_date = utc_str.split()[0]
        group = TEAM_TO_GROUP.get(h, "?")
        new.append((h, a, hs, as_i, utc_date, group))
    return sorted(new, key=lambda x: x[4])


def insert_results(content, new_results):
    m = re.search(r"(MANUAL_RESULTS\s*=\s*\[)(.*?)(\n\])", content, re.DOTALL)
    if not m:
        sys.exit("ERROR: Could not locate MANUAL_RESULTS block in app.py")
    additions = "".join(
        f'\n    ("{h}", "{a}", {hs}, {as_i}, "{date}"),  # Group {group}'
        for h, a, hs, as_i, date, group in new_results
    )
    return (
        content[: m.start()]
        + m.group(1)
        + m.group(2).rstrip()
        + additions
        + "\n"
        + m.group(3)
        + content[m.end() :]
    )


def build_commit_message(new_results):
    dates = sorted({r[4] for r in new_results})
    groups = sorted({r[5] for r in new_results})
    if len(dates) == 1:
        dt = datetime.strptime(dates[0], "%Y-%m-%d")
        date_label = dt.strftime("%b %-d")
    else:
        dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        date_label = " & ".join(dt.strftime("%b %-d") for dt in dts)
    groups_str = " & ".join(f"Group {g}" for g in groups)
    return f"Add {date_label} results ({groups_str})"


def main():
    with open(APP_PY, encoding="utf-8") as f:
        content = f.read()

    existing = parse_existing_results(content)
    match_times = parse_match_times(content)
    print(f"UTC now : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    print(f"Existing: {len(existing)} results, {len(match_times)} fixtures")

    print("Fetching CSV…")
    csv_results = fetch_csv_results()
    print(f"CSV has {len(csv_results)} WC 2026 results")

    new_results = find_new_results(match_times, existing, csv_results)
    if not new_results:
        print("Nothing new to add.")
        return

    print(f"Adding {len(new_results)} result(s):")
    for h, a, hs, as_i, date, group in new_results:
        print(f"  {h} {hs}–{as_i} {a}  ({date})  [Group {group}]")

    new_content = insert_results(content, new_results)
    with open(APP_PY, "w", encoding="utf-8") as f:
        f.write(new_content)

    msg = build_commit_message(new_results)
    with open("/tmp/wc_commit_msg.txt", "w") as f:
        f.write(msg)
    print(f"Commit: {msg}")


if __name__ == "__main__":
    main()
