#!/usr/bin/env python3
"""
Fetch finished World Cup 2026 results and update MANUAL_RESULTS in app.py.
Primary source: ESPN public API (real-time).
Fallback source: martj42 international results CSV (can lag a few hours).
Run from the repo root.
"""

import csv
import io
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

APP_PY = "app.py"
CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date}"

# External name → app.py name (covers both ESPN and CSV differences)
NAME_MAP = {
    # ESPN variants
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo, DR": "DR Congo",
    "Curacao": "Curaçao",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Ivory Coast": "Ivory Coast",
    "Czechia": "Czech Republic",
    # CSV variants
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "USA": "United States",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",  # passthrough
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


def fetch_espn_results(dates):
    """Query ESPN scoreboard API for each date and return completed match scores."""
    results = {}
    for date_str in sorted(dates):
        url = ESPN_URL.format(date=date_str.replace("-", ""))
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  ESPN {date_str}: fetch error — {e}")
            continue

        found = 0
        for event in data.get("events", []):
            if not event.get("status", {}).get("type", {}).get("completed"):
                continue
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            home = away = hs = as_ = None
            for c in competitors:
                name = normalize(c.get("team", {}).get("displayName", ""))
                try:
                    score = int(c.get("score", "0") or 0)
                except ValueError:
                    score = 0
                if c.get("homeAway") == "home":
                    home, hs = name, score
                else:
                    away, as_ = name, score
            if home and away and hs is not None and as_ is not None:
                results[(home, away)] = (hs, as_)
                found += 1
        print(f"  ESPN {date_str}: {found} completed match(es)")
    return results


def fetch_csv_results():
    """Fetch martj42 CSV and return completed WC 2026 match scores."""
    req = Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
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
        h = normalize(row["home_team"])
        a = normalize(row["away_team"])
        results[(h, a)] = (hs_i, as_i)
    return results


def find_new_results(match_times, existing, espn, csv_results):
    """Return confirmed new results for finished fixtures not yet in MANUAL_RESULTS."""
    now_utc = datetime.now(timezone.utc)
    new = []
    for (h, a), utc_str in match_times.items():
        if (h, a) in existing:
            continue
        kickoff = datetime.strptime(utc_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if now_utc < kickoff + timedelta(hours=2):
            continue  # not finished yet

        score = espn.get((h, a)) or csv_results.get((h, a))
        if score is None:
            source = "ESPN+CSV"
            print(f"  Awaiting {source}: {h} vs {a} (kicked off {utc_str} UTC)")
            continue

        hs, as_i = score
        utc_date = utc_str.split()[0]
        group = TEAM_TO_GROUP.get(h, "?")
        src = "ESPN" if (h, a) in espn else "CSV"
        new.append((h, a, hs, as_i, utc_date, group, src))

    return sorted(new, key=lambda x: x[4])


def insert_results(content, new_results):
    m = re.search(r"(MANUAL_RESULTS\s*=\s*\[)(.*?)(\n\])", content, re.DOTALL)
    if not m:
        sys.exit("ERROR: Could not locate MANUAL_RESULTS block in app.py")
    additions = "".join(
        f'\n    ("{h}", "{a}", {hs}, {as_i}, "{date}"),  # Group {group}'
        for h, a, hs, as_i, date, group, *_ in new_results
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
    now_utc = datetime.now(timezone.utc)
    print(f"UTC now : {now_utc.strftime('%Y-%m-%d %H:%M')}")
    print(f"Existing: {len(existing)} results, {len(match_times)} fixtures")

    # Collect UTC dates of fixtures that should be finished but aren't yet recorded
    pending_dates = set()
    for (h, a), utc_str in match_times.items():
        if (h, a) in existing:
            continue
        kickoff = datetime.strptime(utc_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if now_utc >= kickoff + timedelta(hours=2):
            # Also fetch the day before: ESPN may list a late UTC match under local date
            pending_dates.add(utc_str.split()[0])
            prev = (kickoff - timedelta(days=1)).strftime("%Y-%m-%d")
            pending_dates.add(prev)

    espn_results: dict = {}
    if pending_dates:
        print(f"Fetching ESPN for dates: {', '.join(sorted(pending_dates))}")
        espn_results = fetch_espn_results(pending_dates)
        print(f"ESPN total: {len(espn_results)} completed match(es)")

    print("Fetching CSV fallback…")
    csv_results = fetch_csv_results()
    print(f"CSV total: {len(csv_results)} WC 2026 result(s)")

    new_results = find_new_results(match_times, existing, espn_results, csv_results)
    if not new_results:
        print("Nothing new to add.")
        return

    print(f"Adding {len(new_results)} result(s):")
    for h, a, hs, as_i, date, group, src in new_results:
        print(f"  [{src}] {h} {hs}–{as_i} {a}  ({date})  [Group {group}]")

    new_content = insert_results(content, new_results)
    with open(APP_PY, "w", encoding="utf-8") as f:
        f.write(new_content)

    msg = build_commit_message(new_results)
    with open("/tmp/wc_commit_msg.txt", "w") as f:
        f.write(msg)
    print(f"Commit: {msg}")


if __name__ == "__main__":
    main()
