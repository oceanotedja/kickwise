"""Diagnostic: see exactly what API-Football returns for our request."""
import requests

API_KEY = "8cff0f9812c1efb9e40bc575089ff090"   # same key as in predictor_step10.py

r = requests.get(
    "https://v3.football.api-sports.io/injuries",
    headers={"x-apisports-key": API_KEY},
    params={"league": 1, "season": 2026},
    timeout=30,
)
data = r.json()
print("HTTP status:", r.status_code)
print("Errors:", data.get("errors"))
print("Results found:", data.get("results"))
print("Your plan:", requests.get(
    "https://v3.football.api-sports.io/status",
    headers={"x-apisports-key": API_KEY}, timeout=30,
).json().get("response", {}).get("subscription"))