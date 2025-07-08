"""Email QC Bot – first draft

Checks marketing email drafts against authoritative event metadata and flags factual or style issues.

Setup:
1. Put your event metadata in `events.csv` with columns:
   event_id,title,date,start_time,zoom_link,price,salespage_url
2. Place *.md or *.txt drafts in an `emails/` folder. Filenames should start with the corresponding event_id, e.g. `EVT23_invite_v1.md`.
3. Export `OPENAI_API_KEY` in your environment.
4. (Optional) set a `SLACK_WEBHOOK_URL` env var to send summaries to Slack.

Usage:
$ python email_qc_bot.py

Outputs per‑file JSON report and a colourised CLI summary; sends a condensed Slack message if webhook configured.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from termcolor import colored
from openai import OpenAI

client = OpenAI()

EVENT_CSV = Path("events.csv")
EMAIL_DIR = Path("emails")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

STYLE_RULES = {
    "subject_max_chars": 55,
    "preview_max_chars": 120,
    "body_max_chars": 1800,
}

SYSTEM_PROMPT = (
    "You are an email quality‑assurance assistant. "
    "Given authoritative event metadata and an email draft, produce:\n"
    "1. A list of factual errors (if any) – cite the field.\n"
    "2. Style warnings (length limits, missing links, branding tone).\n"
    "3. A one‑sentence traffic‑light summary: GREEN OK, YELLOW warnings, RED critical.\n"
    "Respond as JSON with keys: errors, warnings, traffic_light."
)


def load_events():
    if not EVENT_CSV.exists():
        raise FileNotFoundError("events.csv not found – create your source‑of‑truth file first.")
    events_df = pd.read_csv(EVENT_CSV, dtype=str).fillna("")
    return {row.event_id: row.to_dict() for _, row in events_df.iterrows()}


def gather_emails():
    if not EMAIL_DIR.exists():
        raise FileNotFoundError("emails/ directory not found.")
    return list(EMAIL_DIR.glob("*.md")) + list(EMAIL_DIR.glob("*.txt"))


def extract_event_id(filename: Path):
    return filename.stem.split("_", 1)[0]


def call_llm(event_meta: dict, email_text: str):
    user_prompt = {
        "event": event_meta,
        "email": email_text,
        "style_rules": STYLE_RULES,
    }
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap & fast
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt)}
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def colour_light(light):
    return {
        "GREEN": colored("GREEN", "green"),
        "YELLOW": colored("YELLOW", "yellow"),
        "RED": colored("RED", "red"),
    }.get(light, light)


def send_slack(message: str):
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        return
    requests.post(webhook, json={"text": message})


def main():
    events = load_events()
    email_files = gather_emails()
    overall = []

    for f in email_files:
        event_id = extract_event_id(f)
        if event_id not in events:
            print(colored(f"[SKIP] {f.name}: event_id '{event_id}' not in events.csv", "red"))
            continue
        email_text = f.read_text(encoding="utf-8")
        report = call_llm(events[event_id], email_text)

        # write full report
        report_path = REPORT_DIR / f"{f.stem}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # CLI summary
        tl = report.get("traffic_light", "?")
        print(f"{f.name}: {colour_light(tl)} – {len(report.get('errors', []))} errors, {len(report.get('warnings', []))} warnings")
        overall.append((f.name, tl))

    # Slack summary
    if os.getenv("SLACK_WEBHOOK_URL"):
        summary_lines = [f"*{name}* → {tl}" for name, tl in overall]
        send_slack("QC Summary (" + datetime.utcnow().strftime("%Y‑%m‑%d %H:%M") + "):\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
