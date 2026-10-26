"""Check a folder for keys, tokens and private files BEFORE you zip it or send it to anyone.

    python3 scan_for_secrets.py                 # scan this folder
    python3 scan_for_secrets.py ~/some/folder   # scan another one

Exit code 0 = nothing found. Exit code 1 = something must come out first (listed, values masked).

It looks for:
  * Alpaca, AWS and GitHub keys, Discord webhook URLs, private keys, "secret = ..." style values
  * files that should never leave your machine: .env, logs, trade journals, order exports,
    downloaded market data, AWS credentials
It only reads files. Nothing is changed or sent anywhere.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".cfg", ".ini",
                 ".env", ".example", ".log", ".js", ".ts", ".html", ""}
MAX_BYTES = 5_000_000

PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Alpaca API key id", re.compile(r"\b(?:PK|AK|CK)[A-Z0-9]{16,30}\b")),
    ("Discord webhook URL (anyone with it can post as your bot)",
     re.compile(r"discord(?:app)?\.com/api/webhooks/\d+/[\w\-]{20,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("secret value", re.compile(r"(?i)\b[\w-]*(?:secret|api_?key|token|password|passwd)[\w-]*\s*[=:]\s*['\"]?"
                                r"(?!your-|<|\$\{|os\.getenv|None\b|\"\"|''|$)([A-Za-z0-9/+_\-]{16,})")),
]

PRIVATE_FILES = [
    (re.compile(r"^\.env(?!\.example$)(\..*)?$"), "your keys (.env)"),
    (re.compile(r".*\.log$"), "a log file (can contain order ids, account details and old webhook URLs)"),
    (re.compile(r".*journal.*\.csv$", re.I), "your trade journal (your trading history)"),
    (re.compile(r"^orders.*\.(txt|csv)$", re.I), "an Alpaca order export (your trading history)"),
    (re.compile(r"^(credentials|config)$"), "possible AWS credentials file"),
    (re.compile(r".*\.(key|p12|pfx)$"), "a private key or certificate"),
    (re.compile(r".*\.tfstate(\.backup)?$"), "Terraform state (can contain secrets)"),
    (re.compile(r"^sparticus\.lock$"), "a lock file from a running bot"),
]
PRIVATE_DIRS = {"lab_data": "downloaded market data (large, and tied to your account)",
                "reviews": "your trade reviews", "backup_before_v6_3_1": "backups of your files",
                "backup_before_v6_4": "backups of your files", ".aws": "AWS credentials",
                ".aws-sam": "SAM build output (sam build recreates it; don't ship it)"}


def _mask(s: str) -> str:
    return s[:4] + "…" + s[-2:] if len(s) > 8 else "…"


def scan(root: Path) -> List[str]:
    problems: List[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir():
            if path.name in PRIVATE_DIRS:
                problems.append(f"{rel}/  -> remove: {PRIVATE_DIRS[path.name]}")
            continue
        for rx, why in PRIVATE_FILES:
            if rx.match(path.name):
                problems.append(f"{rel}  -> remove: {why}")
                break
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for label, rx in PATTERNS:
                m = rx.search(line)
                if not m:
                    continue
                hit = m.group(m.lastindex or 0)
                if label == "secret value" and not (re.search(r"\d", hit) and re.search(r"[A-Za-z]", hit)):
                    continue  # names and paths (secrets_from_ssm, /app/secret_key) aren't values; real keys mix digits in
                problems.append(f"{rel}:{n}  -> {label}: {_mask(hit)}")
    return problems


def main(argv: List[str]) -> int:
    root = Path(argv[0]).expanduser() if argv else Path(".")
    if not root.is_dir():
        print(f"Not a folder: {root}")
        return 2
    problems = scan(root.resolve())
    if not problems:
        print(f"OK: no keys, tokens or private files found in {root.resolve()}")
        return 0
    print(f"STOP: {len(problems)} thing(s) in {root.resolve()} must not be shared:")
    for p in problems:
        print("  " + p)
    print("Remove them (or move them out of this folder), then run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
