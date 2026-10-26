"""Read the lab's results and say in plain words what to do: keep your settings, or change these lines.

    python3 lab_verdict.py                       # after ./optimize.sh (or any backtest.py run)
    python3 lab_verdict.py lab_results live,hold5,live_hold

Judges only the out-of-sample part of each run (the months the lab never tuned on) and only trusts a
config that passed the whole bar (trades, profit factor, t-stat, random baseline, months positive).
Reads lab_results/expect_<config>.json; changes nothing.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import backtest as bt
from sparticus_strategy import DEFAULT_UNIVERSE, LIVE_DEFAULTS, LIVE_STRATEGIES, config_id


def live_tradeable(p) -> bool:
    return p.strategy in LIVE_STRATEGIES and p.bar_minutes == 1 and p.market_filter == "none"


def setting_changes(p) -> dict:
    now, new = asdict(LIVE_DEFAULTS), asdict(p)
    return {k: (now[k], new[k]) for k in now if now[k] != new[k]}


def verdict(out: Path, names: Optional[List[str]] = None) -> str:
    names = names or list(bt.EXPERIMENTS)
    current = config_id(LIVE_DEFAULTS, DEFAULT_UNIVERSE)
    rows = []
    for name in names:
        if name not in bt.EXPERIMENTS:
            continue
        p, _ = bt.EXPERIMENTS[name]
        cid = config_id(p, DEFAULT_UNIVERSE)
        f = out / f"expect_{cid}.json"
        if f.exists():
            rows.append((name, p, cid, json.loads(f.read_text())))
    if not rows:
        return f"No lab results in {out}/. Run ./optimize.sh first."

    L = ["Lab verdict (out-of-sample: the months the lab never tuned on)", "",
         f"{'experiment':16s} {'trades':>6s} {'per trade':>10s} {'PF':>5s} {'t-stat':>7s} {'months+':>8s}  bar"]
    for name, p, cid, e in sorted(rows, key=lambda r: -float(r[3].get("t_stat", 0))):
        tag = "  (your settings)" if cid == current else ("  (lab-only)" if not live_tradeable(p) else "")
        L.append(f"{name:16s} {int(e.get('oos_trades', 0)):6d} {float(e.get('per_trade_mean', 0)):+10.2f} "
                 f"{float(e.get('profit_factor', 0)):5.2f} {float(e.get('t_stat', 0)):+7.2f} "
                 f"{float(e.get('months_positive', 0)):8.0%}  {'PASS' if e.get('passed') else 'not yet'}{tag}")
    L.append("")

    passed = [r for r in rows if r[3].get("passed")]
    if not passed:
        L += ["Nothing passed the bar. Keep your current settings unchanged and keep paper trading:",
              "no setting change here is backed by evidence, so changing one now would only reset the test."]
        return "\n".join(L)
    best = max(passed, key=lambda r: float(r[3].get("t_stat", 0)))
    name, p, cid, e = best
    if cid == current:
        L.append("Your current settings passed the bar. Keep them and keep paper trading them unchanged.")
    elif not live_tradeable(p):
        L += [f"Best result: {name}, but the live bot can't trade it yet (lab-only). Keep your settings;",
              f"{name} is a candidate for a future version."]
    else:
        L += [f"Best config that passed: {name}. To switch, change these lines in sparticus_strategy.py:"]
        for k, (old, new) in setting_changes(p).items():
            L.append(f"    {k}: {old!r} -> {new!r}")
        L += ["Then paper trade it unchanged for at least 2 weeks and review it every Sunday."]
    return "\n".join(L)


def main(argv: List[str]) -> int:
    out = Path(argv[0]) if argv else Path("lab_results")
    names = argv[1].split(",") if len(argv) > 1 else None
    print(verdict(out, names))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
