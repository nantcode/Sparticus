"""Incubation tracker and strategy kill switch.

Idea (from forward-testing practice): a config that looks good in the lab still has to prove it
on live paper trades it has never seen. This module compares the live journal for the CURRENT
config against what the lab expects from that exact config (lab_results/expect_<config_id>.json),
and decides when to switch autopilot off. Pure functions + two small file helpers; no broker calls.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

MIN_TRADES = 30                 # below this, no verdict: just collect data
DEFAULT_KILL_DD_PCT = 0.03      # no lab baseline -> kill at a drawdown of 3% of equity
BELOW_Z = -2.0                  # live average this many standard errors below the lab = warning


def load_expectations(folder: str, config_id: str) -> Optional[Dict[str, Any]]:
    path = Path(folder) / f"expect_{config_id}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def evaluate(live: Dict[str, Any], expect: Optional[Dict[str, Any]], equity: float) -> Dict[str, Any]:
    """live = sparticus_stats.summarize() of this config's journal rows.

    Returns {status, line, kill, kill_at, z}. status is one of: collecting, no_baseline, on_track,
    below, above, kill."""
    n = int(live.get("trades", 0) or 0)
    dd = float(live.get("max_drawdown", 0.0) or 0.0)            # <= 0, in $
    expect = scaled_to_equity(expect, equity)
    if expect and float(expect.get("mc_dd95", 0.0)) < 0:
        kill_at = float(expect["mc_dd95"])
        kill_src = "the lab's 95% worst case"
    else:
        kill_at = -abs(equity) * DEFAULT_KILL_DD_PCT
        kill_src = f"{DEFAULT_KILL_DD_PCT:.0%} of equity (no lab baseline)"
    out: Dict[str, Any] = {"kill_at": kill_at, "kill": False, "z": None, "trades": n}

    if n and dd <= kill_at:
        out.update(status="kill", kill=True,
                   line=f"KILL: drawdown {_m(dd)} passed {_m(kill_at)} ({kill_src}). Autopilot off.")
        return out

    mean = float(live.get("expectancy", 0.0) or 0.0)
    tail = f" | {n} trades, {_m(mean)}/trade, drawdown {_m(dd)} (kill at {_m(kill_at)})"
    if expect is None:
        out.update(status="no_baseline",
                   line="No lab baseline for this config: run backtest.py with these settings" + tail)
        return out
    if n < MIN_TRADES:
        out.update(status="collecting", line=f"Incubating: collecting data ({n}/{MIN_TRADES} trades)" + tail)
        return out
    lab_mean = float(expect.get("per_trade_mean", 0.0))
    lab_std = float(expect.get("per_trade_std", 0.0))
    z = (mean - lab_mean) / (lab_std / math.sqrt(n)) if lab_std > 0 else 0.0
    out["z"] = z
    if z <= BELOW_Z:
        out.update(status="below", line=f"BELOW the lab: live {_m(mean)} vs lab {_m(lab_mean)} per trade (z {z:+.1f})" + tail)
    elif z >= -BELOW_Z:
        out.update(status="above", line=f"Above the lab (luck, or the lab is pessimistic): z {z:+.1f}" + tail)
    else:
        out.update(status="on_track", line=f"On track with the lab ({_m(lab_mean)}/trade expected, z {z:+.1f})" + tail)
    return out


def scaled_to_equity(expect: Optional[Dict[str, Any]], equity: float) -> Optional[Dict[str, Any]]:
    """Positions sized as a % of equity (micro mode): the lab's $ figures come from its own start
    equity (e.g. $100), so rescale them to this account. Fixed-$ sizing: unchanged."""
    if not expect or not expect.get("pct_sizing"):
        return expect
    lab_eq = float(expect.get("lab_equity", 0.0) or 0.0)
    if lab_eq <= 0 or equity <= 0:
        return expect
    k = equity / lab_eq
    out = dict(expect)
    for key in ("per_trade_mean", "per_trade_std", "mc_dd_median", "mc_dd95"):
        if key in out:
            out[key] = float(out[key]) * k
    return out


def write_kill(path: str, config_id: str, reason: str) -> None:
    Path(path).write_text(json.dumps({"config_id": config_id, "reason": reason,
                                      "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))


def read_kill(path: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _m(x: float) -> str:
    sign = "-" if x < 0 else "+"
    if abs(x) < 1:
        return f"{sign}${abs(x):.3f}"          # micro accounts: fractions of a cent matter
    return f"{sign}${abs(x):,.2f}" if abs(x) < 100 else f"{sign}${abs(x):,.0f}"
