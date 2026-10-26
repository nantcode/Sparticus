"""Sparticus trade review: turn an Alpaca order history into a report you can act on.

    python3 trade_review.py orders.txt            # the Orders page copied from the Alpaca dashboard
    python3 trade_review.py orders.csv            # or a CSV export of the same page

How to get orders.txt: Alpaca dashboard -> Orders -> set Page Size to 500 -> select the whole table
-> copy -> paste into a text file. Works for the paper and live accounts, whole and fractional shares.

It rebuilds every round trip (buy -> sell) from the FILLED orders, first in first out, and writes:
    reviews/review_<first day>_<last day>.md      the report
    reviews/trades_<first day>_<last day>.csv     one row per round trip

Exit types come from the order that closed the trade:
    target  = the bracket's take-profit limit filled
    stop    = the bracket's stop (or micro mode's broker-side stop) filled
    eod     = a market sell at 15:55 or later (the end-of-day flatten)
    signal  = any other market sell: trend-reversal exits, the watchdog stop, manual closes
              (the journal knows which; the order history doesn't)

Reads only the file you give it. No keys, no network.
"""
from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

TIME_FMT = "%b %d, %Y, %I:%M:%S %p"
SYMBOL_RE = re.compile(r"^[A-Z][A-Z.]{0,6}$")


# ---------------------------------------------------------------------------- reading
def _num(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "").replace("$", "")
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _time(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if s in ("", "-"):
        return None
    for fmt in (TIME_FMT, "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _order(sym, otype, side, qty, filled, px, status, submitted, filled_at) -> Dict:
    return {"sym": sym.strip().upper(), "type": (otype or "").strip(), "side": (side or "").strip().lower(),
            "qty": _num(qty) or 0.0, "filled": _num(filled) or 0.0, "px": _num(px),
            "status": (status or "").strip().lower(), "submitted": _time(submitted), "filled_at": _time(filled_at)}


def parse_paste(text: str) -> List[Dict]:
    """The Orders table copied from the dashboard: a symbol line, then one tab-separated line with
    type, side, qty, filled qty, avg fill price, status, source, submitted at, filled at, expires at."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines) - 1:
        sym = lines[i].strip()
        fields = lines[i + 1].split("\t")
        if SYMBOL_RE.match(sym) and len(fields) >= 9:
            out.append(_order(sym, fields[0], fields[1], fields[2], fields[3], fields[4], fields[5],
                              fields[7], fields[8]))
            i += 2
        else:
            i += 1
    return out


def parse_csv(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def pick(r: Dict, *names: str) -> str:
        low = {k.strip().lower(): v for k, v in r.items() if k}
        for n in names:
            if n in low:
                return low[n] or ""
        return ""
    return [_order(pick(r, "asset", "symbol"), pick(r, "order type", "type"), pick(r, "side"),
                   pick(r, "qty", "quantity"), pick(r, "filled qty", "filled_qty"),
                   pick(r, "avg. fill price", "filled_avg_price", "avg fill price"), pick(r, "status"),
                   pick(r, "submitted at", "submitted_at"), pick(r, "filled at", "filled_at"))
            for r in rows]


def load_orders(path: Path) -> List[Dict]:
    text = path.read_text(errors="replace")
    orders = parse_paste(text)
    if not orders and path.suffix.lower() == ".csv":
        orders = parse_csv(path)
    return orders


# ---------------------------------------------------------------------------- round trips
def round_trips(orders: List[Dict]):
    """FIFO-match filled buys and sells per symbol. Returns (trades, sells_without_a_buy, still_open)."""
    fills = sorted((o for o in orders if o["status"] == "filled" and o["filled"] > 0 and o["px"] and o["filled_at"]),
                   key=lambda o: o["filled_at"])
    lots: Dict[str, deque] = defaultdict(deque)
    trades, orphans = [], []
    for o in fills:
        kind = o["type"].split(" ")[0].lower()
        if o["side"] == "buy":
            m = re.search(r"\$([\d,\.]+)", o["type"])
            lots[o["sym"]].append({"qty": o["filled"], "px": o["px"], "t": o["filled_at"],
                                   "limit": _num(m.group(1)) if m else None})
            continue
        if kind == "limit":
            reason = "target"
        elif kind.startswith("stop"):
            reason = "stop"
        elif o["filled_at"].strftime("%H:%M") >= "15:55":
            reason = "eod"
        else:
            reason = "signal"
        left = o["filled"]
        while left > 1e-9:
            if not lots[o["sym"]]:
                orphans.append({"sym": o["sym"], "qty": left, "px": o["px"], "t": o["filled_at"]})
                break
            lot = lots[o["sym"]][0]
            q = min(left, lot["qty"])
            pnl = (o["px"] - lot["px"]) * q
            trades.append({"sym": o["sym"], "day": lot["t"].date().isoformat(), "entry_time": lot["t"],
                           "exit_time": o["filled_at"], "qty": q, "entry": lot["px"], "exit": o["px"],
                           "entry_limit": lot["limit"], "reason": reason, "pnl": pnl,
                           "notional": lot["px"] * q, "bps": (o["px"] / lot["px"] - 1) * 1e4,
                           "hold_min": (o["filled_at"] - lot["t"]).total_seconds() / 60})
            lot["qty"] -= q
            left -= q
            if lot["qty"] <= 1e-9:
                lots[o["sym"]].popleft()
    still_open = {s: sum(l["qty"] for l in q) for s, q in lots.items() if q}
    trades.sort(key=lambda t: t["entry_time"])
    seen: Dict[tuple, int] = defaultdict(int)
    for t in trades:
        seen[(t["sym"], t["day"])] += 1
        t["nth_today"] = seen[(t["sym"], t["day"])]
    return trades, orphans, still_open


# ---------------------------------------------------------------------------- statistics
def stats(ts: List[Dict]) -> Dict:
    n = len(ts)
    if not n:
        return {"n": 0}
    pnl = [t["pnl"] for t in ts]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    mean = sum(pnl) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnl) / (n - 1)) if n > 1 else 0.0
    notional = sum(t["notional"] for t in ts)
    return {"n": n, "net": sum(pnl), "mean": mean, "bps": sum(pnl) / notional * 1e4 if notional else 0.0,
            "win": len(wins) / n, "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf"),
            "t": mean / (sd / math.sqrt(n)) if sd > 0 else 0.0,
            "hold": sorted(t["hold_min"] for t in ts)[n // 2]}


def verdict(s: Dict) -> str:
    if s.get("n", 0) < 30:
        return "too few trades to judge (under 30)"
    if s["t"] <= -2:
        return "losing, and it's not bad luck (t-stat at or below -2)"
    if s["t"] >= 2 and s["pf"] >= 1.2:
        return "making money beyond luck so far: confirm it in the lab and keep incubating"
    return "no edge visible yet: results are within luck of zero"


def money(x: float) -> str:
    sign = "-" if x < 0 else "+"
    return f"{sign}${abs(x):,.2f}" if abs(x) < 1000 else f"{sign}${abs(x):,.0f}"


def _row(label: str, s: Dict) -> str:
    if not s.get("n"):
        return f"| {label} | 0 | | | | | |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {label} | {s['n']} | {money(s['net'])} | {s['bps']:+.1f} | {s['win']:.0%} | {pf} | "
            f"{s['hold']:.0f} min |")


HEAD = "| | Trades | Net | Per trade (bps) | Win rate | Profit factor | Median hold |\n|---|---:|---:|---:|---:|---:|---:|"
HOLD_BUCKETS = [(0, 7, "under 7 min"), (7, 15, "7-15 min"), (15, 25, "15-25 min"), (25, 45, "25-45 min"),
                (45, 1e9, "over 45 min")]


def report(trades: List[Dict], orphans: List[Dict], still_open: Dict[str, float], source: str) -> str:
    s = stats(trades)
    first, last = trades[0]["day"], trades[-1]["day"]
    L = [f"# Sparticus trade review: {first} to {last}", "",
         f"From `{source}`: {s['n']} round trips rebuilt from filled orders.", ""]
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    L += ["## Summary", "",
          f"* **Net:** {money(s['net'])} over {s['n']} trades ({money(s['mean'])} per trade, {s['bps']:+.1f} bps of each position)",
          f"* **Win rate:** {s['win']:.0%}. Average win {money(s['avg_win'])}, average loss {money(s['avg_loss'])}",
          f"* **Profit factor:** {pf} (above 1.2 is the lab's bar)",
          f"* **t-stat:** {s['t']:+.2f}. **Verdict:** {verdict(s)}",
          f"* **Median hold:** {s['hold']:.0f} min. Trades per day: "
          + ", ".join(f"{d} {sum(1 for t in trades if t['day'] == d)}" for d in sorted({t['day'] for t in trades})), ""]
    if still_open:
        L += [f"**⚠ Still open at the end of this history:** {', '.join(f'{k} {v:g} sh' for k, v in still_open.items())}. "
              "If the history ends after the close, these were held overnight.", ""]
    if orphans:
        L += [f"*{len(orphans)} sell(s) closed positions opened before this history starts; they're left out:* "
              + ", ".join(f"{o['sym']} {o['qty']:g}" for o in orphans), ""]

    L += ["## By day", "", HEAD]
    for d in sorted({t["day"] for t in trades}):
        L.append(_row(d, stats([t for t in trades if t["day"] == d])))
    L += ["", "## How trades ended", "", HEAD]
    for r, label in (("target", "take-profit filled"), ("stop", "stop filled"), ("signal", "signal / watchdog / manual sell"),
                     ("eod", "15:55 flatten")):
        L.append(_row(label, stats([t for t in trades if t["reason"] == r])))
    hits = sum(1 for t in trades if t["reason"] in ("target", "stop"))
    if hits == 0:
        L += ["", "**Not one bracket filled.** Every exit was a signal exit or the flatten, so the bracket "
              "settings aren't shaping results: the exit signal is."]

    L += ["", "## By hold time", "", HEAD]
    for lo, hi, label in HOLD_BUCKETS:
        L.append(_row(label, stats([t for t in trades if lo <= t["hold_min"] < hi])))
    L += ["", "Careful reading this one: losing trades are short *because* the reversal exit cut them, and "
          "winners are long because nothing cut them. It shows where the money goes, not that holding longer "
          "would fix it. Only the lab can test that (`live_hold`, `hold30`)."]

    L += ["", "## First trade vs re-entries (same ticker, same day)", "", HEAD,
          _row("first trade of the day", stats([t for t in trades if t["nth_today"] == 1])),
          _row("2nd trade", stats([t for t in trades if t["nth_today"] == 2])),
          _row("3rd or later", stats([t for t in trades if t["nth_today"] >= 3]))]

    L += ["", "## By entry hour", "", HEAD]
    for h in sorted({t["entry_time"].hour for t in trades}):
        L.append(_row(f"{h:02d}:00", stats([t for t in trades if t["entry_time"].hour == h])))

    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["sym"]].append(t)
    ranked = sorted(by_sym.items(), key=lambda kv: sum(t["pnl"] for t in kv[1]))
    L += ["", "## Tickers", "", "Worst: " + ", ".join(f"{k} {money(sum(t['pnl'] for t in v))} ({len(v)})" for k, v in ranked[:6]),
          "", "Best: " + ", ".join(f"{k} {money(sum(t['pnl'] for t in v))} ({len(v)})" for k, v in ranked[::-1][:6])]

    lim = [t for t in trades if t["entry_limit"]]
    if lim:
        imp = sum((t["entry_limit"] - t["entry"]) / t["entry_limit"] * 1e4 for t in lim) / len(lim)
        L += ["", "## Fills", "", f"* Entries filled on average {imp:+.1f} bps versus their limit price "
              "(positive = paid less than the limit)."]
    sized = [t["notional"] for t in trades]
    L += [f"* Position size: ${min(sized):,.2f} to ${max(sized):,.2f}, average ${sum(sized) / len(sized):,.2f}.", ""]

    L += ["## Worst 5 and best 5 trades", "", "| Ticker | Entry | Exit | Held | P&L | Ended by |", "|---|---|---|---:|---:|---|"]
    for t in sorted(trades, key=lambda t: t["pnl"])[:5] + sorted(trades, key=lambda t: -t["pnl"])[:5]:
        L.append(f"| {t['sym']} | {t['entry_time']:%m-%d %H:%M} @ {t['entry']:.2f} | {t['exit_time']:%H:%M} @ {t['exit']:.2f} | "
                 f"{t['hold_min']:.0f} min | {money(t['pnl'])} | {t['reason']} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    src = Path(argv[0])
    orders = load_orders(src)
    trades, orphans, still_open = round_trips(orders)
    if not trades:
        print(f"No complete round trips found in {src} ({len(orders)} orders read).")
        return 1
    out = Path("reviews")
    out.mkdir(exist_ok=True)
    tag = f"{trades[0]['day']}_{trades[-1]['day']}"
    text = report(trades, orphans, still_open, src.name)
    (out / f"review_{tag}.md").write_text(text)
    with open(out / f"trades_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        w.writeheader()
        w.writerows(trades)
    print(text)
    print(f"Saved: reviews/review_{tag}.md and reviews/trades_{tag}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
