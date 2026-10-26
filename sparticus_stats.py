"""Sparticus statistics: the numbers that say whether there's an edge, whether it's real, and
where money leaks. Shared by the live dashboard (from the trade journal) and the lab (from
simulated trades), so both report the same thing the same way. Pure Python, no I/O.

A trade is a dict with at least: pnl, notional, exit_time ("YYYY-MM-DD HH:MM").
Optional keys add sections: gross_pnl, entry_time, hold_min, reason, mfe_bps, mae_bps,
spread_bps, slippage_bps, sma_gap_bps, rsi, symbol.
"""
from __future__ import annotations

import math
import statistics as st
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def _f(t: dict, key: str) -> Optional[float]:
    v = t.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def verdict(n: int, t_stat: float, mean: float) -> str:
    if n < 30:
        return "too few trades to judge"
    if t_stat >= 2:
        return "likely a real edge"
    if t_stat <= -2:
        return "losing, and it's not bad luck"
    return "can't tell from luck yet" if mean > 0 else "no edge visible"


def summarize(trades: Sequence[dict]) -> Dict[str, object]:
    """Edge, confidence, leaks and risk for a list of closed trades."""
    pnl = [_f(t, "pnl") or 0.0 for t in trades]
    n = len(pnl)
    out: Dict[str, object] = {"trades": n}
    if n == 0:
        out.update(verdict="no trades yet")
        return out
    notional = sum(_f(t, "notional") or 0.0 for t in trades)
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    mean = sum(pnl) / n
    sd = st.stdev(pnl) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t_stat = mean / se if se > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = avg_win / -avg_loss if avg_loss < 0 else float("inf") if wins else 0.0
    gross = [_f(t, "gross_pnl") for t in trades]
    out.update(
        net_pnl=sum(pnl),
        gross_pnl=sum(g for g in gross if g is not None) if any(g is not None for g in gross) else None,
        expectancy=mean,
        expectancy_bps=sum(pnl) / notional * 1e4 if notional else 0.0,
        std=sd,
        t_stat=t_stat,
        ci95=(mean - 1.96 * se, mean + 1.96 * se),
        win_rate=len(wins) / n,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff=payoff,
        breakeven_win_rate=1 / (1 + payoff) if payoff not in (0.0, float("inf")) else None,
        profit_factor=sum(wins) / -sum(losses) if losses else float("inf") if wins else 0.0,
        verdict=verdict(n, t_stat, mean),
        # trades needed to prove the CURRENT average is real (t = 2), if it's positive
        trades_needed=math.ceil((2 * sd / mean) ** 2) if mean > 0 and sd > 0 else None,
    )

    # ---- leaks ----
    for key in ("spread_bps", "slippage_bps", "mfe_bps", "mae_bps", "hold_min"):
        vals = [v for v in (_f(t, key) for t in trades) if v is not None]
        out[f"avg_{key}"] = sum(vals) / len(vals) if vals else None
        out[f"median_{key}"] = st.median(vals) if vals else None
    kept = [(_f(t, "pnl") / (_f(t, "notional") or 1) * 1e4, _f(t, "mfe_bps")) for t in trades
            if _f(t, "mfe_bps") is not None]
    mfe_sum = sum(m for _, m in kept if m > 0)
    out["kept_of_best_move"] = (sum(p for p, m in kept if m > 0) / mfe_sum) if mfe_sum > 0 else None
    by_reason: Dict[str, List[float]] = defaultdict(list)
    for t, x in zip(trades, pnl):
        by_reason[str(t.get("reason") or "?")].append(x)
    out["exits"] = {r: (len(v), sum(v) / len(v)) for r, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))}

    # ---- risk ----
    order = sorted(range(n), key=lambda i: str(trades[i].get("exit_time", "")))
    eq = peak = dd = 0.0
    streak = longest = 0
    for i in order:
        eq += pnl[i]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        streak = streak + 1 if pnl[i] < 0 else 0
        longest = max(longest, streak)
    days: Dict[str, float] = defaultdict(float)
    for t, x in zip(trades, pnl):
        days[str(t.get("exit_time", ""))[:10]] += x
    out.update(max_drawdown=dd, longest_losing_streak=longest, days=len(days),
               worst_day=min(days.values()) if days else 0.0, best_day=max(days.values()) if days else 0.0,
               green_days=sum(1 for v in days.values() if v > 0) / len(days) if days else 0.0,
               trades_per_day=n / len(days) if days else 0.0)
    return out


def breakdown(trades: Sequence[dict], key: Callable[[dict], Optional[str]]) -> List[Tuple[str, int, float, float, float]]:
    """Rows of (bucket, trades, net $, $ per trade, win rate) grouped by key(trade)."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        k = key(t)
        if k is not None:
            groups[k].append(_f(t, "pnl") or 0.0)
    return [(k, len(v), sum(v), sum(v) / len(v), sum(1 for x in v if x > 0) / len(v))
            for k, v in sorted(groups.items())]


def by_hour(t: dict) -> Optional[str]:
    et = str(t.get("entry_time") or "")
    return f"{et[11:13]}:00" if len(et) >= 13 else None


def by_month(t: dict) -> Optional[str]:
    return str(t.get("exit_time", ""))[:7] or None


def by_hold(t: dict) -> Optional[str]:
    h = _f(t, "hold_min")
    if h is None:
        return None
    for edge, label in ((5, "a) under 5 min"), (15, "b) 5-15 min"), (30, "c) 15-30 min"), (60, "d) 30-60 min")):
        if h < edge:
            return label
    return "e) over 60 min"


def by_signal_strength(t: dict) -> Optional[str]:
    g = _f(t, "sma_gap_bps")
    if g is None:
        return None
    for edge, label in ((2, "a) gap under 2 bps"), (5, "b) 2-5 bps"), (10, "c) 5-10 bps")):
        if g < edge:
            return label
    return "d) over 10 bps"


def by_rsi(t: dict) -> Optional[str]:
    r = _f(t, "rsi")
    if r is None:
        return None
    lo = int(r // 10 * 10)
    return f"RSI {lo}-{lo + 10}"


def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{'-' if x < 0 else '+'}${abs(x):,.2f}" if abs(x) < 100 else f"{'-' if x < 0 else '+'}${abs(x):,.0f}"
