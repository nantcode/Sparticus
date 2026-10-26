"""Sparticus Lab (V6.3): replay months of history through the exact live strategy rules.

    pip install alpaca-py python-dotenv pandas numpy matplotlib
    python3 backtest.py                                   # last 6 full months, every experiment
    python3 backtest.py --start 2026-04-01 --end 2026-09-30 --experiments live,cap2
    python3 backtest.py --list                            # show the experiments

V6.2 adds strategy families (opening range breakout, SPY intraday momentum), volatility-scaled
brackets, a multiple-testing bar (the more ideas you test, the higher the bar), Monte Carlo
drawdowns, month-by-month consistency, a correlation table between ideas, and an expectations
file per config that the live bot uses for incubation and its kill switch.

V6.3 adds micro mode: experiments on a $100 and a $20 account with fractional shares, positions sized
as a share of equity, and a flat fee on every sale (--sell-fee, default $0.01).
    python3 backtest.py --experiments micro100,micro20
    python3 backtest.py --experiments micro100,micro20 --sell-fee 0 --out lab_results_nofee

What it does
  * Downloads 1-minute bars from Alpaca once and caches them in ./lab_data (read-only API use).
  * Replays every trading minute with the rules in sparticus_strategy.py: the same file the live
    bot imports, so the backtest can't silently drift from what the bot trades.
  * Charges realistic trading costs, judges every idea on months it was NOT tuned on
    (out-of-sample), and compares it with two baselines: random stock picks at the same times,
    and simply holding SPY.

What it can't do (stated so nobody over-trusts it)
  * No bid/ask quotes: the live 0.3% spread filter isn't applied; costs are a flat estimate.
  * Fills: entries and signal exits fill at the next bar's open; brackets fill at their price
    (a stop that gaps fills at the open); if a stop and a target are both touched in one bar,
    the stop is assumed to have hit first. Paper/live fills will differ somewhat.
  * Past results don't promise future ones. That's why the out-of-sample split exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sparticus_stats as S
from sparticus_strategy import (DEFAULT_UNIVERSE, LAB_STRATEGIES, LIVE_DEFAULTS, StrategyParams,
                                bracket_levels, compute_signal, config_id, min_bars, position_size)

ET = "America/New_York"
NS_PER_MIN = 60_000_000_000

# ---------------------------------------------------------------------------- experiments
# Each experiment changes ONE idea versus what the live bot trades today.
EXPERIMENTS: Dict[str, Tuple[StrategyParams, str]] = {
    "live": (LIVE_DEFAULTS, "exactly what LIVE_DEFAULTS says: what the live bot trades"),
    "cap2": (replace(LIVE_DEFAULTS, max_entries_per_symbol_per_day=2),
             "at most 2 entries per ticker per day"),
    "cooldown60": (replace(LIVE_DEFAULTS, reentry_cooldown_minutes=60),
                   "wait 60 min (not 15) before re-entering a ticker"),
    "bars5": (replace(LIVE_DEFAULTS, bar_minutes=5),
              "same rules on 5-minute bars instead of 1-minute"),
    "spy_trend": (replace(LIVE_DEFAULTS, market_filter="spy_trend"),
                  "only buy while SPY is above its 60-bar average"),
    "tight_brackets": (replace(LIVE_DEFAULTS, take_profit_pct=0.006, stop_loss_pct=0.004),
                       "TP +0.6% / SL -0.4%, sized to how far prices really move"),
    # V6.2: the style you've traded since Oct 2 (volatility-scaled 2:1, 0.4% floor, ~$7.3k positions)
    "atr_2to1": (replace(LIVE_DEFAULTS, bracket_mode="atr", reward_risk=2.0, min_stop_pct=0.004,
                         target_dollars_per_trade=7300.0),
                 "volatility-scaled 2:1 brackets, 0.4% minimum stop, $7.3k positions (your Oct 2 style)"),
    "atr_2to1_hold": (replace(LIVE_DEFAULTS, bracket_mode="atr", reward_risk=2.0, min_stop_pct=0.004,
                              target_dollars_per_trade=7300.0, signal_exit=False),
                      "same, but no signal exits: the bracket (or the 15:55 flatten) decides every exit"),
    # V6.3.1: Oct 8-9 review: 0 of 121 brackets filled, every exit was the 1-minute reversal signal
    "live_hold": (replace(LIVE_DEFAULTS, signal_exit=False),
                  "your current rules with no signal exits: the bracket or the 15:55 flatten ends every trade"),
    "hold30": (replace(LIVE_DEFAULTS, min_hold_minutes=30.0),
               "signal exits allowed only after 30 minutes; the bracket protects before that"),
    "hold5": (replace(LIVE_DEFAULTS, min_hold_minutes=5.0),
              "signal exits allowed after 5 minutes: the rule Sparticus traded until Oct 9"),
    # V6.2: strategy families from published research (lab-only until they pass)
    "orb": (replace(LIVE_DEFAULTS, strategy="orb", signal_exit=False, max_entries_per_symbol_per_day=1),
            "opening range breakout: 5-min range, first candle up, top 6 by relative volume, "
            "stop 10% of daily ATR, hold to the close (Zarattini, Barbon & Aziz)"),
    "spy_momentum": (replace(LIVE_DEFAULTS, strategy="spy_momentum", signal_exit=False),
                     "buy SPY at 15:30 when its first half hour was up, sell at the 15:55 flatten "
                     "(Gao, Han, Li & Zhou 2018)"),
}

# V6.3: micro mode on small accounts (fractional shares, 30% of equity per position, 3 positions)
MICRO = replace(LIVE_DEFAULTS, fractional=True, position_pct_of_equity=0.30, max_open_positions=3)
EXPERIMENTS["micro100"] = (MICRO, "micro mode on a $100 account: fractional shares, 30% of equity per position, "
                                  "3 positions, a penny fee on every sale")
EXPERIMENTS["micro20"] = (MICRO, "the same rules on a $20 account ($6 positions)")
LAB_EQUITY: Dict[str, float] = {"micro100": 100.0, "micro20": 20.0}   # default $100,000
EXPECT_SKIP = {"micro20"}   # same rules as micro100: its expectations file would overwrite micro100's

# min_t_stat is the floor; the real bar rises with the number of configs tested (see required_t).
SUCCESS_BAR = {"min_trades": 200, "min_profit_factor": 1.2, "random_percentile": 90, "min_t_stat": 2.0,
               "min_months_positive": 0.6}


# ---------------------------------------------------------------------------- data
def _client():
    import dotenv
    from alpaca.data.historical import StockHistoricalDataClient
    dotenv.load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY missing: run this from the bot's folder (where .env is).")
    return StockHistoricalDataClient(key, secret)


def _download(client, symbol: str, start: dt.datetime, end: dt.datetime, feed: str) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end,
                           feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX, adjustment=Adjustment.ALL)
    df = client.get_stock_bars(req).df
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_bars(symbols: List[str], start: dt.date, end: dt.date, feed: str, cache: Path) -> Dict[str, pd.DataFrame]:
    """All-session 1-minute bars per symbol, cached by month in cache/<feed>/<SYMBOL>/<YYYY-MM>.csv.gz."""
    client = None
    now = dt.datetime.now(dt.timezone.utc)
    out: Dict[str, pd.DataFrame] = {}
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    for i, sym in enumerate(symbols, 1):
        parts = []
        for m in months:
            path = cache / feed / sym / f"{m}.csv.gz"
            m_start = m.start_time.tz_localize(ET).tz_convert("UTC").to_pydatetime()
            m_end = (m.end_time + pd.Timedelta(seconds=1)).tz_localize(ET).tz_convert("UTC").to_pydatetime()
            complete = m_end < now - dt.timedelta(days=1)
            if path.exists() and complete:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                client = client or _client()
                q_end = min(m_end, now - dt.timedelta(minutes=20))  # free plans can't query the last 15 min of SIP
                if q_end <= m_start:
                    continue
                print(f"  downloading {sym} {m} ({i}/{len(symbols)})", flush=True)
                df = _download(client, sym, m_start, q_end, feed)
                time.sleep(0.3)  # stay well under the data API rate limit
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, compression="gzip")
            parts.append(df)
        if parts:
            df = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else parts[0]
            if not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                df = df[~df.index.duplicated()].sort_index()
                lo = pd.Timestamp(start).tz_localize(ET) - pd.Timedelta(days=1)
                hi = pd.Timestamp(end).tz_localize(ET) + pd.Timedelta(days=1)
                out[sym] = df[(df.index >= lo) & (df.index < hi)]
    return out


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1 or df.empty:
        return df
    agg = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg.dropna(subset=["close"])


# ---------------------------------------------------------------------------- simulation
@dataclass
class Position:
    qty: float                  # fractional in micro mode
    entry_px: float
    entry_ns: int
    entry_step: int
    tp: float
    sl: float
    cost_paid: float
    hi: float = 0.0
    lo: float = 0.0
    gap_bps: Optional[float] = None
    rsi: Optional[float] = None
    intrabar: bool = False      # filled by a resting stop order inside this bar (ORB)


@dataclass
class Trade:
    symbol: str
    day: str
    entry_time: str
    exit_time: str
    qty: float                  # fractional in micro mode
    entry_px: float
    exit_px: float
    pnl: float
    gross_pnl: float
    pnl_pct: float
    reason: str
    hold_min: float
    entry_step: int
    exit_step: int
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    sma_gap_bps: Optional[float] = None
    rsi: Optional[float] = None

    def as_stats(self) -> dict:
        d = asdict(self)
        d["notional"] = self.entry_px * self.qty
        return d


@dataclass
class Result:
    name: str
    params: StrategyParams
    trades: List[Trade]
    daily: pd.DataFrame          # day, equity_close, pnl
    max_dd_by_day: Dict[str, float]
    timeline_days: List[str]
    sym_index: Dict[str, int] = field(default_factory=dict)
    arrays: Dict[str, Tuple[np.ndarray, ...]] = field(default_factory=dict)
    mats: Dict[str, np.ndarray] = field(default_factory=dict)


def build_timeline(bars: Dict[str, pd.DataFrame], start: dt.date, end: dt.date, bar_min: int):
    """Decision steps: every regular-hours bar start (09:30 .. 16:00 - bar) on days the market traded."""
    ref = bars.get("SPY")
    if ref is None or ref.empty:
        ref = max(bars.values(), key=len)
    et_idx = ref.index.tz_convert(ET)
    days = sorted({d for d in et_idx.date if start <= d <= end})
    starts, day_of = [], []
    for d in days:
        open_ = pd.Timestamp(dt.datetime.combine(d, dt.time(9, 30)), tz=ET)
        # half-days: use the last regular-hours bar actually present for that day
        day_bars = et_idx[(et_idx.date == d)]
        rth = day_bars[(day_bars >= open_) & (day_bars < open_ + pd.Timedelta(hours=6, minutes=30))]
        if len(rth) == 0:
            continue
        close_ = min(open_ + pd.Timedelta(hours=6, minutes=30), rth.max() + pd.Timedelta(minutes=1))
        t = open_
        while t + pd.Timedelta(minutes=bar_min) <= close_:
            starts.append(t.tz_convert("UTC").value)
            day_of.append(d.isoformat())
            t += pd.Timedelta(minutes=bar_min)
        starts.append(None)  # day separator marker (removed below)
        day_of.append(None)
    steps = np.array([s for s in starts if s is not None], dtype=np.int64)
    day_arr = [d for d in day_of if d is not None]
    # each day's real open and close (handles half-days)
    closes_ns: Dict[str, int] = {}
    opens_ns: Dict[str, int] = {}
    for s, d in zip(steps, day_arr):
        closes_ns[d] = max(closes_ns.get(d, 0), int(s) + bar_min * NS_PER_MIN)
        opens_ns[d] = min(opens_ns.get(d, int(s)), int(s))
    return steps, day_arr, closes_ns, opens_ns


def precompute(bars: Dict[str, pd.DataFrame], symbols: List[str], steps: np.ndarray, p: StrategyParams):
    """Vectorised pre-filter per symbol x step. Exact for entries because every candidate is then
    confirmed with compute_signal() on the same window the live scanner sees."""
    bar_ns = p.bar_minutes * NS_PER_MIN
    look_ns = p.lookback_minutes * p.bar_minutes * NS_PER_MIN
    need = min_bars(p)
    n_steps, n_sym = len(steps), len(symbols)
    li_m = np.full((n_steps, n_sym), -1, dtype=np.int32)
    wi_m = np.zeros((n_steps, n_sym), dtype=np.int32)
    at_m = np.full((n_steps, n_sym), -1, dtype=np.int32)
    cand_m = np.zeros((n_steps, n_sym), dtype=bool)
    exit_long_m = np.zeros((n_steps, n_sym), dtype=bool)
    atr_m = np.full((n_steps, n_sym), np.nan)
    arrays: Dict[str, Tuple[np.ndarray, ...]] = {}
    decision = steps + bar_ns
    for j, sym in enumerate(symbols):
        df = bars.get(sym)
        if df is None or df.empty:
            arrays[sym] = tuple(np.zeros(0) for _ in range(5))
            continue
        df = resample(df, p.bar_minutes)
        ts = np.asarray(df.index.as_unit("ns").asi8, dtype=np.int64)
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        arrays[sym] = (ts, o, h, l, c)
        cs = np.concatenate([[0.0], np.cumsum(c)])

        def rolling(n):
            r = np.full(len(c), np.nan)
            if len(c) >= n:
                r[n - 1:] = (cs[n:] - cs[:-n]) / n
            return r
        fast, slow = rolling(p.fast_sma), rolling(p.slow_sma)
        # ATR = plain mean of the last atr_period true ranges (same as sparticus_strategy.atr)
        prev_c = np.concatenate([[np.nan], c[:-1]])
        tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
        tr[0] = np.nan
        tr_cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(tr))])
        atr_arr = np.full(len(c), np.nan)
        n_atr = p.atr_period
        if len(c) > n_atr:
            atr_arr[n_atr:] = (tr_cs[n_atr + 1:] - tr_cs[1:-n_atr]) / n_atr
        up = fast > slow
        down = fast < slow
        crossed = np.zeros(len(c), dtype=bool)
        for k in range(1, p.cross_lookback + 1):
            prev = np.zeros(len(c), dtype=bool)
            prev[k:] = fast[:-k] <= slow[:-k]
            crossed |= prev
        li = np.searchsorted(ts, steps, side="right") - 1
        wi = np.searchsorted(ts, decision - look_ns, side="left")
        have = li >= 0
        li_c = np.where(have, li, 0)
        age_ok = have & ((decision - (ts[li_c] + bar_ns)) <= p.max_bar_age_minutes * NS_PER_MIN)
        enough = have & ((li - wi + 1) >= need)
        li_m[:, j], wi_m[:, j] = li, wi
        at_m[:, j] = np.where(have & (ts[li_c] == steps), li, -1)
        cand_m[:, j] = age_ok & enough & up[li_c] & (crossed[li_c] | (not p.require_fresh_cross))
        exit_long_m[:, j] = age_ok & enough & down[li_c]
        atr_m[:, j] = np.where(have, atr_arr[li_c], np.nan)
    mats = {"li": li_m, "wi": wi_m, "at": at_m, "cand": cand_m, "exit_long": exit_long_m, "atr": atr_m}
    if p.market_filter == "spy_trend" and "SPY" in symbols:
        ts, o, h, l, c = arrays["SPY"]
        cs = np.concatenate([[0.0], np.cumsum(c)])
        n = p.market_filter_sma
        trend = np.full(len(c), np.nan)
        if len(c) >= n:
            trend[n - 1:] = (cs[n:] - cs[:-n]) / n
        li = mats["li"][:, symbols.index("SPY")]
        ok = (li >= 0) & (c[np.maximum(li, 0)] > trend[np.maximum(li, 0)])
        mats["market_ok"] = ok
    else:
        mats["market_ok"] = np.ones(n_steps, dtype=bool)
    return arrays, mats


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    """Regular-hours 1-minute bars with ET date/time columns."""
    et = df.index.tz_convert(ET)
    keep = (et.time >= dt.time(9, 30)) & (et.time < dt.time(16, 0))
    out = df[keep].copy()
    et_k = et[keep]
    out["d"] = et_k.date
    out["t"] = et_k.time
    return out


def orb_plan(bars: Dict[str, pd.DataFrame], symbols: List[str], p: StrategyParams) -> Dict[str, List[Tuple[str, float, float, float]]]:
    """Per day: the tickers armed for an opening range breakout, as (symbol, range high, stop distance,
    relative volume), best "in play" first. Only uses information available at the end of the range."""
    or_end = (dt.datetime.combine(dt.date(2000, 1, 1), dt.time(9, 30)) + dt.timedelta(minutes=p.orb_minutes)).time()
    cands: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for sym in symbols:
        df = bars.get(sym)
        if df is None or df.empty:
            continue
        r = _rth(df)
        if r.empty:
            continue
        g = r.groupby("d")
        daily = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
        prev_close = daily["close"].shift(1)
        tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_close).abs(),
                        (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
        datr = tr.rolling(p.orb_atr_days).mean().shift(1)          # prior days only
        w = r[r["t"] < or_end].groupby("d")
        rng = pd.DataFrame({"open": w["open"].first(), "high": w["high"].max(), "close": w["close"].last(),
                            "vol": w["volume"].sum()})
        rel = rng["vol"] / rng["vol"].rolling(p.orb_atr_days).mean().shift(1)
        for d in rng.index:
            a, rv = datr.get(d), rel.get(d)
            if a is None or rv is None or not np.isfinite(a) or not np.isfinite(rv):
                continue
            row = rng.loc[d]
            if row["close"] > row["open"] and rv >= p.orb_min_rel_volume:   # long-only: first candle up
                cands.setdefault(d.isoformat(), []).append((sym, float(row["high"]), float(a) * p.orb_stop_atr_frac, float(rv)))
    return {d: sorted(v, key=lambda x: -x[3])[:p.orb_top_n] for d, v in cands.items()}


def momentum_plan(bars: Dict[str, pd.DataFrame], p: StrategyParams) -> Dict[str, float]:
    """Per day: the first half-hour return of mom_symbol (previous close -> 10:00 ET)."""
    df = bars.get(p.mom_symbol)
    if df is None or df.empty:
        return {}
    r = _rth(df)
    g = r.groupby("d")
    closes = g["close"].last()
    at10 = r[r["t"] < dt.time(10, 0)].groupby("d")["close"].last()
    prev = closes.shift(1)
    out = {}
    for d in at10.index:
        pc = prev.get(d)
        if pc is not None and np.isfinite(pc) and pc > 0:
            out[d.isoformat()] = float(at10[d] / pc - 1)
    return out


def simulate(name: str, p: StrategyParams, bars: Dict[str, pd.DataFrame], symbols: List[str],
             start: dt.date, end: dt.date, cost_bps: float, start_equity: float = 100_000.0,
             verbose: bool = True, sell_fee: float = 0.0) -> Result:
    """sell_fee: flat $ per sale (regulatory fees round up to at least a penny per sell)."""
    t0 = time.time()
    steps, day_of, close_ns, open_ns = build_timeline(bars, start, end, p.bar_minutes)
    arrays, M = precompute(bars, symbols, steps, p)
    li_m, wi_m, at_m, cand_m, exitl_m, mkt_ok, atr_m = (M[k] for k in ("li", "wi", "at", "cand", "exit_long",
                                                                       "market_ok", "atr"))
    if p.strategy not in LAB_STRATEGIES:
        raise ValueError(f"unknown strategy {p.strategy!r}")
    orb = orb_plan(bars, symbols, p) if p.strategy == "orb" else {}
    mom = momentum_plan(bars, p) if p.strategy == "spy_momentum" else {}
    orb_end_ns_off = p.orb_minutes * NS_PER_MIN
    orb_armed: Dict[str, Tuple[float, float]] = {}     # today's armed ORB tickers: sym -> (range high, stop dist)
    bar_ns = p.bar_minutes * NS_PER_MIN
    look_ns = p.lookback_minutes * p.bar_minutes * NS_PER_MIN
    cost = cost_bps / 10_000.0
    sym_idx = {s: j for j, s in enumerate(symbols)}

    cash = start_equity
    positions: Dict[str, Position] = {}
    pending_entries: List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]] = []
    # (symbol, decision close, SMA gap bps, RSI, ATR at decision)
    pending_exits: List[Tuple[str, str]] = []          # (symbol, reason)
    cooldown_until: Dict[str, int] = {}
    entries_today: Dict[str, int] = {}
    trades: List[Trade] = []
    daily_rows, dd_by_day = [], {}
    cur_day, sod_equity, halted = None, start_equity, False
    peak, day_min_dd = start_equity, 0.0

    def fmt(ns: int) -> str:
        return pd.Timestamp(ns, tz="UTC").tz_convert(ET).strftime("%Y-%m-%d %H:%M")

    def price_now(sym: str, step: int) -> float:
        li = li_m[step, sym_idx[sym]]
        return float(arrays[sym][4][li]) if li >= 0 else positions[sym].entry_px

    def close_pos(sym: str, px: float, step: int, ns: int, reason: str, pay_cost: bool = True):
        nonlocal cash
        pos = positions.pop(sym)
        fill = px * (1 - cost) if pay_cost else px
        exit_cost = px * cost * pos.qty if pay_cost else 0.0
        cash += pos.qty * fill - sell_fee
        pnl = (fill - pos.entry_px) * pos.qty - sell_fee
        gross = pnl + pos.cost_paid + exit_cost + sell_fee
        hi, lo = max(pos.hi, px), min(pos.lo, px)
        trades.append(Trade(sym, day_of[step], fmt(pos.entry_ns), fmt(ns), pos.qty, round(pos.entry_px, 4),
                            round(fill, 4), round(pnl, 4), round(gross, 4), round(fill / pos.entry_px - 1, 6),
                            reason, round((ns - pos.entry_ns) / NS_PER_MIN, 1), pos.entry_step, step,
                            round((hi / pos.entry_px - 1) * 1e4, 1), round((lo / pos.entry_px - 1) * 1e4, 1),
                            pos.gap_bps, pos.rsi))
        cooldown_until[sym] = ns + int(p.reentry_cooldown_minutes * NS_PER_MIN)

    for step in range(len(steps)):
        t_ns = int(steps[step])
        decision_ns = t_ns + bar_ns
        day = day_of[step]
        if day != cur_day:
            # new trading day: anything still open is force-closed at the previous close
            # (live flattens at 15:55; this only matters on half-days or data gaps)
            for sym in list(positions):
                close_pos(sym, price_now(sym, max(step - 1, 0)), max(step - 1, 0), t_ns, "carry")
            if cur_day is not None:
                eq = cash
                daily_rows.append((cur_day, eq, eq - sod_equity))
                dd_by_day[cur_day] = day_min_dd
            cur_day, sod_equity, halted, day_min_dd = day, cash, False, 0.0
            entries_today = {}
            pending_entries, pending_exits = [], []
            orb_armed = {s_: (hi_, dist_) for s_, hi_, dist_, _ in orb.get(day, []) if s_ in sym_idx}

        # 1) orders decided last bar fill at this bar's open
        for sym, reason in pending_exits:
            if sym in positions:
                j = sym_idx[sym]
                k = at_m[step, j]
                px = float(arrays[sym][1][k]) if k >= 0 else price_now(sym, step)
                close_pos(sym, px, step, t_ns, reason)
        pending_exits = []
        for sym, ref_px, gap_bps, rsi_v, atr_v in pending_entries:
            j = sym_idx[sym]
            k = at_m[step, j]
            if k < 0 or sym in positions:
                continue  # no trade printed: the marketable limit wouldn't have filled
            open_px = float(arrays[sym][1][k])
            qty, _why = position_size(ref_px, sod_equity, p)
            if qty <= 0 or qty * ref_px > cash:
                continue
            fill = open_px * (1 + cost)
            tp, sl = bracket_levels(ref_px, True, p, atr_v)
            sl = max(sl, round(ref_px * (1 + p.hard_stop_pct), 2))
            cash -= qty * fill
            positions[sym] = Position(qty, fill, t_ns, step, tp, sl, open_px * cost * qty,
                                      hi=open_px, lo=open_px, gap_bps=gap_bps, rsi=rsi_v)
            entries_today[sym] = entries_today.get(sym, 0) + 1
        pending_entries = []

        # 1b) ORB: resting buy-stop orders at the range high, live once the range is complete
        if orb_armed and t_ns >= open_ns[day] + orb_end_ns_off and not halted \
                and (close_ns[day] - t_ns) / NS_PER_MIN > p.no_entry_last_minutes:
            for sym in list(orb_armed):
                if len(positions) >= p.max_open_positions:
                    break
                j = sym_idx[sym]
                k = at_m[step, j]
                if k < 0:
                    continue
                _, o, h, l, c = arrays[sym]
                orh, dist = orb_armed[sym]
                if h[k] < orh:
                    continue
                orb_armed.pop(sym)                    # one shot per ticker per day
                px = max(float(o[k]), orh)            # a gap above the range fills at the open
                qty, _why = position_size(px, sod_equity, p)
                if qty <= 0 or dist <= 0 or qty * px > cash:
                    continue
                fill = px * (1 + cost)
                sl = round(px - dist, 2)
                tp = round(px + p.orb_reward_risk * dist, 2) if p.orb_reward_risk > 0 else float("inf")
                cash -= qty * fill
                positions[sym] = Position(qty, fill, t_ns, step, tp, sl, px * cost * qty, hi=px, lo=px,
                                          intrabar=float(o[k]) < orh)
                entries_today[sym] = entries_today.get(sym, 0) + 1

        # 2) brackets live at the broker: check this bar's range (stop first if both touched)
        for sym in list(positions):
            j = sym_idx[sym]
            k = at_m[step, j]
            if k < 0:
                continue
            _, o, h, l, c = arrays[sym]
            pos = positions[sym]
            if pos.intrabar and pos.entry_step == step:
                # Filled mid-bar: the bar's low may have come before the fill, so only count the stop
                # if the bar closed below it. (An approximation; see the module docstring.)
                pos.hi = max(pos.hi, float(h[k]))
                if c[k] <= pos.sl:
                    close_pos(sym, pos.sl, step, t_ns, "stop")
                continue
            pos.hi, pos.lo = max(pos.hi, float(h[k])), min(pos.lo, float(l[k]))
            if o[k] <= pos.sl:
                close_pos(sym, float(o[k]), step, t_ns, "stop")
            elif l[k] <= pos.sl:
                close_pos(sym, pos.sl, step, t_ns, "stop")
            elif o[k] >= pos.tp:
                close_pos(sym, float(o[k]), step, t_ns, "target", pay_cost=False)
            elif h[k] >= pos.tp:
                close_pos(sym, pos.tp, step, t_ns, "target", pay_cost=False)

        # 3) decision at the end of the bar
        equity = cash + sum(pos.qty * price_now(s, step) for s, pos in positions.items())
        peak = max(peak, equity)
        day_min_dd = min(day_min_dd, equity / peak - 1)
        mins_to_close = (close_ns[day] - decision_ns) / NS_PER_MIN
        mins_since_open = (decision_ns - open_ns[day]) / NS_PER_MIN

        if not halted and equity - sod_equity <= p.daily_loss_limit_pct * sod_equity:
            halted = True
            for sym in list(positions):
                close_pos(sym, price_now(sym, step), step, decision_ns, "breaker")
        if mins_to_close <= p.eod_flatten_minutes:
            for sym in list(positions):
                close_pos(sym, price_now(sym, step), step, decision_ns, "eod")
            continue

        if p.signal_exit and p.strategy == "sma_cross":
            for sym, pos in positions.items():
                if exitl_m[step, sym_idx[sym]] and decision_ns - pos.entry_ns >= p.min_hold_minutes * NS_PER_MIN:
                    pending_exits.append((sym, "signal"))

        if halted or mins_since_open < p.no_entry_first_minutes or mins_to_close <= p.no_entry_last_minutes:
            continue
        if not mkt_ok[step]:
            continue
        slots = p.max_open_positions - len(positions)
        if slots <= 0:
            continue
        if p.strategy == "spy_momentum":
            sym = p.mom_symbol
            if (sym in sym_idx and sym not in positions and day in mom
                    and abs(mins_to_close - p.mom_entry_minutes_before_close) < p.bar_minutes / 2
                    and mom[day] > p.mom_min_first_half_hour_ret):
                li = int(li_m[step, sym_idx[sym]])
                if li >= 0:
                    pending_entries.append((sym, float(arrays[sym][4][li]), None, None, None))
            continue
        if p.strategy != "sma_cross":
            continue
        for j in np.flatnonzero(cand_m[step]):
            if slots <= 0:
                break
            sym = symbols[j]
            if sym in positions or cooldown_until.get(sym, 0) > decision_ns:
                continue
            if p.max_entries_per_symbol_per_day and entries_today.get(sym, 0) >= p.max_entries_per_symbol_per_day:
                continue
            ts, o, h, l, c = arrays[sym]
            li, wi = int(li_m[step, j]), int(wi_m[step, j])
            sig = compute_signal(c[wi:li + 1].tolist(), p)   # the live rule, on the live window
            if sig is None or not sig.long_entry:
                continue
            atr_v = float(atr_m[step, j]) if np.isfinite(atr_m[step, j]) else None
            pending_entries.append((sym, float(c[li]), (sig.fast - sig.slow) / sig.slow * 1e4, sig.rsi, atr_v))
            slots -= 1

    if cur_day is not None:
        last = len(steps) - 1
        for sym in list(positions):
            close_pos(sym, price_now(sym, last), last, int(steps[last]), "carry")
        daily_rows.append((cur_day, cash, cash - sod_equity))
        dd_by_day[cur_day] = day_min_dd
    daily = pd.DataFrame(daily_rows, columns=["day", "equity", "pnl"])
    if verbose:
        print(f"  {name}: {len(trades)} trades over {len(daily)} days in {time.time() - t0:.0f}s", flush=True)
    return Result(name, p, trades, daily, dd_by_day, sorted(set(day_of)), sym_idx, arrays,
                  {"at": at_m, "li": li_m})


# ---------------------------------------------------------------------------- evaluation
def metrics(trades: List[Trade], daily: pd.DataFrame, start_equity: float) -> Dict[str, float]:
    n = len(trades)
    pnl = np.array([t.pnl for t in trades]) if n else np.zeros(0)
    gross = np.array([t.gross_pnl for t in trades]) if n else np.zeros(0)
    notional = np.array([t.entry_px * t.qty for t in trades]) if n else np.zeros(0)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    days = max(len(daily), 1)
    eq = start_equity + daily["pnl"].cumsum() if len(daily) else pd.Series([start_equity])
    dd = float((eq / eq.cummax() - 1).min()) if len(daily) else 0.0
    dret = daily["pnl"] / start_equity if len(daily) else pd.Series(dtype=float)
    sharpe = float(dret.mean() / dret.std() * math.sqrt(252)) if len(dret) > 1 and dret.std() > 0 else 0.0
    reasons = pd.Series([t.reason for t in trades]).value_counts(normalize=True).to_dict() if n else {}
    return {
        "trades": n,
        "trades_per_day": n / days,
        "net_pnl": float(pnl.sum()),
        "gross_pnl": float(gross.sum()),
        "win_rate": float(len(wins) / n) if n else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        "expectancy": float(pnl.mean()) if n else 0.0,
        "expectancy_bps": float(pnl.sum() / notional.sum() * 10_000) if n and notional.sum() else 0.0,
        "return_pct": float(pnl.sum() / start_equity * 100),
        "max_drawdown_pct": dd * 100,
        "sharpe": sharpe,
        "green_days_pct": float((daily["pnl"] > 0).mean() * 100) if len(daily) else 0.0,
        "median_hold_min": float(np.median([t.hold_min for t in trades])) if n else 0.0,
        **{f"exit_{k}": float(v) for k, v in reasons.items()},
    }


def random_baseline(res: Result, trades: List[Trade], symbols: List[str], cost_bps: float,
                    seeds: int, target: float = 0.0, sell_fee: float = 0.0) -> np.ndarray:
    """Same entry and exit TIMES as the strategy's trades, but a random other ticker each time.
    Answers: does Sparticus pick better stocks than chance, after the same costs?"""
    cost = cost_bps / 10_000.0
    at, li = res.mats["at"], res.mats["li"]
    totals = []
    for seed in range(seeds):
        rng = random.Random(seed)
        total = 0.0
        for t in trades:
            for _ in range(10):
                r = rng.choice(symbols)
                if r == t.symbol:
                    continue
                j = res.sym_index[r]
                ke, kx = at[t.entry_step, j], li[t.exit_step, j]
                if ke < 0 or kx < 0:
                    continue
                o, c = res.arrays[r][1], res.arrays[r][4]
                entry, exit_ = o[ke] * (1 + cost), c[kx] * (1 - cost)
                qty = (t.entry_px * t.qty) / o[ke]          # same dollars as the real trade
                total += (exit_ - entry) * qty - sell_fee
                break
        totals.append(total)
    return np.array(totals)


def spy_buy_hold(bars: Dict[str, pd.DataFrame], first_day: str, last_day: str) -> Optional[float]:
    df = bars.get("SPY")
    if df is None or df.empty:
        return None
    et = df.index.tz_convert(ET)
    rth = df[(et.time >= dt.time(9, 30)) & (et.time < dt.time(16, 0))]
    et_r = rth.index.tz_convert(ET)
    seg = rth[(et_r.date >= dt.date.fromisoformat(first_day)) & (et_r.date <= dt.date.fromisoformat(last_day))]
    if seg.empty:
        return None
    return float(seg["close"].iloc[-1] / seg["open"].iloc[0] - 1) * 100


def required_t(n_configs: int) -> float:
    """The t-stat bar after testing n_configs ideas (Bonferroni, two-sided 5%), never below 2.
    Test 20 ideas and one will look great by luck; the bar rises so luck can't pass."""
    n = max(1, int(n_configs))
    return max(SUCCESS_BAR["min_t_stat"], NormalDist().inv_cdf(1 - 0.025 / n))


def update_ledger(out: Path, rows: List[Tuple[str, str]]) -> int:
    """Append (name, config_id) rows to the lab's ledger and return how many distinct configs have
    ever been tested. The ledger is how the multiple-testing bar remembers past runs."""
    path = out / "ledger.csv"
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                seen.add(parts[2])
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("date,name,config_id\n")
        for name, cid in rows:
            f.write(f"{dt.date.today().isoformat()},{name},{cid}\n")
            seen.add(cid)
    return len(seen)


def monte_carlo_dd(pnl: List[float], runs: int = 2000, seed: int = 7) -> Tuple[float, float]:
    """Reshuffle the order of the trades many times. Returns (median, 95th-percentile-worst) max
    drawdown in $. The live kill switch uses the 95% figure: worse than that = not bad luck."""
    if len(pnl) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(pnl, dtype=float)
    dds = np.empty(runs)
    for i in range(runs):
        eq = np.cumsum(rng.permutation(arr))
        dds[i] = float(np.min(eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]))
    return float(np.median(dds)), float(np.percentile(dds, 5))


def months_positive(trades: List["Trade"]) -> Tuple[float, int]:
    """Share of calendar months that made money (walk-forward style consistency)."""
    by: Dict[str, float] = {}
    for t in trades:
        by[t.day[:7]] = by.get(t.day[:7], 0.0) + t.pnl
    if not by:
        return 0.0, 0
    return sum(1 for v in by.values() if v > 0) / len(by), len(by)


def judge(m: Dict[str, float], rand: np.ndarray, t_stat: float = 0.0, t_bar: Optional[float] = None,
          months_pos: Optional[Tuple[float, int]] = None) -> Dict[str, Tuple[bool, str]]:
    pct = float((rand < m["net_pnl"]).mean() * 100) if len(rand) else 0.0
    t_bar = SUCCESS_BAR["min_t_stat"] if t_bar is None else t_bar
    out = {
        f"t-stat >= {t_bar:.2f} (not luck, after counting every idea tested)": (t_stat >= t_bar, f"{t_stat:+.2f}"),
        f"at least {SUCCESS_BAR['min_trades']} trades": (m["trades"] >= SUCCESS_BAR["min_trades"], f"{m['trades']}"),
        f"profit factor >= {SUCCESS_BAR['min_profit_factor']}": (
            m["profit_factor"] >= SUCCESS_BAR["min_profit_factor"], f"{m['profit_factor']:.2f}"),
        f"beats {SUCCESS_BAR['random_percentile']}% of random picks": (
            pct >= SUCCESS_BAR["random_percentile"], f"beats {pct:.0f}%"),
        "makes money after costs": (m["net_pnl"] > 0, f"${m['net_pnl']:,.0f}"),
    }
    if months_pos is not None:
        share, n_m = months_pos
        need = SUCCESS_BAR["min_months_positive"]
        out[f"profitable in at least {need:.0%} of months"] = (share >= need, f"{share:.0%} of {n_m} months")
    return out


# ---------------------------------------------------------------------------- report
def fmt_money(x: float) -> str:
    """Whole dollars, but cents under $100 (micro accounts trade in cents)."""
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}" if abs(x) < 100 else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _num(x, f="{:.1f}") -> str:
    return "-" if x is None else f.format(x)


def stats_block(title: str, s: Dict[str, object]) -> List[str]:
    """The statistics pack (edge, confidence, leaks, risk) as a markdown table."""
    ci = s.get("ci95") or (0.0, 0.0)
    rows = [
        ("Trades", f"{s['trades']}"),
        ("Verdict", f"**{s.get('verdict')}**"),
        ("Net P&L / before costs", f"{fmt_money(s.get('net_pnl', 0))} / {fmt_money(s['gross_pnl']) if s.get('gross_pnl') is not None else '-'}"),
        ("Per trade", f"{S.fmt_money(s.get('expectancy'))} ({_num(s.get('expectancy_bps'), '{:+.1f}')} bps)"),
        ("95% range for per-trade result", f"{S.fmt_money(ci[0])} to {S.fmt_money(ci[1])}"),
        ("t-stat (above 2 = probably not luck)", _num(s.get("t_stat"), "{:+.2f}")),
        ("Trades needed to prove the current average", f"{s['trades_needed']:,}" if s.get("trades_needed") else "-"),
        ("Win rate vs breakeven win rate", f"{_pct(s.get('win_rate'))} vs {_pct(s.get('breakeven_win_rate'))}"),
        ("Payoff (avg win / avg loss)", f"{S.fmt_money(s.get('avg_win'))} / {S.fmt_money(s.get('avg_loss'))} = {_num(s.get('payoff'), '{:.2f}')}"),
        ("Profit factor", _num(s.get("profit_factor"), "{:.2f}")),
        ("Best move while open (avg MFE)", f"{_num(s.get('avg_mfe_bps'), '{:+.0f}')} bps"),
        ("Worst move while open (avg MAE)", f"{_num(s.get('avg_mae_bps'), '{:+.0f}')} bps"),
        ("Share of the best move kept", _pct(s.get("kept_of_best_move"))),
        ("Median hold", f"{_num(s.get('median_hold_min'), '{:.0f}')} min"),
        ("Max drawdown / worst day", f"{fmt_money(s.get('max_drawdown', 0))} / {fmt_money(s.get('worst_day', 0))}"),
        ("Longest losing streak", f"{s.get('longest_losing_streak', 0)} trades"),
        ("Trades per day / green days", f"{_num(s.get('trades_per_day'))} / {_pct(s.get('green_days'))}"),
    ]
    out = [f"**{title}**", "", "| Statistic | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    exits = s.get("exits") or {}
    if exits:
        out += ["", "| How trades ended | Trades | Avg $ |", "|---|---:|---:|"]
        out += [f"| {r} | {n} | {S.fmt_money(a)} |" for r, (n, a) in exits.items()]
    return out + [""]


def breakdown_md(title: str, rows) -> List[str]:
    if not rows:
        return []
    out = [f"**{title}**", "", "| Bucket | Trades | Net | Per trade | Win rate |", "|---|---:|---:|---:|---:|"]
    out += [f"| {k} | {n} | {fmt_money(net)} | {S.fmt_money(avg)} | {wr * 100:.0f}% |" for k, n, net, avg, wr in rows]
    return out + [""]


def write_detail(out: Path, res: "Result", split_day: str) -> None:
    all_t = [t.as_stats() for t in res.trades]
    oos = [t for t in all_t if t["day"] >= split_day]
    lines = [f"# `{res.name}` in detail", "",
             "Breakdowns use **all** trades so the buckets have enough data; treat them as clues to test, "
             "not conclusions. A bucket that looks great is the easiest thing to overfit.", ""]
    lines += stats_block("Out-of-sample", S.summarize(oos))
    lines += stats_block("Whole period", S.summarize(all_t))
    lines += breakdown_md("By month", S.breakdown(all_t, S.by_month))
    lines += breakdown_md("By hour of entry", S.breakdown(all_t, S.by_hour))
    lines += breakdown_md("By hold time", S.breakdown(all_t, S.by_hold))
    lines += breakdown_md("By signal strength at entry (5-SMA minus 20-SMA)", S.breakdown(all_t, S.by_signal_strength))
    lines += breakdown_md("By RSI at entry", S.breakdown(all_t, S.by_rsi))
    sym = S.breakdown(all_t, lambda t: t["symbol"])
    sym.sort(key=lambda r: r[2])
    lines += breakdown_md("Tickers: 5 worst", sym[:5]) + breakdown_md("Tickers: 5 best", sym[-5:][::-1])
    (out / f"detail_{res.name}.md").write_text("\n".join(lines) + "\n")


def write_report(out: Path, results, args, split_day: str, spy_is: Optional[float], spy_oos: Optional[float],
                 sweep_rows: Optional[List[Tuple]] = None, extras: Optional[Dict[str, object]] = None) -> str:
    extras = extras or {}
    lines = ["# Sparticus Lab results", "",
             f"Period {args.start} to {args.end}, {len(args.symbols)} tickers, {args.feed.upper()} 1-minute bars, "
             f"costs {args.cost_bps:g} bps per side plus ${getattr(args, 'sell_fee', 0.01):.2f} per sale, $100k start "
             f"and $5k per trade unless the experiment says otherwise.",
             f"In-sample (tuning) = days before {split_day}. **Out-of-sample (the verdict) = {split_day} onward.**",
             f"SPY buy-and-hold: in-sample {spy_is:+.2f}%, out-of-sample {spy_oos:+.2f}%." if spy_is is not None and spy_oos is not None else "",
             "",
             f"**The bar this run: t-stat ≥ {extras.get('t_bar', 2.0):.2f}**, because the ledger shows "
             f"{extras.get('n_configs', 1)} different configs tested so far. Every new idea raises it a little: "
             "that's what stops one lucky result out of many from looking like an edge.",
             "", "## Out-of-sample scoreboard", "",
             "| Experiment | Config | Trades | Net P&L | Before costs | Per trade | t-stat | Profit factor | Win / breakeven | "
             "Months + | Worst DD (95%) | Beats random | Bar |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for res, m_is, m_oos, rand, verdict, s_oos in results:
        passed = all(ok for ok, _ in verdict.values())
        rand_pct = float((rand < m_oos["net_pnl"]).mean() * 100) if len(rand) else 0
        ex = extras.get("per", {}).get(res.name, {})
        share, n_m = ex.get("months", (0.0, 0))
        lines.append(
            f"| `{res.name}` | `{ex.get('config_id', '-')}` | {m_oos['trades']} | {fmt_money(m_oos['net_pnl'])} | "
            f"{fmt_money(m_oos['gross_pnl'])} | {m_oos['expectancy_bps']:+.1f} bps | {s_oos.get('t_stat', 0):+.2f} | "
            f"{m_oos['profit_factor']:.2f} | {_pct(s_oos.get('win_rate'))} / {_pct(s_oos.get('breakeven_win_rate'))} | "
            f"{share:.0%} of {n_m} | {fmt_money(ex.get('mc_dd95', 0.0))} | {rand_pct:.0f}% | "
            f"{'PASS' if passed else 'not yet'} |")
    lines += ["", "*Per trade* is the average result in basis points of the position (1 bp = 0.01%). "
              "*t-stat* above 2 means the result is probably not luck. *Win / breakeven*: the win rate needs to be "
              "above the breakeven rate set by the payoff. *Kept of best move*: how much of each trade's best "
              "unrealised gain was actually banked. *Months +*: share of calendar months that made money. "
              "*Worst DD (95%)*: reshuffle the out-of-sample trades 2,000 times; 95% of orderings had a smaller "
              "drawdown than this. The live kill switch uses it.", ""]
    corr = extras.get("corr")
    if corr is not None and len(corr) > 1:
        lines += ["## Do the ideas move together? (daily P&L correlation, out-of-sample)", "",
                  "Near +1 = the same bet twice. Near 0 or negative = genuinely different, which is what a "
                  "portfolio of strategies needs.", "",
                  "| | " + " | ".join(f"`{c}`" for c in corr.columns) + " |",
                  "|---|" + "---:|" * len(corr.columns)]
        for name, row in corr.iterrows():
            lines.append(f"| `{name}` | " + " | ".join("-" if not np.isfinite(v) else f"{v:+.2f}" for v in row) + " |")
        lines.append("")
    if sweep_rows:
        lines += ["## Sensitivity sweep (out-of-sample)", "",
                  f"`{args.sweep}`: a real edge shows up as a **plateau** (neighbouring values also good), "
                  "not a single lucky spike.", "",
                  "| Value | Trades | Net | Per trade | t-stat | Profit factor |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {v} | {n} | {fmt_money(net)} | {bps:+.1f} bps | {t:+.2f} | {pf:.2f} |" for v, n, net, bps, t, pf in sweep_rows]
        lines.append("")
    for res, m_is, m_oos, rand, verdict, s_oos in results:
        lines += [f"## `{res.name}`: {EXPERIMENTS.get(res.name, (None, ''))[1]}", ""]
        lines += stats_block("Out-of-sample statistics", s_oos)
        lines += [f"Random-pick baseline (OOS, {len(rand)} runs): mean {fmt_money(float(rand.mean()) if len(rand) else 0)}, "
                  f"90th percentile {fmt_money(float(np.percentile(rand, 90)) if len(rand) else 0)}. "
                  f"In-sample for comparison: {m_is['trades']} trades, {fmt_money(m_is['net_pnl'])}, "
                  f"profit factor {m_is['profit_factor']:.2f}.", "", "Success bar (out-of-sample):", ""]
        for crit, (ok, val) in verdict.items():
            lines.append(f"- {'PASS' if ok else 'FAIL'}: {crit} ({val})")
        lines += ["", f"Breakdowns by month, hour, hold time, signal strength, RSI and ticker: `detail_{res.name}.md`.", ""]
    lines += ["## How to read this", "",
              "* Only the **out-of-sample** numbers count. If an idea wins in-sample but not out-of-sample, it was fitted to noise.",
              "* An idea goes to paper trading only if it passes every line of the success bar.",
              "* If even *before costs* is negative, the signal has no edge to tune: try a different signal.",
              "* Breakdowns (hour, ticker, signal strength) are where new ideas come from. Test each one as its own "
              "experiment before believing it.",
              "* Each experiment wrote `expect_<config>.json`. When the live bot runs that exact config, it compares "
              "its real trades against it (incubation) and switches autopilot off if the drawdown passes the 95% "
              "worst case."]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


def plot(out: Path, results, split_day: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    for res, *_ in results:
        if len(res.daily):
            d = pd.to_datetime(res.daily["day"])
            ax.plot(d, res.daily["pnl"].cumsum(), label=res.name, linewidth=1.6)
    ax.axvline(pd.Timestamp(split_day), color="grey", linestyle="--", linewidth=1)
    ax.text(pd.Timestamp(split_day), ax.get_ylim()[1], "  out-of-sample", va="top", color="grey")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Cumulative net P&L by experiment (after costs)")
    ax.set_ylabel("$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    live = next((r for r, *_ in results if r.name == "live"), results[0][0])
    mfe = [t.mfe_bps for t in live.trades]
    pnl_bps = [t.pnl_pct * 1e4 for t in live.trades]
    ax2.scatter(mfe, pnl_bps, s=6, alpha=0.35)
    lim = max([abs(x) for x in mfe + pnl_bps] + [10])
    ax2.plot([0, lim], [0, lim], color="grey", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("best move while open (bps)")
    ax2.set_ylabel("result (bps)")
    ax2.set_title(f"`{live.name}`: how much of each move was kept")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "equity_curves.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    today = dt.date.today()
    first_of_month = today.replace(day=1)
    default_end = first_of_month - dt.timedelta(days=1)
    default_start = (pd.Timestamp(first_of_month) - pd.DateOffset(months=6)).date()
    ap = argparse.ArgumentParser(description="Sparticus Lab: backtest the live rules on history")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=default_start)
    ap.add_argument("--end", type=dt.date.fromisoformat, default=default_end)
    ap.add_argument("--experiments", default="all")
    ap.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    ap.add_argument("--feed", choices=["sip", "iex"], default="sip")
    ap.add_argument("--cost-bps", type=float, default=2.0, help="cost per side in basis points (spread + slippage)")
    ap.add_argument("--sell-fee", type=float, default=0.01, help="flat $ fee per sale (regulatory fees round up to a penny)")
    ap.add_argument("--oos", type=float, default=0.33, help="fraction of days held out for the verdict")
    ap.add_argument("--seeds", type=int, default=50, help="random-baseline runs")
    ap.add_argument("--out", default="lab_results")
    ap.add_argument("--cache", default="lab_data")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sweep", default="", help='one setting, several values, e.g. "reentry_cooldown_minutes=5,15,30,60"')
    args = ap.parse_args(argv)
    if args.list:
        for k, (_, d) in EXPERIMENTS.items():
            print(f"{k:16s} {d}")
        return 0
    args.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    names = list(EXPERIMENTS) if args.experiments == "all" else [e.strip() for e in args.experiments.split(",")]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"Unknown experiment(s): {unknown}. Use --list.")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.feed.upper()} bars {args.start} to {args.end} for {len(args.symbols)} tickers...")
    try:
        bars = load_bars(args.symbols, args.start, args.end, args.feed, Path(args.cache))
    except Exception as e:
        if args.feed == "sip" and "subscription" in str(e).lower():
            print(f"SIP data refused ({e}). Falling back to IEX.")
            args.feed = "iex"
            bars = load_bars(args.symbols, args.start, args.end, "iex", Path(args.cache))
        else:
            raise
    missing = [s for s in args.symbols if s not in bars or bars[s].empty]
    if missing:
        print(f"No data for {missing}; continuing without them.")
    symbols = [s for s in args.symbols if s not in missing]
    args.requested_symbols = list(args.symbols)
    return run_lab(bars, symbols, names, args, out)


def _split(res: "Result", oos: float) -> str:
    days = res.timeline_days
    return days[int(len(days) * (1 - oos))] if days else ""


def write_expectations(out: Path, name: str, cid: str, p: StrategyParams, s_oos: Dict[str, object],
                       mc: Tuple[float, float], months: Tuple[float, int], passed: bool,
                       lab_equity: float = 100_000.0) -> None:
    """What the live bot should expect from this exact config (incubation + kill switch)."""
    pnl_ci = s_oos.get("ci95") or (0.0, 0.0)
    data = {
        "name": name, "config_id": cid, "strategy": p.strategy, "passed": passed,
        "created": dt.date.today().isoformat(),
        "oos_trades": s_oos.get("trades", 0), "trades_per_day": s_oos.get("trades_per_day", 0.0),
        "per_trade_mean": s_oos.get("expectancy", 0.0), "per_trade_std": s_oos.get("std", 0.0),
        "per_trade_ci95": list(pnl_ci), "win_rate": s_oos.get("win_rate", 0.0),
        "profit_factor": s_oos.get("profit_factor", 0.0) if s_oos.get("profit_factor") != float("inf") else 99.0,
        "t_stat": s_oos.get("t_stat", 0.0),
        "mc_dd_median": mc[0], "mc_dd95": mc[1], "months_positive": months[0],
        # sizing by % of equity: the live bot rescales the $ figures to its own equity
        "lab_equity": lab_equity, "pct_sizing": p.position_pct_of_equity > 0,
    }
    (out / f"expect_{cid}.json").write_text(json.dumps(data, indent=2))


def run_lab(bars, symbols, names, args, out: Path) -> int:
    results = []
    per: Dict[str, Dict[str, object]] = {}
    universe = getattr(args, "requested_symbols", symbols)
    cids = {name: config_id(EXPERIMENTS[name][0], universe) for name in names}
    sweep_specs: List[Tuple[str, StrategyParams]] = []
    if getattr(args, "sweep", ""):
        field_name, _, values = args.sweep.partition("=")
        field_name = field_name.strip()
        base_val = getattr(LIVE_DEFAULTS, field_name)
        for raw in values.split(","):
            v = type(base_val)(float(raw)) if not isinstance(base_val, bool) else raw.strip().lower() == "true"
            sweep_specs.append((f"{field_name}={v}", replace(LIVE_DEFAULTS, **{field_name: v})))
    n_configs = update_ledger(out, [(n, c) for n, c in cids.items()] +
                              [(n, config_id(sp, universe)) for n, sp in sweep_specs])
    t_bar = required_t(n_configs)
    print(f"Running experiments (ledger: {n_configs} configs tested so far -> t-stat bar {t_bar:.2f}):")
    daily_oos: Dict[str, pd.Series] = {}
    for name in names:
        p, _ = EXPERIMENTS[name]
        eq0 = LAB_EQUITY.get(name, 100.0 if p.fractional else 100_000.0)   # micro configs: a $100 account
        fee = getattr(args, "sell_fee", 0.01)
        res = simulate(name, p, bars, symbols, args.start, args.end, args.cost_bps, start_equity=eq0, sell_fee=fee)
        split_day = _split(res, args.oos)
        is_tr = [t for t in res.trades if t.day < split_day]
        oos_tr = [t for t in res.trades if t.day >= split_day]
        d_is, d_oos = res.daily[res.daily["day"] < split_day], res.daily[res.daily["day"] >= split_day]
        m_is = metrics(is_tr, d_is, eq0)
        m_oos = metrics(oos_tr, d_oos, eq0)
        s_oos = S.summarize([t.as_stats() for t in oos_tr])
        rand = random_baseline(res, oos_tr, symbols, args.cost_bps, args.seeds, sell_fee=fee)
        mc = monte_carlo_dd([t.pnl for t in oos_tr])
        months = months_positive(res.trades)
        verdict = judge(m_oos, rand, float(s_oos.get("t_stat", 0.0)), t_bar, months)
        passed = all(ok for ok, _ in verdict.values())
        results.append((res, m_is, m_oos, rand, verdict, s_oos))
        per[name] = {"config_id": cids[name], "mc_dd95": mc[1], "months": months}
        daily_oos[name] = d_oos.set_index("day")["pnl"]
        if name not in EXPECT_SKIP:
            eq_at_oos = float(d_is["equity"].iloc[-1]) if len(d_is) else eq0   # account size the verdict trades saw
            write_expectations(out, name, cids[name], p, s_oos, mc, months, passed, eq_at_oos)
        pd.DataFrame([asdict(t) for t in res.trades]).to_csv(out / f"trades_{name}.csv", index=False)
        res.daily.to_csv(out / f"daily_{name}.csv", index=False)
        write_detail(out, res, split_day)
    corr = pd.DataFrame(daily_oos).fillna(0.0).corr() if len(daily_oos) > 1 else None

    sweep_rows = None
    if sweep_specs:
        sweep_rows = []
        print(f"Sweeping {args.sweep.partition('=')[0].strip()}:")
        for label, sp in sweep_specs:
            v = label.partition("=")[2]
            res = simulate(label, sp, bars, symbols, args.start, args.end, args.cost_bps,
                           sell_fee=getattr(args, "sell_fee", 0.01))
            split_day = _split(res, args.oos)
            oos_tr = [t for t in res.trades if t.day >= split_day]
            s_ = S.summarize([t.as_stats() for t in oos_tr])
            pf = s_.get("profit_factor", 0.0)
            sweep_rows.append((v, s_["trades"], s_.get("net_pnl", 0.0), s_.get("expectancy_bps", 0.0),
                               s_.get("t_stat", 0.0), pf if pf != float("inf") else 99.0))

    first = results[0][0]
    days = first.timeline_days
    split_day = _split(first, args.oos)
    spy_is = spy_buy_hold(bars, days[0], days[max(0, days.index(split_day) - 1)])
    spy_oos = spy_buy_hold(bars, split_day, days[-1])
    text = write_report(out, results, args, split_day, spy_is, spy_oos, sweep_rows,
                        {"t_bar": t_bar, "n_configs": n_configs, "per": per, "corr": corr})
    plot(out, results, split_day)
    print()
    print(text.split("## `")[0])
    print(f"Full report: {out / 'summary.md'}  |  details: {out}/detail_*.md  |  chart: {out / 'equity_curves.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
