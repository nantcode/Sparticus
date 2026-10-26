"""Sparticus strategy rules: the ONE place the trading logic lives.

The live bot (sparticus_v6_4.py) and the backtester (backtest.py) both import this file,
so a backtest result describes exactly the rules the live bot trades. Change a rule here and
both change together; there is no second copy to drift out of sync.

Everything in this file is pure: no broker calls, no clocks, no I/O.

V6.2 adds:
  * Strategy families (lab-only until they pass): "orb" (opening range breakout) and
    "spy_momentum" (first half-hour -> last half-hour). The live bot trades "sma_cross" only.
  * Bracket modes: "pct" (fixed %, as before) or "atr" (volatility-scaled, target = reward_risk x stop).
  * config_id(): a short fingerprint of every setting, so the journal, the lab and the
    incubation tracker always know exactly which rules produced a trade.

V6.3 adds micro mode for small accounts: fractional=True + position_pct_of_equity (sizing as a share
of equity). position_size() is the one sizing rule, used by both the bot and the lab.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

# The watchlist the live bot scans (order = priority when several tickers signal at once).
DEFAULT_UNIVERSE: List[str] = [
    # original 28
    "MU", "AVGO", "ANET", "SMCI", "COST", "DIS", "JPM", "XOM", "LLY", "LMT", "RTX", "GD", "AVAV", "NOC",
    "GOOGL", "KO", "PEP", "WMT", "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "NFLX",
    # +25 (off for now: Oct 7-9 traded the original 28, so this week changes only the hold time).
    # To scan all 53, remove the "#" from the next two lines.
    # "BAC", "GS", "V", "MA", "UNH", "JNJ", "ABBV", "MRK", "CVX", "COP", "CAT", "BA", "GE", "HON",
    # "HD", "MCD", "NKE", "SBUX", "ORCL", "CRM", "PLTR", "UBER", "IWM", "GLD", "TLT",
]

LIVE_STRATEGIES = ("sma_cross",)          # what the live bot can trade today
LAB_STRATEGIES = ("sma_cross", "orb", "spy_momentum")


@dataclass(frozen=True)
class StrategyParams:
    """Every knob of the strategy. Defaults = exactly what V5.5.x / V6.x trade live."""
    strategy: str = "sma_cross"
    # signal (sma_cross)
    fast_sma: int = 5
    slow_sma: int = 20
    rsi_period: int = 14
    min_rsi: float = 50.0
    max_rsi: float = 80.0
    require_fresh_cross: bool = True
    cross_lookback: int = 3
    enable_shorts: bool = False
    bar_minutes: int = 1                    # live trades 1-minute bars; 5 is an experiment
    lookback_minutes: int = 120             # history window the live scanner requests
    max_bar_age_minutes: float = 5.0        # older latest bar = stale ticker, no entries
    # exits
    bracket_mode: str = "pct"               # "pct" = fixed % below | "atr" = volatility-scaled
    take_profit_pct: float = 0.04
    stop_loss_pct: float = 0.02
    atr_period: int = 14                    # bars used for ATR (bracket_mode="atr")
    atr_stop_mult: float = 3.0              # stop distance = mult x ATR of 1-minute bars...
    min_stop_pct: float = 0.004             # ...but never tighter than this
    reward_risk: float = 2.0                # target distance = reward_risk x stop distance
    hard_stop_pct: float = -0.03
    signal_exit: bool = True
    min_hold_minutes: float = 30.0          # Oct 12: was 5.0. Oct 8-9 review: 1-min reversals cut trades in minutes
    # pacing and gating
    reentry_cooldown_minutes: float = 15.0
    max_entries_per_symbol_per_day: int = 0  # 0 = unlimited
    max_open_positions: int = 6
    no_entry_first_minutes: float = 5.0
    no_entry_last_minutes: float = 15.0
    eod_flatten_minutes: float = 5.0
    target_dollars_per_trade: float = 5000.0
    daily_loss_limit_pct: float = -0.05
    # sizing for small accounts (micro mode)
    position_pct_of_equity: float = 0.0     # > 0: each position = this share of equity (0.30 = 30%)
    fractional: bool = False                # True: fractional shares (long only, no broker brackets)
    min_order_dollars: float = 1.0          # broker minimum for fractional orders
    # experiment-only filters (off by default, not used by the live bot yet)
    market_filter: str = "none"             # "none" | "spy_trend": longs only when SPY > its SMA
    market_filter_sma: int = 60
    # ORB: opening range breakout (Zarattini, Barbon & Aziz)
    orb_minutes: int = 5                    # opening range length
    orb_atr_days: int = 14                  # daily ATR used for the stop
    orb_stop_atr_frac: float = 0.10         # stop = 10% of the daily ATR below the entry
    orb_reward_risk: float = 0.0            # 0 = no target: hold to the end-of-day flatten
    orb_min_rel_volume: float = 1.0         # opening-range volume vs its 14-day average
    orb_top_n: int = 6                      # trade the N "most in play" tickers each day
    # SPY intraday momentum (Gao, Han, Li & Zhou, JFE 2018)
    mom_symbol: str = "SPY"
    mom_entry_minutes_before_close: int = 30
    mom_min_first_half_hour_ret: float = 0.0


LIVE_DEFAULTS = StrategyParams()
_ADDED_LATER = {"position_pct_of_equity": 0.0, "fractional": False, "min_order_dollars": 1.0}   # V6.3


def config_id(p: StrategyParams, symbols: Sequence[str] = DEFAULT_UNIVERSE) -> str:
    """Short fingerprint of every setting + the watchlist. Same rules -> same id, everywhere."""
    d = asdict(p)
    for k, v in _ADDED_LATER.items():     # settings added after V6.2: left at their defaults they don't
        if d.get(k) == v:                 # change the id, so an upgrade never restarts an incubation
            d.pop(k)
    blob = json.dumps({"p": d, "u": sorted(symbols)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def sma(values: List[float], period: int, offset: int = 0) -> Optional[float]:
    """Simple moving average ending `offset` bars before the latest bar."""
    end = len(values) - offset
    start = end - period
    if start < 0 or period <= 0:
        return None
    return sum(values[start:end]) / period


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Wilder-smoothed RSI. Returns None if there isn't enough data."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> Optional[float]:
    """Average true range: the plain mean of the last `period` true ranges. None if too few bars."""
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return None
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(n - period, n)]
    return sum(trs) / period


def min_bars(p: StrategyParams) -> int:
    return max(p.slow_sma + p.cross_lookback, p.rsi_period + 1, p.atr_period + 1) + 1


@dataclass(frozen=True)
class Signal:
    fast: float
    slow: float
    rsi: float
    crossed_up: bool
    crossed_down: bool
    long_entry: bool
    short_entry: bool
    exit_long: bool    # trend reversed against a long
    exit_short: bool   # trend reversed against a short


def compute_signal(closes: List[float], p: StrategyParams = LIVE_DEFAULTS) -> Optional[Signal]:
    """The entire sma_cross entry/exit signal. None = not enough bars yet."""
    if len(closes) < max(p.slow_sma + p.cross_lookback, p.rsi_period + 1) + 1:
        return None
    fast = sma(closes, p.fast_sma)
    slow = sma(closes, p.slow_sma)
    rsi = calculate_rsi(closes, p.rsi_period)
    if fast is None or slow is None or rsi is None:
        return None
    prior = [(sma(closes, p.fast_sma, k), sma(closes, p.slow_sma, k)) for k in range(1, p.cross_lookback + 1)]
    crossed_up = any(f is not None and s is not None and f <= s for f, s in prior)
    crossed_down = any(f is not None and s is not None and f >= s for f, s in prior)
    long_entry = (fast > slow and p.min_rsi <= rsi <= p.max_rsi
                  and (crossed_up or not p.require_fresh_cross))
    short_entry = (p.enable_shorts and fast < slow
                   and (100 - p.max_rsi) <= rsi <= (100 - p.min_rsi)
                   and (crossed_down or not p.require_fresh_cross))
    return Signal(fast, slow, rsi, crossed_up, crossed_down, long_entry, short_entry,
                  exit_long=fast < slow, exit_short=fast > slow)


def position_size(price: float, equity: float, p: StrategyParams = LIVE_DEFAULTS) -> Tuple[float, str]:
    """Shares to buy at `price`. Returns (qty, "") or (0, reason). Shared by the bot and the lab.

    Budget = position_pct_of_equity x equity when set, otherwise target_dollars_per_trade.
    Fractional: budget / price rounded down to 4 decimals. Whole shares: at least 1 (the bot's
    hard cap then rejects a single share that costs more than 110% of the budget)."""
    if price <= 0:
        return 0.0, "no price"
    budget = equity * p.position_pct_of_equity if p.position_pct_of_equity > 0 else p.target_dollars_per_trade
    if p.fractional:
        qty = int(budget / price * 10_000) / 10_000
        if qty * price < p.min_order_dollars:
            return 0.0, f"${budget:,.2f} budget is under the ${p.min_order_dollars:g} minimum order"
        return qty, ""
    return float(max(1, int(budget / price))), ""


def position_budget(equity: float, p: StrategyParams = LIVE_DEFAULTS) -> float:
    return equity * p.position_pct_of_equity if p.position_pct_of_equity > 0 else p.target_dollars_per_trade


def bracket_levels(entry: float, is_long: bool, p: StrategyParams = LIVE_DEFAULTS,
                   atr_value: Optional[float] = None) -> Tuple[float, float]:
    """(take_profit, stop_loss) prices for a new position, rounded like the live orders.

    pct mode: fixed take_profit_pct / stop_loss_pct.
    atr mode: stop distance = max(atr_stop_mult x ATR, min_stop_pct x entry), target = reward_risk x
    that distance. Without an ATR value it falls back to the min_stop_pct floor, never to pct mode.
    """
    if p.bracket_mode == "atr":
        dist = max((atr_value or 0.0) * p.atr_stop_mult, entry * p.min_stop_pct)
        tp_dist = dist * p.reward_risk
        if is_long:
            return round(entry + tp_dist, 2), round(entry - dist, 2)
        return round(entry - tp_dist, 2), round(entry + dist, 2)
    if is_long:
        return round(entry * (1 + p.take_profit_pct), 2), round(entry * (1 - p.stop_loss_pct), 2)
    return round(entry * (1 - p.take_profit_pct), 2), round(entry * (1 + p.stop_loss_pct), 2)
