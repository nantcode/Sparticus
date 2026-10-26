"""Offline tests for the Sparticus Lab. No Alpaca account, network or real data needed.

    python3 -m unittest discover -s tests -v
"""
import datetime as dt
import random
import os
import tempfile
import sys
import unittest
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest as bt  # noqa: E402
from sparticus_strategy import LIVE_DEFAULTS, bracket_levels, calculate_rsi, compute_signal, sma  # noqa: E402

ET = "America/New_York"
DAYS = [dt.date(2026, 9, 21) + dt.timedelta(days=i) for i in range(12)]
DAYS = [d for d in DAYS if d.weekday() < 5]


# Every test runs in a throwaway folder, never in your bot folder: a test that creates the engine
# would otherwise open (and could rewrite) your real sparticus_journal.csv and log files.
_SAFE_CWD = []


def setUpModule():
    _SAFE_CWD.append((os.getcwd(), tempfile.TemporaryDirectory()))
    os.chdir(_SAFE_CWD[-1][1].name)


def tearDownModule():
    cwd, tmp = _SAFE_CWD.pop()
    os.chdir(cwd)
    tmp.cleanup()


def session_index(day: dt.date) -> pd.DatetimeIndex:
    """Every minute 04:00-19:59 ET, like Alpaca's all-session minute bars."""
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(4, 0)), tz=ET)
    return pd.date_range(start, periods=16 * 60, freq="1min").tz_convert("UTC")


def frame(idx: pd.DatetimeIndex, closes: np.ndarray, wiggle: float = 0.0004) -> pd.DataFrame:
    opens = np.concatenate([[closes[0]], closes[:-1]])
    hi = np.maximum(opens, closes) * (1 + wiggle)
    lo = np.minimum(opens, closes) * (1 - wiggle)
    return pd.DataFrame({"open": opens, "high": hi, "low": lo, "close": closes, "volume": 1000.0}, index=idx)


def random_walk(seed: int, days=DAYS, vol=0.0012, drift=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts, p = [], 50 + seed * 7.0
    for d in days:
        idx = session_index(d)
        steps = rng.normal(drift, vol, len(idx))
        closes = p * np.exp(np.cumsum(steps))
        p = closes[-1]
        parts.append(frame(idx, closes))
    return pd.concat(parts)


def scripted(day: dt.date, path_after_1030) -> pd.DataFrame:
    """Flat at 100 (tiny alternating noise) until 10:30 ET, then follows the scripted returns."""
    idx = session_index(day)
    et = idx.tz_convert(ET)
    closes = np.array([100.0 + (0.01 if i % 2 else -0.01) for i in range(len(idx))])
    k0 = int(np.argmax(et >= pd.Timestamp(dt.datetime.combine(day, dt.time(10, 30)), tz=ET)))
    p = closes[k0 - 1]
    for i, r in enumerate(path_after_1030):
        p *= 1 + r
        closes[k0 + i] = p
    closes[k0 + len(path_after_1030):] = p
    return frame(idx, closes, wiggle=0.0)


def live_inline_signal(closes, p=LIVE_DEFAULTS):
    """Verbatim copy of the V5.5.4 scanner's inline signal logic, to prove compute_signal matches it."""
    fast = sma(closes, p.fast_sma)
    slow = sma(closes, p.slow_sma)
    rsi = calculate_rsi(closes, p.rsi_period)
    if fast is None or slow is None or rsi is None:
        return None
    prior = [(sma(closes, p.fast_sma, k), sma(closes, p.slow_sma, k)) for k in range(1, p.cross_lookback + 1)]
    crossed_up = any(f is not None and s is not None and f <= s for f, s in prior)
    long_signal = fast > slow and p.min_rsi <= rsi <= p.max_rsi and (crossed_up or not p.require_fresh_cross)
    return long_signal, fast < slow


class TestStrategyMatchesLiveBot(unittest.TestCase):
    def test_compute_signal_equals_live_scanner_logic(self):
        rng = random.Random(1)
        checked = 0
        for _ in range(6000):
            n = rng.randint(24, 140)
            p0, closes = 100.0, []
            for _ in range(n):
                p0 *= 1 + rng.gauss(0, 0.002)
                closes.append(round(p0, 2))
            live = live_inline_signal(closes)
            new = compute_signal(closes)
            if live is None:
                self.assertIsNone(new)
                continue
            self.assertEqual((new.long_entry, new.exit_long), live)
            checked += 1
        self.assertGreater(checked, 5000)


class TestSimulatorRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        syms = ["SPY"] + [f"S{i}" for i in range(9)]
        cls.bars = {s: random_walk(i + 1, vol=0.0015) for i, s in enumerate(syms)}
        cls.syms = syms
        cls.res = bt.simulate("live", LIVE_DEFAULTS, cls.bars, syms, DAYS[0], DAYS[-1], cost_bps=2.0, verbose=False)

    def test_produces_trades(self):
        self.assertGreater(len(self.res.trades), 20)

    def test_entries_respect_the_session_windows(self):
        for t in self.res.trades:
            hm = t.entry_time[-5:]
            self.assertGreaterEqual(hm, "09:35", t)   # live: first scan allowed 09:35:05, fills in the 09:35 bar
            self.assertLessEqual(hm, "15:45", t)

    def test_never_holds_overnight(self):
        for t in self.res.trades:
            self.assertEqual(t.entry_time[:10], t.exit_time[:10], t)
            self.assertNotEqual(t.reason, "carry", t)

    def test_cooldown_between_trades_in_the_same_ticker(self):
        by_sym = defaultdict(list)
        for t in self.res.trades:
            by_sym[t.symbol].append(t)
        for trades in by_sym.values():
            for a, b in zip(trades, trades[1:]):
                if a.day == b.day:
                    gap = (pd.Timestamp(b.entry_time) - pd.Timestamp(a.exit_time)).total_seconds() / 60
                    self.assertGreaterEqual(gap, LIVE_DEFAULTS.reentry_cooldown_minutes, (a, b))

    def test_never_more_than_max_positions(self):
        events = []
        for t in self.res.trades:
            events += [(t.entry_time, 1), (t.exit_time, -1)]
        open_, peak = 0, 0
        for _, d in sorted(events, key=lambda e: (e[0], e[1])):
            open_ += d
            peak = max(peak, open_)
        self.assertLessEqual(peak, LIVE_DEFAULTS.max_open_positions)

    def test_costs_are_charged(self):
        net = sum(t.pnl for t in self.res.trades)
        gross = sum(t.gross_pnl for t in self.res.trades)
        self.assertLess(net, gross)

    def test_signal_exits_wait_for_minimum_hold(self):
        for t in self.res.trades:
            if t.reason == "signal":
                self.assertGreaterEqual(t.hold_min, LIVE_DEFAULTS.min_hold_minutes, t)

    def test_entry_cap(self):
        res = bt.simulate("cap2", replace(LIVE_DEFAULTS, max_entries_per_symbol_per_day=2), self.bars,
                          self.syms, DAYS[0], DAYS[-1], cost_bps=2.0, verbose=False)
        counts = defaultdict(int)
        for t in res.trades:
            counts[(t.symbol, t.entry_time[:10])] += 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_five_minute_bars_run(self):
        res = bt.simulate("bars5", replace(LIVE_DEFAULTS, bar_minutes=5), self.bars, self.syms,
                          DAYS[0], DAYS[-1], cost_bps=2.0, verbose=False)
        self.assertTrue(all(t.entry_time[-1] in "05" for t in res.trades))   # fills on 5-minute boundaries

    def test_random_baseline_and_metrics(self):
        rand = bt.random_baseline(self.res, self.res.trades, self.syms, 2.0, 5, 5000.0)
        self.assertEqual(len(rand), 5)
        m = bt.metrics(self.res.trades, self.res.daily, 100_000.0)
        self.assertEqual(m["trades"], len(self.res.trades))
        self.assertAlmostEqual(m["net_pnl"], sum(t.pnl for t in self.res.trades), places=2)


class TestScriptedScenarios(unittest.TestCase):
    """Hand-built price paths where the right answer is known in advance."""
    day = DAYS[0]

    def run_one(self, path, p=LIVE_DEFAULTS):
        bars = {"SPY": scripted(self.day, [0.0] * 10), "X": scripted(self.day, path)}
        return bt.simulate("t", p, bars, ["SPY", "X"], self.day, self.day, cost_bps=0.0, verbose=False).trades

    # a choppy climb (RSI stays between 50 and 80): +0.12%, -0.06% repeating
    climb = [0.0012, -0.0006] * 120

    def test_take_profit_hits_at_the_bracket_price(self):
        trades = self.run_one(self.climb, replace(LIVE_DEFAULTS, take_profit_pct=0.01, signal_exit=False))
        self.assertTrue(trades)
        first = trades[0]
        self.assertEqual(first.reason, "target")
        tp, _ = bracket_levels(first.entry_px, True, replace(LIVE_DEFAULTS, take_profit_pct=0.01))
        self.assertAlmostEqual(first.exit_px, tp, delta=first.entry_px * 0.0015)

    def test_stop_loss_when_price_collapses(self):
        path = self.climb[:20] + [-0.006] * 8
        trades = self.run_one(path, replace(LIVE_DEFAULTS, signal_exit=False))
        self.assertTrue(trades)
        self.assertEqual(trades[0].reason, "stop")
        self.assertLess(trades[0].pnl_pct, -0.015)

    def test_eod_flatten_closes_whatever_is_open(self):
        trades = self.run_one(self.climb, replace(LIVE_DEFAULTS, take_profit_pct=0.5, signal_exit=False))
        self.assertEqual(trades[-1].reason, "eod")
        self.assertEqual(trades[-1].exit_time[-5:], "15:55")


if __name__ == "__main__":
    unittest.main()
