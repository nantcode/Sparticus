"""V6.2 tests: strategy families, bracket modes, the lab's validation upgrades, incubation, the kill
switch and the journal upgrade. Offline: no Alpaca account, network or real data.

    python3 -m unittest discover -s tests -v
"""
import csv
import datetime as dt
import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import backtest as bt  # noqa: E402
import sparticus_incubation as inc  # noqa: E402
import sparticus_stats as S  # noqa: E402
from sparticus_journal import FIELDS, TradeJournal  # noqa: E402
from sparticus_strategy import LIVE_DEFAULTS, atr, bracket_levels, config_id  # noqa: E402

ET = "America/New_York"


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


def weekdays(start: dt.date, n: int):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def session(day: dt.date) -> pd.DatetimeIndex:
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(4, 0)), tz=ET)
    return pd.date_range(start, periods=16 * 60, freq="1min").tz_convert("UTC")


def day_frame(day: dt.date, rth_price, before: float = 100.0) -> pd.DataFrame:
    """rth_price(m) -> close for the bar starting m minutes after 09:30. Flat outside regular hours."""
    idx = session(day)
    et = idx.tz_convert(ET)
    open_ = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET)
    closes, last = [], before
    for t in et:
        m = int((t - open_).total_seconds() // 60)
        if 0 <= m < 390:
            last = float(rth_price(m))
        closes.append(last)
    closes = np.array(closes)
    opens = np.concatenate([[before], closes[:-1]])
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes),
                         "close": closes, "volume": 1000.0}, index=idx)


def history_day(day):  # oscillates 99..101: daily range 2 -> daily ATR about 2
    return day_frame(day, lambda m: 100 + math.sin(m / 20.0))


class TestBrackets(unittest.TestCase):
    def test_pct_mode_unchanged(self):
        self.assertEqual(bracket_levels(100.0, True), (104.0, 98.0))
        self.assertEqual(bracket_levels(100.0, False), (96.0, 102.0))

    def test_atr_mode_floor_and_scaling(self):
        p = replace(LIVE_DEFAULTS, bracket_mode="atr", atr_stop_mult=3.0, min_stop_pct=0.004, reward_risk=2.0)
        self.assertEqual(bracket_levels(100.0, True, p, atr_value=0.05), (100.8, 99.6))   # floor wins
        self.assertEqual(bracket_levels(100.0, True, p, atr_value=0.25), (101.5, 99.25))  # ATR wins
        self.assertEqual(bracket_levels(100.0, False, p, atr_value=0.25), (98.5, 100.75))
        self.assertEqual(bracket_levels(100.0, True, p, atr_value=None), (100.8, 99.6))

    def test_atr_function(self):
        self.assertIsNone(atr([1, 2], [0, 1], [0.5, 1.5], 14))
        h, l, c = [10.0] * 16, [9.0] * 16, [9.5] * 16
        self.assertAlmostEqual(atr(h, l, c, 14), 1.0)

    def test_config_id(self):
        a = config_id(LIVE_DEFAULTS)
        self.assertEqual(a, config_id(replace(LIVE_DEFAULTS)))
        self.assertNotEqual(a, config_id(replace(LIVE_DEFAULTS, take_profit_pct=0.008)))
        self.assertEqual(len(a), 8)


class TestORB(unittest.TestCase):
    days = weekdays(dt.date(2026, 6, 1), 17)

    def run_orb(self, test_path):
        hist = {d: history_day(d) for d in self.days[:-1]}
        last = self.days[-1]
        x = pd.concat(list(hist.values()) + [day_frame(last, test_path)])
        spy = pd.concat(list(hist.values()) + [day_frame(last, lambda m: 100 - 0.01 * m)])  # SPY: first candle down
        p = replace(LIVE_DEFAULTS, strategy="orb", signal_exit=False, max_entries_per_symbol_per_day=1)
        res = bt.simulate("orb", p, {"SPY": spy, "X": x}, ["SPY", "X"], self.days[0], last, cost_bps=0.0, verbose=False)
        return [t for t in res.trades if t.symbol == "X" and t.day == last.isoformat()]

    @staticmethod
    def breakout_then(after):
        def f(m):
            if m < 5:
                return (100.1, 100.5, 100.3, 100.3, 100.4)[m]   # first 5 minutes up, range high 100.5
            if m < 10:
                return 100.3                    # back inside the range
            return after(m)
        return f

    def test_breakout_held_to_the_close(self):
        trades = self.run_orb(self.breakout_then(lambda m: 101.0))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.entry_time[-5:], "09:40")
        self.assertAlmostEqual(t.entry_px, 100.5, places=6)       # filled at the range high
        self.assertEqual(t.reason, "eod")
        self.assertGreater(t.pnl, 0)

    def test_breakout_then_collapse_hits_the_stop(self):
        trades = self.run_orb(self.breakout_then(lambda m: 100.6 if m < 12 else 99.0))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].reason, "stop")
        self.assertLess(trades[0].pnl, 0)
        self.assertGreater(trades[0].exit_px, 99.5)               # stop price, not the 99 low

    def test_no_trade_when_the_first_candle_is_down(self):
        trades = self.run_orb(lambda m: 100.0 - 0.1 * m if m < 5 else 101.0)
        self.assertEqual(trades, [])


class TestSpyMomentum(unittest.TestCase):
    days = weekdays(dt.date(2026, 6, 1), 3)

    def run_mom(self, first_half_hour_up: bool):
        d0, d1, d2 = self.days
        path = (lambda m: 100 + min(m, 30) / 30 + (0.5 if m >= 360 else 0)) if first_half_hour_up \
            else (lambda m: 100 - min(m, 30) / 30)
        spy = pd.concat([history_day(d0), day_frame(d1, lambda m: 100.0), day_frame(d2, path)])
        p = replace(LIVE_DEFAULTS, strategy="spy_momentum", signal_exit=False)
        res = bt.simulate("mom", p, {"SPY": spy}, ["SPY"], d0, d2, cost_bps=0.0, verbose=False)
        return [t for t in res.trades if t.day == d2.isoformat()]

    def test_up_morning_buys_at_1530_and_sells_at_the_flatten(self):
        trades = self.run_mom(True)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].entry_time[-5:], "15:30")
        self.assertEqual(trades[0].reason, "eod")
        self.assertGreater(trades[0].pnl, 0)

    def test_down_morning_does_nothing(self):
        self.assertEqual(self.run_mom(False), [])


class TestValidation(unittest.TestCase):
    def test_required_t_rises_with_ideas_tested(self):
        self.assertEqual(bt.required_t(1), 2.0)
        self.assertAlmostEqual(bt.required_t(10), 2.807, places=2)
        self.assertGreater(bt.required_t(100), bt.required_t(10))

    def test_ledger_counts_distinct_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(bt.update_ledger(out, [("a", "c1"), ("b", "c2")]), 2)
            self.assertEqual(bt.update_ledger(out, [("a", "c1"), ("c", "c3")]), 3)   # c1 already counted

    def test_monte_carlo_drawdown(self):
        pnl = [5.0, -3.0, 4.0, -6.0, 2.0, -1.0, 3.0, -4.0] * 10
        med, worst = bt.monte_carlo_dd(pnl, runs=500)
        self.assertLessEqual(worst, med)
        self.assertLess(worst, 0)
        self.assertGreaterEqual(worst, -sum(-x for x in pnl if x < 0))   # can't lose more than all losers

    def test_months_positive(self):
        T = types.SimpleNamespace
        trades = [T(day="2026-04-02", pnl=5.0), T(day="2026-04-09", pnl=-2.0), T(day="2026-05-01", pnl=-1.0)]
        self.assertEqual(bt.months_positive(trades), (0.5, 2))

    def test_run_lab_writes_expectations_per_config(self):
        import test_lab as TL
        days = [d for d in TL.DAYS]
        syms = ["SPY", "A", "B"]
        bars = {s: TL.random_walk(i + 1, vol=0.0015) for i, s in enumerate(syms)}
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(start=days[0], end=days[-1], symbols=syms, requested_symbols=syms,
                                         feed="sip", cost_bps=2.0, oos=0.33, seeds=3, sweep="")
            saved = dict(bt.EXPERIMENTS)
            try:
                bt.EXPERIMENTS.clear()
                bt.EXPERIMENTS.update({k: saved[k] for k in ("live", "cap2")})
                bt.run_lab(bars, syms, ["live", "cap2"], args, Path(tmp))
            finally:
                bt.EXPERIMENTS.clear()
                bt.EXPERIMENTS.update(saved)
            cid = config_id(LIVE_DEFAULTS, syms)
            data = json.loads((Path(tmp) / f"expect_{cid}.json").read_text())
            self.assertEqual(data["config_id"], cid)
            self.assertLessEqual(data["mc_dd95"], 0)
            summary = (Path(tmp) / "summary.md").read_text()
            self.assertIn("The bar this run", summary)
            self.assertIn("Do the ideas move together", summary)


def rows(pnls, cid="cfg1"):
    return [{"pnl": x, "notional": 5000.0, "exit_time": f"2026-10-0{1 + i % 5} 10:{i % 60:02d}",
             "config_id": cid} for i, x in enumerate(pnls)]


class TestIncubation(unittest.TestCase):
    expect = {"per_trade_mean": 1.0, "per_trade_std": 10.0, "mc_dd95": -300.0}

    def test_no_baseline_and_collecting(self):
        self.assertEqual(inc.evaluate(S.summarize(rows([1.0, -1.0])), None, 100_000)["status"], "no_baseline")
        self.assertEqual(inc.evaluate(S.summarize(rows([1.0, -1.0])), self.expect, 100_000)["status"], "collecting")

    def test_on_track_and_below(self):
        rng = np.random.default_rng(1)
        ok = list(rng.normal(1.0, 10.0, 60))
        self.assertEqual(inc.evaluate(S.summarize(rows(ok)), self.expect, 100_000)["status"], "on_track")
        bad = [-6.0, 1.0] * 30                      # mean -2.5 vs lab +1, std 10, n 60 -> z about -2.7
        self.assertEqual(inc.evaluate(S.summarize(rows(bad)), self.expect, 100_000)["status"], "below")

    def test_kill_on_lab_drawdown_and_default(self):
        r = inc.evaluate(S.summarize(rows([-50.0] * 7)), self.expect, 100_000)
        self.assertTrue(r["kill"])
        self.assertEqual(r["kill_at"], -300.0)
        r = inc.evaluate(S.summarize(rows([-500.0] * 7)), None, 100_000)   # no lab: 3% of equity
        self.assertTrue(r["kill"])
        self.assertEqual(r["kill_at"], -3000.0)


class TestJournalUpgrade(unittest.TestCase):
    def test_old_journal_gets_new_columns_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.csv"
            old = [f for f in FIELDS if f not in ("strategy", "config_id")]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=old)
                w.writeheader()
                w.writerow({"symbol": "NVDA", "pnl": "12.5", "exit_time": "2026-10-06 10:00"})
            j = TradeJournal(str(path), "V6.2", "sma_cross", "abc12345")
            j.note_fill("KO", types.SimpleNamespace(qty="10", avg_entry_price="70", current_price="70"))
            j.close("KO", 70.5)
            back = j.read()
            self.assertEqual(path.read_text().splitlines()[0].split(","), FIELDS)
            self.assertEqual(back[0]["symbol"], "NVDA")
            self.assertEqual(back[0]["config_id"], "")
            self.assertEqual((back[1]["symbol"], back[1]["config_id"], back[1]["strategy"]), ("KO", "abc12345", "sma_cross"))


class TestKillSwitchInTheBot(unittest.TestCase):
    def test_drawdown_trips_the_kill_switch_and_restart_keeps_it(self):
        from test_stats_journal import load_bot
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                bot = load_bot()
                eng = bot.SparticusEngineV55()
                eng.account_snapshot = types.SimpleNamespace(equity="100000")
                for i in range(8):
                    eng.journal._append({"symbol": "X", "pnl": -500.0, "notional": 5000.0,
                                         "exit_time": f"2026-10-06 10:{i:02d}", "config_id": bot.CONFIG_ID})
                self.assertTrue(eng.autopilot_enabled)
                eng.refresh_stats()
                self.assertTrue(eng.killed)
                self.assertFalse(eng.autopilot_enabled)
                saved = json.loads(Path(bot.KILL_FILE).read_text())
                self.assertEqual(saved["config_id"], bot.CONFIG_ID)
                self.assertEqual(eng.stats_cache["incubation"]["status"], "kill")
                self.assertEqual(eng.stats_cache["config"]["trades"], 8)
            finally:
                os.chdir(cwd)

    def test_entry_brackets_use_the_shared_function(self):
        from test_stats_journal import load_bot
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)   # keep the bot's log file out of your folder
            try:
                bot = load_bot()
            finally:
                os.chdir(cwd)
        src = (ROOT / "sparticus_v6_4.py").read_text()
        self.assertIn("bracket_levels(limit_price, side == OrderSide.BUY, STRATEGY, atr_val)", src)
        self.assertNotIn("limit_price * (1.0 + TAKE_PROFIT_PCT)", src)
        self.assertEqual(bot.CONFIG_ID, config_id(LIVE_DEFAULTS))


if __name__ == "__main__":
    unittest.main()


class TestPartialFillKeepsPlannedLevels(unittest.TestCase):
    def test_reattached_protection_uses_the_entry_brackets(self):
        from test_stats_journal import load_bot
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                bot = load_bot()
            finally:
                os.chdir(cwd)
        submitted, closed = [], []
        journal = types.SimpleNamespace(pending={}, open={"MRK": {"tp": 141.92, "sl": 140.12}})
        fake = types.SimpleNamespace(
            journal=journal, get_signal=lambda s: {"atr": 0.2},
            trading_client=types.SimpleNamespace(
                get_orders=lambda req: [],
                get_open_position=lambda s: types.SimpleNamespace(qty="48", avg_entry_price="140.71875"),
                submit_order=lambda r: submitted.append(r)),
            close_position_safely=lambda *a, **k: closed.append(a))
        order = types.SimpleNamespace(symbol="MRK", qty="53", limit_price="140.72")
        from dataclasses import replace as _replace
        bot.STRATEGY = _replace(bot.STRATEGY, fractional=False, position_pct_of_equity=0.0)   # whole-share mode
        bot.SparticusEngineV55.protect_partial_fill(fake, order)
        self.assertEqual(closed, [])
        r = submitted[0]
        self.assertEqual((r.qty, r.take_profit.limit_price, r.stop_loss.stop_price), (48, 141.92, 140.12))
