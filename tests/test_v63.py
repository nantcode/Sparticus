"""V6.3 tests: micro mode for small accounts (fractional shares, sizing by % of equity, broker-side
stops placed after the fill, the take-profit run by Sparticus). Offline: no Alpaca account or network.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import backtest as bt  # noqa: E402
import sparticus_strategy as strat  # noqa: E402
from sparticus_journal import REASON_MAP  # noqa: E402
from sparticus_strategy import LIVE_DEFAULTS, position_budget, position_size  # noqa: E402
from test_lab import DAYS, random_walk  # noqa: E402

MICRO = replace(LIVE_DEFAULTS, fractional=True, position_pct_of_equity=0.30, max_open_positions=3)


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


class _Side(str):
    """The stubbed broker enum is a plain string; the bot also reads side.value."""
    @property
    def value(self):
        return str(self)


BUY = _Side("buy")


def quiet_bot():
    from test_stats_journal import load_bot
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)   # keep the bot's log file out of your folder
        try:
            return load_bot()
        finally:
            os.chdir(cwd)


class TestSizing(unittest.TestCase):
    def test_fractional_share_of_equity(self):
        qty, why = position_size(70.0, 100.0, MICRO)              # $30 budget
        self.assertEqual(why, "")
        self.assertAlmostEqual(qty, 0.4285, places=4)
        self.assertLessEqual(qty * 70.0, 30.0)

    def test_expensive_stock_still_fits(self):
        qty, _ = position_size(1100.0, 100.0, MICRO)             # a $1,100 stock with $30
        self.assertGreater(qty, 0)
        self.assertLess(qty, 1)

    def test_under_the_broker_minimum_is_refused(self):
        qty, why = position_size(50.0, 3.0, MICRO)                # 30% of $3 = $0.90 < $1
        self.assertEqual(qty, 0.0)
        self.assertIn("minimum", why)

    def test_whole_shares_unchanged_by_default(self):
        whole = replace(LIVE_DEFAULTS, fractional=False, position_pct_of_equity=0.0)
        qty, _ = position_size(70.0, 100_000.0, whole)
        self.assertEqual(qty, int(whole.target_dollars_per_trade / 70.0))
        self.assertEqual(position_budget(100_000.0, whole), whole.target_dollars_per_trade)
        self.assertEqual(position_budget(100.0, MICRO), 30.0)


class TestMicroLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        syms = ["SPY"] + [f"S{i}" for i in range(9)]
        cls.bars = {s: random_walk(i + 1, vol=0.0015) for i, s in enumerate(syms)}
        cls.syms = syms
        cls.res = bt.simulate("micro100", MICRO, cls.bars, syms, DAYS[0], DAYS[-1], cost_bps=2.0,
                              start_equity=100.0, verbose=False, sell_fee=0.01)

    def test_positions_are_fractional_and_sized_to_equity(self):
        self.assertGreater(len(self.res.trades), 10)
        self.assertTrue(any(t.qty != int(t.qty) for t in self.res.trades))
        for t in self.res.trades:
            self.assertLessEqual(t.qty * t.entry_px, 0.30 * 130.0, t)   # 30% of (at most) ~$130 equity

    def test_every_sale_pays_a_penny(self):
        # fixed $30 positions, so the fee can't change the sizing: the only difference is the pennies
        fixed = replace(MICRO, position_pct_of_equity=0.0, target_dollars_per_trade=30.0)
        runs = [bt.simulate("micro100", fixed, self.bars, self.syms, DAYS[0], DAYS[-1], cost_bps=2.0,
                            start_equity=100.0, verbose=False, sell_fee=fee) for fee in (0.0, 0.01)]
        self.assertEqual(len(runs[0].trades), len(runs[1].trades))
        for a, b in zip(*(r.trades for r in runs)):
            self.assertAlmostEqual(a.pnl - b.pnl, 0.01, places=3)
            self.assertAlmostEqual(a.gross_pnl, b.gross_pnl, places=3)

    def test_experiments_registered_with_small_equity(self):
        self.assertEqual(bt.LAB_EQUITY["micro100"], 100.0)
        self.assertEqual(bt.LAB_EQUITY["micro20"], 20.0)
        self.assertTrue(bt.EXPERIMENTS["micro100"][0].fractional)
        self.assertIn("micro20", bt.EXPECT_SKIP)


class TestMicroEntry(unittest.TestCase):
    def _engine(self, bot, params):
        bot.STRATEGY = params
        eng = bot.SparticusEngineV55()
        sent = []
        eng.account_snapshot = types.SimpleNamespace(equity="100.00")
        eng.session_info = lambda: {"open": True, "mins_to_close": 200}
        eng._check_cooldown = lambda s: True
        eng._check_risk_gate = lambda *a: (True, "")
        eng.data_client = types.SimpleNamespace(
            get_stock_latest_quote=lambda req: {"KO": types.SimpleNamespace(bid_price=69.99, ask_price=70.01)})
        eng.trading_client = types.SimpleNamespace(
            submit_order=lambda r: sent.append(r) or types.SimpleNamespace(id="ord-1"))
        return eng, sent

    def test_fractional_entry_is_a_plain_day_limit(self):
        bot = quiet_bot()
        try:
            eng, sent = self._engine(bot, MICRO)
            eng.execute_entry("KO", BUY, "manual")
        finally:
            bot.STRATEGY = LIVE_DEFAULTS
        self.assertEqual(len(sent), 1)
        r = sent[0].__dict__
        self.assertNotIn("order_class", r)                  # no bracket: Alpaca refuses them on fractions
        self.assertEqual(r["time_in_force"], bot.TimeInForce.DAY)
        self.assertEqual(r["qty"], 0.4285)                  # $30 / $70.01, rounded down to 4 decimals
        self.assertEqual(r["limit_price"], 70.01)

    def test_whole_share_mode_keeps_its_bracket(self):
        bot = quiet_bot()
        eng, sent = self._engine(bot, replace(LIVE_DEFAULTS, target_dollars_per_trade=5000.0, fractional=False,
                                                position_pct_of_equity=0.0))
        eng.account_snapshot = types.SimpleNamespace(equity="100000")
        try:
            eng.execute_entry("KO", BUY, "manual")
        finally:
            bot.STRATEGY = LIVE_DEFAULTS
        r = sent[0].__dict__
        self.assertEqual(r["order_class"], bot.OrderClass.BRACKET)
        self.assertEqual(r["qty"], 71)

    def test_too_small_account_is_refused_not_sent(self):
        bot = quiet_bot()
        eng, sent = self._engine(bot, MICRO)
        eng.account_snapshot = types.SimpleNamespace(equity="2.00")
        try:
            eng.execute_entry("KO", BUY, "manual")
        finally:
            bot.STRATEGY = LIVE_DEFAULTS
        self.assertEqual(sent, [])
        self.assertIn("minimum", eng.get_status())


class FakeBroker:
    def __init__(self, orders=()):
        self.orders = list(orders)
        self.cancelled, self.submitted = [], []

    def get_orders(self, req):
        return [o for o in self.orders if o.id not in self.cancelled]

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def get_order_by_id(self, oid):
        return types.SimpleNamespace(status="canceled" if oid in self.cancelled else "new")

    def submit_order(self, r):
        self.submitted.append(r)
        return types.SimpleNamespace(id="new")


def stop(sym, qty, price, oid="s1", prefix="spx-protect-"):
    return types.SimpleNamespace(id=oid, symbol=sym, qty=str(qty), stop_price=str(price),
                                 client_order_id=f"{prefix}{sym}-1")


def pos(sym, qty, price, entry=70.0):
    return types.SimpleNamespace(symbol=sym, qty=str(qty), current_price=str(price), avg_entry_price=str(entry))


class TestFractionalProtection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = quiet_bot()

    def setUp(self):
        self.bot.STRATEGY = MICRO
        self.eng = self.bot.SparticusEngineV55()
        self.eng.journal.open["KO"] = {"tp": 72.80, "sl": 68.60}
        self.closed = []
        self.eng.close_position_safely = lambda *a: self.closed.append(a)

    def tearDown(self):
        self.bot.STRATEGY = LIVE_DEFAULTS

    def run_once(self, broker, positions):
        self.eng.trading_client = broker
        self.eng.ensure_fractional_protection(positions)
        import time
        time.sleep(0.05)    # exits run on a thread

    def test_places_a_stop_for_the_full_fractional_size(self):
        b = FakeBroker()
        self.run_once(b, {"KO": pos("KO", 0.4284, 70.10)})
        r = b.submitted[0].__dict__
        self.assertEqual((r["qty"], r["stop_price"], r["side"]), (0.4284, 68.60, self.bot.OrderSide.SELL))
        self.assertEqual(r["time_in_force"], self.bot.TimeInForce.DAY)
        self.assertTrue(r["client_order_id"].startswith(self.bot.PROTECT_PREFIX))
        self.assertFalse(r["client_order_id"].startswith(self.bot.ORDER_PREFIX))   # never mistaken for an entry

    def test_leaves_a_correct_stop_alone(self):
        b = FakeBroker([stop("KO", 0.4284, 68.60)])
        self.run_once(b, {"KO": pos("KO", 0.4284, 70.10)})
        self.assertEqual((b.submitted, b.cancelled), ([], []))

    def test_replaces_a_stop_when_the_size_changes(self):
        b = FakeBroker([stop("KO", 0.2, 68.60)])
        self.run_once(b, {"KO": pos("KO", 0.4284, 70.10)})
        self.assertEqual(b.cancelled, ["s1"])
        self.assertEqual(b.submitted[0].__dict__["qty"], 0.4284)

    def test_take_profit_sells(self):
        b = FakeBroker([stop("KO", 0.4284, 68.60)])
        self.run_once(b, {"KO": pos("KO", 0.4284, 72.85)})
        self.assertEqual(self.closed, [("KO", "take profit", "auto")])
        self.assertEqual(b.submitted, [])

    def test_software_stop_sells_if_the_broker_stop_is_missing(self):
        b = FakeBroker()
        self.run_once(b, {"KO": pos("KO", 0.4284, 68.50)})
        self.assertEqual(self.closed, [("KO", "software stop", "auto")])

    def test_waits_while_the_entry_is_still_working(self):
        b = FakeBroker([stop("KO", 0.3, 70.01, oid="e1", prefix="sparticus-manual-")])
        self.run_once(b, {"KO": pos("KO", 0.2, 70.10)})
        self.assertEqual(b.submitted, [])

    def test_skips_symbols_being_closed(self):
        b = FakeBroker()
        self.eng.closing_symbols.add("KO")
        self.run_once(b, {"KO": pos("KO", 0.4284, 72.85)})
        self.assertEqual((b.submitted, self.closed), ([], []))

    def test_falls_back_to_bracket_levels_without_a_journal_row(self):
        self.eng.journal.open.clear()
        b = FakeBroker()
        self.run_once(b, {"KO": pos("KO", 0.5, 70.0, entry=70.0)})
        tp, sl = strat.bracket_levels(70.0, True, MICRO)
        self.assertEqual(b.submitted[0].__dict__["stop_price"], sl)

    def test_exit_reasons_are_labelled_in_the_journal(self):
        self.assertEqual(REASON_MAP["take profit"], "target")
        self.assertEqual(REASON_MAP["software stop"], "stop")


class TestIncubationScaling(unittest.TestCase):
    def test_percent_sizing_baseline_scales_to_the_account(self):
        import sparticus_incubation as inc
        expect = {"per_trade_mean": 0.02, "per_trade_std": 0.10, "mc_dd95": -5.0,
                  "lab_equity": 100.0, "pct_sizing": True}
        out = inc.evaluate({"trades": 3, "max_drawdown": -1.2, "expectancy": 0.0}, expect, 20.0)
        self.assertAlmostEqual(out["kill_at"], -1.0)          # $5 on $100 -> $1 on $20
        self.assertTrue(out["kill"])

    def test_fixed_dollar_baseline_is_not_scaled(self):
        import sparticus_incubation as inc
        expect = {"per_trade_mean": 2.0, "per_trade_std": 30.0, "mc_dd95": -900.0}
        out = inc.evaluate({"trades": 3, "max_drawdown": -100.0, "expectancy": 0.0}, expect, 50_000.0)
        self.assertEqual(out["kill_at"], -900.0)

    def test_expectation_file_records_the_lab_account_size(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            bt.write_expectations(Path(tmp), "micro100", "abcd1234", MICRO, {"expectancy": 0.01, "std": 0.2},
                                  (-2.0, -6.0), (0.5, 4), False, 100.0)
            data = json.loads((Path(tmp) / "expect_abcd1234.json").read_text())
        self.assertEqual((data["lab_equity"], data["pct_sizing"]), (100.0, True))


class TestMicroStartup(unittest.TestCase):
    def test_fractionable_filter_keeps_order_and_skips_failures(self):
        bot = quiet_bot()
        assets = {"AAPL": True, "TLT": False, "KO": True}

        def get_asset(sym):
            if sym not in assets:
                raise RuntimeError("asset not found")
            return types.SimpleNamespace(fractionable=assets[sym])
        keep, skipped = bot.fractionable_only(types.SimpleNamespace(get_asset=get_asset), ["KO", "TLT", "ZZZ", "AAPL"])
        self.assertEqual(keep, ["KO", "AAPL"])
        self.assertEqual(skipped, ["TLT", "ZZZ"])

    def test_fractional_shorts_refuse_to_start(self):
        saved = strat.LIVE_DEFAULTS
        strat.LIVE_DEFAULTS = replace(MICRO, enable_shorts=True)
        try:
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                quiet_bot()
            self.assertEqual(cm.exception.code, 4)
        finally:
            strat.LIVE_DEFAULTS = saved

    def test_upgrading_keeps_the_v62_config_id(self):
        # Settings added after V6.2, left at their defaults, must not change the id (no restarted
        # incubation). Checked against your own settings, whatever they are.
        import hashlib
        import json
        from dataclasses import asdict
        base = replace(LIVE_DEFAULTS, **strat._ADDED_LATER)
        d = asdict(base)
        for k in strat._ADDED_LATER:
            d.pop(k)
        v62 = hashlib.sha256(json.dumps({"p": d, "u": sorted(strat.DEFAULT_UNIVERSE)}, sort_keys=True)
                             .encode()).hexdigest()[:8]
        self.assertEqual(strat.config_id(base), v62)
        self.assertNotEqual(strat.config_id(replace(base, fractional=True, position_pct_of_equity=0.3)), v62)


if __name__ == "__main__":
    unittest.main()
