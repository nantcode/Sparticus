"""V6.4 tests: never left holding positions (flatten on quit, carried positions sold after the open),
the optimizer's verdict, and the autostart / optimize scripts. Offline, in a throwaway folder.

    python3 -m unittest discover -s tests -v
"""
import datetime as dt
import json
import os
import plistlib
import re
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import backtest as bt  # noqa: E402
import lab_verdict  # noqa: E402
from sparticus_strategy import DEFAULT_UNIVERSE, LIVE_DEFAULTS, config_id  # noqa: E402
from test_stats_journal import load_bot  # noqa: E402

ET = ZoneInfo("America/New_York")
_SAFE_CWD = []


def setUpModule():
    _SAFE_CWD.append((os.getcwd(), tempfile.TemporaryDirectory()))
    os.chdir(_SAFE_CWD[-1][1].name)


def tearDownModule():
    cwd, tmp = _SAFE_CWD.pop()
    os.chdir(cwd)
    tmp.cleanup()


def order(side, filled="10"):
    return types.SimpleNamespace(side=side, filled_qty=filled)


class TestCarriedPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot()

    def setUp(self):
        self.eng = self.bot.SparticusEngineV55()
        close = dt.datetime(2026, 10, 23, 16, 0, tzinfo=ET)
        self.eng.get_clock = lambda: types.SimpleNamespace(is_open=True, next_close=close)
        self.today_orders = {"AMZN": [order("buy")], "SMCI": [order("sell")], "META": []}
        self.asked_after = []

        def get_orders(req):
            self.asked_after.append(req.after)
            return self.today_orders[req.symbols[0]]
        self.eng.trading_client = types.SimpleNamespace(get_orders=get_orders)
        self.closed = []
        self.done = threading.Event()

        def close_position_safely(sym, reason, source):
            self.closed.append((sym, reason, source))
            if len(self.closed) == 2:
                self.done.set()
        self.eng.close_position_safely = close_position_safely
        self.positions = {s: types.SimpleNamespace(symbol=s, qty="10") for s in ("AMZN", "SMCI", "META")}

    def test_positions_without_a_buy_today_are_carried(self):
        carried, complete = self.eng.find_carried_positions(self.positions)
        self.assertTrue(complete)
        self.assertEqual(sorted(carried), ["META", "SMCI"])
        self.assertEqual(self.asked_after[0].hour, 4)    # counts fills since 4:00 AM ET, premarket included

    def test_sold_once_after_the_open_not_before(self):
        sess = {"open": True, "mins_since_open": 0.5, "mins_to_close": 389}
        self.bot._check_carried(self.eng, self.positions, sess)
        self.assertEqual(self.closed, [])                # still inside the opening auction minute
        sess["mins_since_open"] = 2.0
        self.bot._check_carried(self.eng, self.positions, sess)
        self.assertTrue(self.done.wait(2))
        self.assertEqual(sorted(c[0] for c in self.closed), ["META", "SMCI"])
        self.assertTrue(all(c[1] == "carried overnight" for c in self.closed))
        self.bot._check_carried(self.eng, self.positions, sess)   # once a day only
        self.assertEqual(len(self.closed), 2)

    def test_a_failed_lookup_is_retried_not_guessed(self):
        def boom(req):
            raise ConnectionError("api down")
        self.eng.trading_client = types.SimpleNamespace(get_orders=boom)
        sess = {"open": True, "mins_since_open": 5.0, "mins_to_close": 300}
        self.bot._check_carried(self.eng, self.positions, sess)
        self.assertEqual(self.closed, [])
        self.assertIsNone(self.eng.carried_check_day)
        self.assertGreater(self.eng.carried_next_try, 0)

    def test_reason_is_labelled_in_the_journal(self):
        from sparticus_journal import REASON_MAP
        self.assertEqual(REASON_MAP["carried overnight"], "carried")


class TestFlattenBeforeQuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = load_bot()

    def make(self, market_open, held):
        eng = self.bot.SparticusEngineV55()
        eng.session_info = lambda: {"open": market_open, "mins_to_close": 27, "mins_since_open": 363}
        eng.get_positions_copy = lambda: {s: object() for s in held}
        calls = []
        eng.flatten_all_positions = lambda reason, disable_autopilot=False: calls.append(
            (reason, disable_autopilot)) or True
        return eng, calls

    def test_closing_the_window_during_market_hours_flattens_first(self):
        eng, calls = self.make(True, ["AMZN", "META", "GOOGL"])
        self.bot.flatten_before_quit(eng, "HUD window closed")
        self.assertEqual(calls, [("stopped (HUD window closed)", True)])

    def test_nothing_to_do_after_hours_or_when_flat(self):
        for market_open, held in ((False, ["AMZN"]), (True, [])):
            eng, calls = self.make(market_open, held)
            self.bot.flatten_before_quit(eng, "Ctrl+C")
            self.assertEqual(calls, [])

    def test_can_be_switched_off(self):
        eng, calls = self.make(True, ["AMZN"])
        saved = self.bot.FLATTEN_ON_QUIT
        self.bot.FLATTEN_ON_QUIT = False
        try:
            self.bot.flatten_before_quit(eng, "Ctrl+C")
        finally:
            self.bot.FLATTEN_ON_QUIT = saved
        self.assertEqual(calls, [])

    def test_closing_terminal_becomes_a_clean_stop(self):
        import signal
        with self.assertRaises(self.bot.StopRequested) as cm:
            self.bot._on_stop_signal(signal.SIGHUP, None)
        self.assertEqual(str(cm.exception), "Terminal closed")


class TestLabVerdict(unittest.TestCase):
    def write(self, folder, name, passed, t):
        p, _ = bt.EXPERIMENTS[name]
        cid = config_id(p, DEFAULT_UNIVERSE)
        (folder / f"expect_{cid}.json").write_text(json.dumps(
            {"passed": passed, "t_stat": t, "oos_trades": 250, "per_trade_mean": 0.5, "profit_factor": 1.3,
             "months_positive": 0.67}))

    def test_nothing_passed_means_keep_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.write(d, "live", False, -0.8)
            self.write(d, "live_hold", False, 1.1)
            text = lab_verdict.verdict(d, ["live", "live_hold"])
        self.assertIn("Nothing passed the bar", text)
        self.assertIn("(your settings)", text)

    def test_a_pass_names_the_exact_lines_to_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.write(d, "live", False, -0.8)
            self.write(d, "live_hold", True, 3.4)
            text = lab_verdict.verdict(d, ["live", "live_hold"])
        self.assertIn("Best config that passed: live_hold", text)
        self.assertIn("signal_exit: True -> False", text)

    def test_lab_only_winner_is_not_recommended_for_the_live_bot(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.write(d, "orb", True, 3.0)
            text = lab_verdict.verdict(d, ["orb"])
        self.assertIn("lab-only", text)
        self.assertNotIn("change these lines", text)

    def test_no_results_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("Run ./optimize.sh first", lab_verdict.verdict(Path(tmp)))

    def test_optimize_runs_only_real_experiments(self):
        text = (ROOT / "optimize.sh").read_text()
        names = re.search(r'EXPERIMENTS="([^"]+)"', text).group(1).split(",")
        self.assertTrue(all(n in bt.EXPERIMENTS for n in names), names)


class TestScripts(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for name in ("run_sparticus.sh", "optimize.sh", "install_autostart.sh", "uninstall_autostart.sh"):
            r = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")

    def test_autostart_writes_a_weekday_launchd_job(self):
        text = (ROOT / "install_autostart.sh").read_text()
        code = text.split("<<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            plist = Path(tmp) / "job.plist"
            subprocess.run([sys.executable, "-c", code, str(plist), "/Users/me/flyplayground", "/usr/bin/python3",
                            "9", "20", "com.sparticus.autostart"], check=True)
            job = plistlib.loads(plist.read_bytes())
        self.assertEqual(job["ProgramArguments"], ["/bin/bash", "/Users/me/flyplayground/run_sparticus.sh"])
        self.assertEqual([d["Weekday"] for d in job["StartCalendarInterval"]], [1, 2, 3, 4, 5])
        self.assertTrue(all((d["Hour"], d["Minute"]) == (9, 20) for d in job["StartCalendarInterval"]))
        self.assertEqual(job["EnvironmentVariables"]["SPARTICUS_PYTHON"], "/usr/bin/python3")

    def test_launcher_uses_the_remembered_python(self):
        text = (ROOT / "run_sparticus.sh").read_text()
        self.assertIn('PY="${SPARTICUS_PYTHON:-python3}"', text)
        self.assertIn('BOT="sparticus_v6_4.py"', text)


if __name__ == "__main__":
    unittest.main()
