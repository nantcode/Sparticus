"""
Sparticus V6.4 - Autonomous entries AND exits, with full manual override.

V6.4 (it never gets left holding positions; trading rules unchanged):
  * Stopping Sparticus during market hours (closing the HUD window, Ctrl+C, closing Terminal) now
    closes its positions first. Sparticus never holds overnight, but on Oct 22 the window closed at
    3:33 PM with four positions open and nothing flattened them at 3:55.
  * Positions left over from an earlier day (the bot was off at 3:55, or the Mac slept) are sold
    right after the next open instead of only triggering an alert.
  * If it still stops with positions open (it can't flatten), Discord gets a loud warning.
  * Log file: sparticus_v6_4.log.

V6.3.2 = the fixes below (your Oct 7 build) + the V6.3.1 package (trade_review.py, two lab experiments,
tests that never touch your folder). Trading rules unchanged.

Fixes on top of V6.3:
  * trip_circuit_breaker() was called by the watchdog but never defined. When the day's loss passed
    the limit the call raised AttributeError, which skipped the rest of that watchdog cycle: the -3%
    stop, the micro-mode stop upkeep, the end-of-day flatten and stale-order cleanup all stopped, and
    the breaker never tripped, so entries continued. It is now defined (flatten + halt for the day,
    flatten retried every 30 s until the account is flat).
  * The watchdog now runs each stage on its own (watchdog_cycle), so one stage failing can never
    switch off the others again.
  * Secrets are redacted from every log line and Discord post: Discord failures used to write the
    full webhook URL (token included) into the log file.
  * An empty watchlist (e.g. micro mode finds nothing fractionable) exits with a clear message instead
    of crashing the HUD.
  * New flag --no-autopilot: start with autopilot OFF, watch the HUD fill in, press [A] to enable.
  * Log file is sparticus_v6_4.log (older versions keep their own logs).

V6.3 (micro mode, for small accounts): set fractional=True and position_pct_of_equity in
sparticus_strategy.py. Fractional shares can't use bracket orders, so each fill gets a broker-side
stop order (protection even if this Mac sleeps) and Sparticus manages the take-profit itself.
Long only. Tickers Alpaca can't trade fractionally are skipped. Off by default: nothing changes
unless you switch it on.

V6.2 (forward-testing release):
  * Every journal row is tagged with the strategy and a config id (a fingerprint of every
    setting), so stats can be read per config instead of mixing several changes together.
  * Incubation: the STATS page and the daily summary compare this config's live trades with what
    the lab expects from the exact same config (lab_results/expect_<config_id>.json).
  * Strategy kill switch: if this config's drawdown passes the lab's 95% worst case (or 3% of
    equity with no lab baseline), autopilot turns off and stays off across restarts until you
    delete sparticus_killed.json. Positions keep their brackets and the 15:55 flatten.
  * Brackets come from sparticus_strategy.bracket_levels: fixed % (as before) or volatility-scaled
    ("atr" mode: target = reward_risk x stop, with a minimum stop). Trading rules are otherwise unchanged.

V6.1: trade journal (sparticus_journal.csv: decision price, spread, slippage, best/worst move
while open, exit reason, signal strength) and a STATS page in the HUD ([TAB]) showing the edge,
whether it's real (t-stat), where money leaks, and risk. Trading behaviour is unchanged.

V6.0: the strategy rules now live in sparticus_strategy.py, shared with the backtester
(backtest.py), so a lab result describes exactly what this bot trades. Edit the rules in
sparticus_strategy.py (LIVE_DEFAULTS); this file reads them. Trading behaviour is unchanged.
V5.5.4 (HUD release, trading rules unchanged): new 1280x800 "brain" HUD. Brain map of every
ticker coloured by trend strength, thought stream, live chart with 5/20 SMA and entry/TP/SL,
RSI panel, decision pipeline (every gate the autopilot checks, live) and today's equity curve.
Click a tile or use the arrow keys to select. Drawing errors never stop trading.
V5.5.3 (reliability release, trading rules unchanged):
  * Partly filled entries keep their protection: when the unfilled rest is cancelled, a new
    stop-loss + take-profit (OCO) is attached to the shares that did fill, or they're closed.
  * Health monitor: alerts if the scanner or watchdog stops or stalls; in --headless mode it
    exits so run_sparticus.sh restarts the bot.
  * Startup report of positions carried over from before the bot started.
  * Daily summary posted to Discord after the end-of-day flatten (green or red, entries, day trades).
  * Optional experiment setting: MAX_ENTRIES_PER_SYMBOL_PER_DAY (0 = off, same as before).
  * Config errors exit with code 4 so run_sparticus.sh doesn't restart-loop on them.
V5.5.2: the end-of-day flatten verifies the account is actually flat and retries until it is.
V5.5.1: everything the bot logs is also posted to Discord, in order, batched to respect
Discord rate limits, with repeats collapsed. Fills and exit P&L are now reported too.
Missing or broken webhook URLs are reported loudly at startup instead of failing silently.

What the autopilot does on its own:
  * Enters on a FRESH 5/20 SMA crossover + RSI confirmation (not "any time fast > slow").
  * Exits on trend reversal (signal exit), on top of the bracket TP/SL.
  * Cancels stale unfilled entry orders.
  * Flattens everything before the close (no naked overnight positions).
  * Flattens + halts when the daily circuit breaker trips.
  * Enforces re-entry cooldowns so it doesn't churn the same ticker after an exit.
  * Can run headless: python sparticus_v6_4.py --headless

Manual control (HUD):
  UP/DOWN  select ticker        B  manual BUY          S  manual SHORT (if ENABLE_SHORTS)
  C        close selected       L  lock/unlock ticker (autopilot hands-off)
  A        autopilot on/off     Shift+X  PANIC flatten all (also turns autopilot off)

Manual positions are never signal-exited by the autopilot; they still get the bracket
TP/SL, the -3% watchdog, the circuit breaker and the end-of-day flatten.
"""
import os
import re
import sys
import time
import datetime
import threading
import logging
import atexit
import signal
import argparse
import collections
import math
from typing import Dict, Any, Optional, List, Tuple

import dotenv
import requests

from sparticus_strategy import (DEFAULT_UNIVERSE, LIVE_DEFAULTS as STRATEGY, LIVE_STRATEGIES, atr,
                                bracket_levels, calculate_rsi, compute_signal, config_id,
                                min_bars as strategy_min_bars, position_budget, position_size, sma)
import sparticus_incubation as incubation
import sparticus_stats as stats
from sparticus_journal import TradeJournal

BOT_VERSION = "V6.4"
CONFIG_ID = config_id(STRATEGY, DEFAULT_UNIVERSE)
JOURNAL_PATH = "sparticus_journal.csv"
EXPECT_DIR = "lab_results"             # where backtest.py writes expect_<config_id>.json
KILL_FILE = "sparticus_killed.json"    # present = this config was switched off by the kill switch
STATS_REFRESH_SECONDS = 60

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest, StopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ==========================================
# 0. SETUP & CONFIGURATION
# ==========================================
dotenv.load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
PAPER_TRADING = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
DATA_FEED = DataFeed.SIP if os.getenv("ALPACA_DATA_FEED", "iex").strip().lower() == "sip" else DataFeed.IEX

if not API_KEY or not SECRET_KEY:
    print("FATAL: ALPACA_API_KEY or ALPACA_SECRET_KEY missing from environment.")
    sys.exit(4)  # config error: run_sparticus.sh won't restart-loop on this

# --- Strategy settings: edit them in sparticus_strategy.py (LIVE_DEFAULTS), not here ---
if STRATEGY.strategy not in LIVE_STRATEGIES:
    print(f"FATAL: strategy {STRATEGY.strategy!r} is lab-only for now (the live bot trades "
          f"{', '.join(LIVE_STRATEGIES)}). Reset it in sparticus_strategy.py LIVE_DEFAULTS.")
    sys.exit(4)
if STRATEGY.fractional and STRATEGY.enable_shorts:
    print("FATAL: fractional shares can't be sold short. Set enable_shorts=False or fractional=False.")
    sys.exit(4)
if STRATEGY.bar_minutes != 1 or STRATEGY.market_filter != "none":
    print("FATAL: bar_minutes != 1 and market filters are lab-only experiments; the live bot "
          "doesn't support them yet. Reset them in sparticus_strategy.py LIVE_DEFAULTS.")
    sys.exit(4)
if STRATEGY.hard_stop_pct >= 0 or STRATEGY.daily_loss_limit_pct >= 0:
    # Both are compared with "<=": a positive value would liquidate everything / trip the breaker at once.
    print("FATAL: hard_stop_pct and daily_loss_limit_pct must be NEGATIVE (e.g. -0.03 and -0.05). "
          "Fix them in sparticus_strategy.py LIVE_DEFAULTS.")
    sys.exit(4)

# --- Allocation & Risk Controls ---
TARGET_DOLLARS_PER_TRADE = STRATEGY.target_dollars_per_trade
MAX_NOTIONAL_PER_ORDER = TARGET_DOLLARS_PER_TRADE * 1.10
TAKE_PROFIT_PCT = STRATEGY.take_profit_pct
STOP_LOSS_PCT = STRATEGY.stop_loss_pct
HARD_WATCHDOG_LOSS_PCT = STRATEGY.hard_stop_pct  # emergency stop per position
MAX_DAILY_DRAWDOWN_PCT = STRATEGY.daily_loss_limit_pct  # daily circuit breaker
MAX_GROSS_EXPOSURE_MULT = 1.5   # gross exposure cap vs equity
ENTRY_COOLDOWN_SECONDS = 15.0

# --- New V5.5 risk controls ---
MAX_OPEN_POSITIONS = STRATEGY.max_open_positions  # positions + pending entries
MAX_SPREAD_PCT = 0.003          # skip entry if bid/ask spread > 0.3%
STALE_ENTRY_SECONDS = 120       # cancel unfilled entry orders after 2 min
FLATTEN_AT_EOD = True           # flatten everything before the close
EOD_FLATTEN_MINUTES = STRATEGY.eod_flatten_minutes  # ...this many minutes before the close
FLATTEN_ON_CIRCUIT_BREAKER = True
FLATTEN_ON_QUIT = True          # V6.4: stopping Sparticus during market hours closes its positions first
CLOSE_CARRIED_AT_OPEN = True    # V6.4: positions left over from an earlier day are sold after the open
CARRIED_CHECK_AFTER_MIN = 1.0   # ...this many minutes after 9:30 (after the opening auction)
CB_FLATTEN_RETRY_SECONDS = 30   # after the breaker trips, re-flatten this often until the account is flat

# --- Autopilot strategy settings ---
AUTOPILOT_ON_START = True
ENABLE_SHORTS = STRATEGY.enable_shorts  # True = autopilot shorts on bearish crosses + manual [S] works
SCAN_INTERVAL_SECONDS = 60
FAST_SMA_PERIOD = STRATEGY.fast_sma
SLOW_SMA_PERIOD = STRATEGY.slow_sma
RSI_PERIOD = STRATEGY.rsi_period
MIN_RSI_THRESHOLD = STRATEGY.min_rsi
MAX_RSI_ENTRY = STRATEGY.max_rsi
REQUIRE_FRESH_CROSS = STRATEGY.require_fresh_cross
CROSS_LOOKBACK_BARS = STRATEGY.cross_lookback
SIGNAL_EXIT_ENABLED = STRATEGY.signal_exit
MIN_HOLD_SECONDS = STRATEGY.min_hold_minutes * 60
REENTRY_COOLDOWN_MINUTES = STRATEGY.reentry_cooldown_minutes
MANUAL_CLOSE_BLOCK_MINUTES = 60 # after YOU close a ticker, autopilot leaves it alone this long
NO_ENTRY_FIRST_MINUTES = STRATEGY.no_entry_first_minutes
NO_ENTRY_LAST_MINUTES = STRATEGY.no_entry_last_minutes
MAX_BAR_AGE_SECONDS = STRATEGY.max_bar_age_minutes * 60
BAR_LOOKBACK_MINUTES = STRATEGY.lookback_minutes

WATCHDOG_INTERVAL_SECONDS = 5

# --- V5.5.3 reliability settings ---
PROTECT_PREFIX = "spx-protect-"      # protective orders; must NOT start with ORDER_PREFIX
HEALTH_CHECK_SECONDS = 30
SCANNER_STALL_ALERT_SECONDS = 300    # alert if no scan for 5 min during market hours
WATCHDOG_STALL_ALERT_SECONDS = 120   # alert if the watchdog (stops, flatten) is silent for 2 min
RESTART_ON_STALL_SECONDS = 900       # --headless only: exit after 15 min stuck so the runner restarts

# --- Experiments (change ONE at a time; the defaults keep V5.5 behaviour) ---
MAX_ENTRIES_PER_SYMBOL_PER_DAY = STRATEGY.max_entries_per_symbol_per_day
CLOCK_CACHE_SECONDS = 20
LOCK_FILE_PATH = "sparticus.lock"
ORDER_PREFIX = "sparticus-"

RECENT_EVENTS: collections.deque = collections.deque(maxlen=40)

# --- Secret redaction (V6.3.1): nothing sensitive reaches the log file, the HUD or Discord ---
#   matches the path only on purpose: requests errors print the host and the path apart
#   ("host='discord.com' ... url: /api/webhooks/<id>/<token>"), so a full-URL pattern would miss them.
_WEBHOOK_RE = re.compile(r"(/api/webhooks/)\d+/[A-Za-z0-9_\-]+")


def redact(text: Any) -> str:
    """Strip Discord webhook tokens and the Alpaca keys from any text before it's logged or posted."""
    s = _WEBHOOK_RE.sub(r"\1<redacted>", str(text))
    for secret in (API_KEY, SECRET_KEY, DISCORD_WEBHOOK_URL):
        if secret and len(secret) > 6:
            s = s.replace(secret, "<redacted>")
    return s


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("sparticus_v6_4.log"), logging.StreamHandler(sys.stdout)]
)
for _h in logging.getLogger().handlers:
    _h.addFilter(_RedactFilter())


DISCORD_MIN_LEVEL = logging.INFO   # raise to logging.WARNING to post only warnings and errors
DISCORD_FLUSH_SECONDS = 2.0        # batch window: at most one Discord message per window
DISCORD_MAX_CHARS = 1900           # Discord's hard limit is 2000 characters per message
_LEVEL_ICON = {logging.WARNING: "⚠️ ", logging.ERROR: "❌ ", logging.CRITICAL: "🚨 "}


class DiscordRelay:
    """Posts log lines to a Discord webhook from ONE background thread, in order.

    V5.5 started a new thread per message: bursts (EOD flatten, scanner signals) hit Discord's
    rate limit, the 429 responses were never checked, and messages vanished silently.
    This relay batches lines every couple of seconds, collapses repeats, honours 429
    retry_after, and flushes whatever is left when the bot shuts down.
    """

    def __init__(self, url: str):
        self.url = url
        self.enabled = bool(url)
        self._q: "collections.deque[Tuple[str, str]]" = collections.deque(maxlen=5000)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._recent: Dict[str, float] = {}
        self.dropped = 0
        if self.enabled:
            threading.Thread(target=self._run, name="discord-relay", daemon=True).start()

    def post(self, stamp: str, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append((stamp, text))
        self._wake.set()

    def post_once(self, stamp: str, text: str, window_s: float = 15.0) -> None:
        """Post unless the same text was posted in the last `window_s` seconds
        (stops an error that repeats every watchdog cycle from flooding the channel)."""
        now = time.time()
        with self._lock:
            last = self._recent.get(text)
            self._recent[text] = now
            if len(self._recent) > 500:
                self._recent = {k: v for k, v in self._recent.items() if now - v < window_s}
        if last is None or now - last > window_s:
            self.post(stamp, text)

    def _take_batch(self) -> List[str]:
        with self._lock:
            items = list(self._q)
            self._q.clear()
        # collapse consecutive repeats of the same text: "X" five times -> "X (x5)"
        out: List[str] = []
        i = 0
        while i < len(items):
            stamp, text = items[i]
            j = i
            while j + 1 < len(items) and items[j + 1][1] == text:
                j += 1
            n = j - i + 1
            out.append(f"`{stamp}` {text}" + (f"  (x{n})" if n > 1 else ""))
            i = j + 1
        return out

    def _chunks(self, lines: List[str]) -> List[str]:
        chunks, cur = [], ""
        for ln in lines:
            ln = ln if len(ln) <= DISCORD_MAX_CHARS else ln[:DISCORD_MAX_CHARS - 1] + "…"
            if cur and len(cur) + 1 + len(ln) > DISCORD_MAX_CHARS:
                chunks.append(cur)
                cur = ln
            else:
                cur = f"{cur}\n{ln}" if cur else ln
        if cur:
            chunks.append(cur)
        return chunks

    def _send(self, content: str) -> bool:
        """True when Discord accepted the message (or rejected it permanently)."""
        for _ in range(5):
            try:
                r = requests.post(self.url, json={"content": content,
                                                  "allowed_mentions": {"parse": []}}, timeout=5)
            except Exception as e:
                logging.error(f"[DISCORD] Post failed: {redact(e)}")   # the URL in the error carries the token
                time.sleep(3)
                continue
            if r.status_code == 429:
                try:
                    wait = float(r.json().get("retry_after", 2.0))
                except Exception:
                    wait = 2.0
                time.sleep(min(max(wait, 0.5), 30.0))
                continue
            if r.status_code >= 400:
                logging.error(f"[DISCORD] Webhook returned HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code in (401, 403, 404):
                    logging.error("[DISCORD] The webhook URL is invalid or was deleted. "
                                  "Create a new webhook in Discord and update DISCORD_WEBHOOK_URL in .env.")
                    self.enabled = False
                return True
            return True
        return False

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=DISCORD_FLUSH_SECONDS)
            self._wake.clear()
            time.sleep(DISCORD_FLUSH_SECONDS)  # let a burst accumulate into one message
            self.flush()

    def flush(self) -> None:
        lines = self._take_batch()
        if self.dropped:
            lines.insert(0, f"⚠️ {self.dropped} older lines were skipped (Discord couldn't keep up).")
            self.dropped = 0
        for chunk in self._chunks(lines):
            if not self.enabled or not self._send(chunk):
                break

    def shutdown(self, timeout_s: float = 8.0) -> None:
        """Called at exit: push whatever is still queued."""
        self._stop = True
        deadline = time.time() + timeout_s
        while self.enabled and self._q and time.time() < deadline:
            self.flush()

    def self_test(self) -> None:
        """Report missing or broken webhook configuration at startup."""
        if not self.enabled:
            logging.warning("[DISCORD] DISCORD_WEBHOOK_URL is not set in .env, so nothing will be "
                            "posted to Discord. Add it and restart.")
            return
        if not self.url.startswith("https://discord.com/api/webhooks/") and \
           not self.url.startswith("https://discordapp.com/api/webhooks/"):
            logging.warning("[DISCORD] DISCORD_WEBHOOK_URL doesn't look like a Discord webhook URL.")
        try:
            r = requests.get(self.url, timeout=5)  # GET on a webhook returns its info; no message sent
            if r.status_code == 200:
                logging.info("[DISCORD] Webhook OK.")
            else:
                logging.error(f"[DISCORD] Webhook check failed: HTTP {r.status_code}. "
                              f"Messages won't arrive until DISCORD_WEBHOOK_URL is fixed.")
        except Exception as e:
            logging.error(f"[DISCORD] Webhook check failed: {redact(e)}")


DISCORD = DiscordRelay(DISCORD_WEBHOOK_URL)
atexit.register(DISCORD.shutdown)


def fractionable_only(trading_client: Any, symbols: List[str]) -> Tuple[List[str], List[str]]:
    """Micro mode: keep the tickers Alpaca lets us trade in fractions (watchlist order kept).
    A ticker whose asset lookup fails is skipped too: better one ticker fewer than a rejected order."""
    keep, skipped = [], []
    for sym in symbols:
        try:
            ok = bool(getattr(trading_client.get_asset(sym), "fractionable", False))
        except Exception:
            ok = False
        (keep if ok else skipped).append(sym)
    return keep, skipped


def log_event(message: str, level: int = logging.INFO) -> None:
    """Log to file + console + HUD, and mirror to Discord."""
    message = redact(message)
    logging.log(level, message)
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    RECENT_EVENTS.append(f"{stamp} {message}")
    if level >= DISCORD_MIN_LEVEL:
        DISCORD.post_once(stamp, f"{_LEVEL_ICON.get(level, '')}{message}")


def send_discord_alert(message: str) -> None:
    """Kept for compatibility. Most callers also call log_event with the same text,
    which already posts it, so this only posts messages that weren't logged."""
    message = redact(message)
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    if any(message in ev for ev in list(RECENT_EVENTS)[-3:]):
        return
    DISCORD.post_once(stamp, message)


def acquire_single_instance_lock(lock_path: str = LOCK_FILE_PATH) -> None:
    """Refuses execution if another instance is already active."""
    pid = os.getpid()
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            print(f"FATAL: Sparticus is already running (PID {old_pid}). "
                  f"Stop that process first, or remove {lock_path} if it crashed.")
            sys.exit(4)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            log_event("[LOCK] Stale lock file found. Reclaiming it.")
    with open(lock_path, "w") as f:
        f.write(str(pid))
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))


# ==========================================
# 1. TECHNICAL INDICATORS
# ==========================================
# sma() and calculate_rsi() now come from sparticus_strategy.py (shared with the backtester).


# ==========================================
# 2. CORE ENGINE & RISK GATE
# ==========================================
class SparticusEngineV55:
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        self.state_lock = threading.RLock()
        self.in_flight_symbols = set()      # entries being submitted
        self.closing_symbols = set()        # exits being processed
        self.last_entry_attempt: Dict[str, float] = {}
        self.reentry_block_until: Dict[str, float] = {}
        self.entry_times: Dict[str, float] = {}
        self.manual_symbols = set()         # positions YOU opened -> no auto signal exits
        self.locked_symbols = set()         # autopilot hands-off (no auto entries / exits)
        self.signals: Dict[str, Dict[str, Any]] = {}

        self.autopilot_enabled = AUTOPILOT_ON_START
        self.cb_trip_day: Optional[str] = None
        self.cb_last_flatten = 0.0
        self.eod_flatten_day: Optional[str] = None
        self.overnight_alert_day: Optional[str] = None
        self._flattening = False
        self._protect_warned: Dict[str, float] = {}   # micro mode: symbol -> last stop-placement warning
        self.carried_check_day: Optional[str] = None   # V6.4: day the carried-position check last completed
        self.carried_next_try = 0.0
        self.entries_today: Dict[str, int] = {}
        self._entries_day: Optional[str] = None
        self.last_scan_ts = time.time()
        self.last_watchdog_ts = time.time()
        self.cached_session: Optional[Dict[str, Any]] = None      # HUD reads this, never the API
        self.journal = TradeJournal(JOURNAL_PATH, BOT_VERSION, STRATEGY.strategy, CONFIG_ID)
        self.expectations = incubation.load_expectations(EXPECT_DIR, CONFIG_ID)
        self.killed = False
        self.stats_cache: Dict[str, Any] = {}
        self.equity_history: "collections.deque[Tuple[float, float]]" = collections.deque(maxlen=2000)
        self._equity_day: Optional[str] = None

        self.account_snapshot = None
        self.positions_snapshot: Dict[str, Any] = {}
        self.daily_pnl_pct = 0.0
        self._snapshot_ready = False

        self._clock = None
        self._clock_fetched = 0.0
        self._et_tz = None

        self._status_message = "Engine initialized. Waiting for first scan..."

    # ---------- status ----------
    def set_status(self, message: str) -> None:
        with self.state_lock:
            self._status_message = message

    def get_status(self) -> str:
        with self.state_lock:
            return self._status_message

    # ---------- clock / session ----------
    def get_clock(self):
        """Cached market clock (one API call per CLOCK_CACHE_SECONDS max, even on failure)."""
        now = time.time()
        with self.state_lock:
            if now - self._clock_fetched < CLOCK_CACHE_SECONDS:
                return self._clock
            self._clock_fetched = now
        try:
            clock = self.trading_client.get_clock()
            with self.state_lock:
                self._clock = clock
                if clock.timestamp is not None and clock.timestamp.tzinfo is not None:
                    self._et_tz = clock.timestamp.tzinfo
            return clock
        except Exception as e:
            log_event(f"[ERROR] Could not fetch market clock: {e}", logging.WARNING)
            with self.state_lock:
                return self._clock

    def today_key(self) -> str:
        """Market-timezone date, no API call."""
        tz = self._et_tz
        if tz is not None:
            return datetime.datetime.now(tz).date().isoformat()
        return datetime.date.today().isoformat()

    def session_info(self) -> Dict[str, Any]:
        clock = self.get_clock()
        info = {"open": False, "mins_to_close": None, "mins_since_open": None}
        if clock is None or not clock.is_open:
            return info
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        info["open"] = True
        info["mins_to_close"] = (clock.next_close - now_utc).total_seconds() / 60.0
        session_open = clock.next_close.replace(hour=9, minute=30, second=0, microsecond=0)
        info["mins_since_open"] = (now_utc - session_open).total_seconds() / 60.0
        return info

    def session_text(self) -> str:
        """HUD-safe (no API call)."""
        clock = self._clock
        if clock is None:
            return "MARKET: ?"
        if not clock.is_open:
            return "MARKET: CLOSED"
        mins = (clock.next_close - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60.0
        return f"MARKET: OPEN ({max(0, int(mins))}m to close)"

    @property
    def circuit_breaker_tripped(self) -> bool:
        return self.cb_trip_day is not None and self.cb_trip_day == self.today_key()

    def trip_circuit_breaker(self, pnl_pct: float) -> None:
        """Daily loss limit hit: halt entries for the day and flatten everything.
        (V6.3 called this from the watchdog but never defined it.)"""
        with self.state_lock:
            self.cb_trip_day = self.today_key()
            self.autopilot_enabled = False
        msg = (f"🚨 [CIRCUIT BREAKER] Day P&L {pnl_pct * 100:.2f}% hit the {MAX_DAILY_DRAWDOWN_PCT * 100:.0f}% "
               f"limit. Autopilot OFF, entries halted for the day.")
        log_event(msg, logging.CRITICAL)
        self.set_status(msg)
        if FLATTEN_ON_CIRCUIT_BREAKER:
            self.cb_last_flatten = time.time()
            threading.Thread(target=self.flatten_all_positions, args=("circuit breaker", True),
                             daemon=True).start()

    # ---------- snapshots ----------
    def refresh_snapshot(self) -> Tuple[Any, Dict[str, Any]]:
        """Fetch account + positions, compute day P&L, detect positions that closed."""
        account = self.trading_client.get_account()
        positions = {p.symbol: p for p in self.trading_client.get_all_positions()}

        equity = float(account.equity)
        last_equity = float(account.last_equity or 0)  # Alpaca: equity at previous close
        with self.state_lock:
            day = self.today_key()
            if self._equity_day != day:
                self._equity_day = day
                self.equity_history.clear()
            if not self.equity_history or time.time() - self.equity_history[-1][0] >= 15:
                self.equity_history.append((time.time(), equity))
        pnl_pct = (equity - last_equity) / last_equity if last_equity > 0 else 0.0

        now = time.time()
        vanished: List[str] = []
        appeared: List[str] = []
        last_pl: Dict[str, str] = {}
        with self.state_lock:
            prev_positions = dict(self.positions_snapshot)
            if self._snapshot_ready:
                vanished = sorted(set(self.positions_snapshot) - set(positions))
                appeared = sorted(set(positions) - set(self.positions_snapshot))
                for sym in vanished:
                    try:
                        last_pl[sym] = f"{float(self.positions_snapshot[sym].unrealized_pl):+,.2f}"
                    except Exception:
                        last_pl[sym] = "?"
                for sym in vanished:
                    self.reentry_block_until[sym] = max(
                        self.reentry_block_until.get(sym, 0.0), now + REENTRY_COOLDOWN_MINUTES * 60
                    )
                    self.manual_symbols.discard(sym)
                    self.entry_times.pop(sym, None)
            self.account_snapshot = account
            self.positions_snapshot = positions
            self.daily_pnl_pct = pnl_pct
            self._snapshot_ready = True

        last_prices: Dict[str, float] = {}
        for sym in vanished:
            try:
                last_prices[sym] = float(prev_positions[sym].current_price)
            except Exception:
                pass
        for sym, pos_ in positions.items():
            try:
                self.journal.mark(sym, float(pos_.current_price), abs(float(pos_.qty)))
            except Exception:
                pass
        for sym in appeared:
            self.journal.note_fill(sym, positions[sym])
        for sym in vanished:
            threading.Thread(target=self._journal_exit, args=(sym, last_prices.get(sym)), daemon=True).start()

        for sym in appeared:
            p = positions[sym]
            try:
                side = str(getattr(p.side, "value", p.side)).upper()
                log_event(f"[FILLED] {sym} {side} {abs(float(p.qty)):g} sh @ ${float(p.avg_entry_price):.2f} "
                          f"(${abs(float(p.market_value)):,.0f} position)")
            except Exception:
                log_event(f"[FILLED] {sym} position opened.")
        for sym in vanished:
            log_event(f"[EXIT DETECTED] {sym} position is closed (last seen P&L ${last_pl.get(sym, '?')}). "
                      f"Autopilot re-entry cooldown {REENTRY_COOLDOWN_MINUTES:g}m "
                      f"(longer if you closed it manually).")
        return account, positions

    def _journal_exit(self, symbol: str, fallback_price: Optional[float]) -> None:
        """Find the actual exit fill, then write the closed trade to the journal."""
        exit_px = None
        try:
            rec = self.journal.open.get(symbol)
            if rec:
                since = datetime.datetime.fromtimestamp(rec["entry_ts"] - 60, tz=datetime.timezone.utc)
                exit_side = OrderSide.SELL if rec["side"] == "long" else OrderSide.BUY
                orders = self.trading_client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED, symbols=[symbol], after=since, limit=50))
                fills = [o for o in orders if getattr(o, "filled_at", None) and o.side == exit_side
                         and float(o.filled_qty or 0) > 0 and o.filled_avg_price]
                if fills:
                    exit_px = float(max(fills, key=lambda o: o.filled_at).filled_avg_price)
        except Exception as e:
            logging.warning(f"[JOURNAL] Couldn't fetch the exit fill for {symbol}: {e}")
        try:
            row = self.journal.close(symbol, exit_px, fallback_price)
            if row:
                log_event(f"[JOURNAL] {symbol} {row['reason']}: {stats.fmt_money(row['pnl'])} "
                          f"({row['pnl_bps']:+.1f} bps) | best {row['mfe_bps']:+.0f} / worst {row['mae_bps']:+.0f} bps "
                          f"while open | {row['hold_min']:.0f} min")
        except Exception as e:
            logging.warning(f"[JOURNAL] Couldn't record {symbol}: {e}")

    def refresh_stats(self) -> None:
        """Recompute the dashboard statistics from the journal (runs in the background)."""
        rows = self.journal.read()
        today = datetime.date.today().isoformat()
        cfg_rows = [r for r in rows if r.get("config_id") == CONFIG_ID]
        cfg = stats.summarize(cfg_rows)
        if self.expectations is None:   # the lab may have been run since startup
            self.expectations = incubation.load_expectations(EXPECT_DIR, CONFIG_ID)
        try:
            equity = float(self.account_snapshot.equity) if self.account_snapshot is not None else 100_000.0
        except Exception:
            equity = 100_000.0
        inc = incubation.evaluate(cfg, self.expectations, equity)
        cache = {
            "today": stats.summarize([r for r in rows if r.get("exit_time", "").startswith(today)]),
            "config": cfg,
            "all": stats.summarize(rows),
            "by_hour": stats.breakdown(cfg_rows or rows, stats.by_hour),
            "incubation": inc,
            "updated": time.time(),
        }
        with self.state_lock:
            self.stats_cache = cache
        if inc.get("kill") and not self.killed:
            self.trip_strategy_kill(inc["line"])

    def trip_strategy_kill(self, reason: str) -> None:
        """Strategy-level kill switch: no new entries until a human looks. Existing positions keep
        their brackets, the watchdog stop and the end-of-day flatten."""
        with self.state_lock:
            self.killed = True
            self.autopilot_enabled = False
        try:
            incubation.write_kill(KILL_FILE, CONFIG_ID, reason)
        except Exception as e:
            logging.warning(f"[KILL] Couldn't write {KILL_FILE}: {e}")
        msg = (f"🚨 [KILL SWITCH] Config {CONFIG_ID}: {reason} Delete {KILL_FILE} and restart to re-arm, "
               f"after you've looked at why.")
        log_event(msg, logging.CRITICAL)
        self.set_status("KILL SWITCH: autopilot off")

    def find_carried_positions(self, positions: Dict[str, Any]) -> Tuple[List[str], bool]:
        """Positions with no opening fill since 4:00 AM ET today were left over from an earlier day.
        Returns (symbols, complete); complete is False if any lookup failed (then nothing is assumed)."""
        clock = self.get_clock()
        if clock is None:
            return [], False
        since = clock.next_close.replace(hour=4, minute=0, second=0, microsecond=0)
        carried, complete = [], True
        for sym, p in positions.items():
            try:
                orders = self.trading_client.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED, symbols=[sym], after=since, limit=100))
            except Exception as e:
                logging.warning(f"[CARRIED] Couldn't check today's orders for {sym}: {e}")
                complete = False
                continue
            opening_side = OrderSide.BUY if float(p.qty) > 0 else OrderSide.SELL
            if not any(o.side == opening_side and float(o.filled_qty or 0) > 0 for o in orders):
                carried.append(sym)
        return carried, complete

    def get_positions_copy(self) -> Dict[str, Any]:
        with self.state_lock:
            return dict(self.positions_snapshot)

    def update_signal(self, symbol: str, data: Dict[str, Any]) -> None:
        with self.state_lock:
            self.signals[symbol] = data

    def get_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self.state_lock:
            return self.signals.get(symbol)

    # ---------- manual controls ----------
    def toggle_autopilot(self) -> None:
        with self.state_lock:
            self.autopilot_enabled = not self.autopilot_enabled
            state = "ON" if self.autopilot_enabled else "OFF"
        msg = f"[AUTOPILOT] Turned {state}. (Watchdog, circuit breaker and EOD flatten stay active.)"
        log_event(msg)
        send_discord_alert(msg)
        self.set_status(msg)

    def toggle_lock(self, symbol: str) -> None:
        with self.state_lock:
            if symbol in self.locked_symbols:
                self.locked_symbols.discard(symbol)
                msg = f"[LOCK] {symbol} unlocked. Autopilot may trade it again."
            else:
                self.locked_symbols.add(symbol)
                msg = f"[LOCK] {symbol} locked. Autopilot won't enter or signal-exit it."
        log_event(msg)
        self.set_status(msg)

    def is_auto_managed(self, symbol: str) -> bool:
        with self.state_lock:
            return symbol not in self.manual_symbols and symbol not in self.locked_symbols

    def held_long_enough(self, symbol: str) -> bool:
        with self.state_lock:
            t = self.entry_times.get(symbol)
        return t is None or (time.time() - t) >= MIN_HOLD_SECONDS

    def symbol_flags(self, symbol: str) -> str:
        flags = []
        with self.state_lock:
            if symbol in self.manual_symbols:
                flags.append("MAN")
            if symbol in self.locked_symbols:
                flags.append("LOCK")
            if time.time() < self.reentry_block_until.get(symbol, 0.0):
                flags.append("CD")
        return " ".join(flags)

    # ---------- entry gating ----------
    def auto_entry_allowed(self, symbol: str, sess: Dict[str, Any]) -> Tuple[bool, str]:
        """Quiet pre-checks for autopilot entries (no logging, to avoid spam)."""
        if not self.autopilot_enabled:
            return False, "autopilot off"
        if self.circuit_breaker_tripped:
            return False, "circuit breaker tripped"
        if not sess["open"]:
            return False, "market closed"
        if sess["mins_since_open"] is not None and sess["mins_since_open"] < NO_ENTRY_FIRST_MINUTES:
            return False, "opening window"
        if sess["mins_to_close"] is not None and sess["mins_to_close"] <= NO_ENTRY_LAST_MINUTES:
            return False, "closing window"
        with self.state_lock:
            if self._flattening:
                return False, "flatten in progress"
            if symbol in self.locked_symbols:
                return False, "locked"
            if time.time() < self.reentry_block_until.get(symbol, 0.0):
                return False, "re-entry cooldown"
            if symbol in self.positions_snapshot:
                return False, "already in position"
            if len(self.positions_snapshot) >= MAX_OPEN_POSITIONS:
                return False, "max positions"
        if MAX_ENTRIES_PER_SYMBOL_PER_DAY > 0 and self._entries_count(symbol) >= MAX_ENTRIES_PER_SYMBOL_PER_DAY:
            return False, f"daily entry cap ({MAX_ENTRIES_PER_SYMBOL_PER_DAY})"
        return True, ""

    def _entries_count(self, symbol: str) -> int:
        with self.state_lock:
            today = self.today_key()
            if self._entries_day != today:
                self._entries_day, self.entries_today = today, {}
            return self.entries_today.get(symbol, 0)

    def _count_entry(self, symbol: str) -> None:
        self._entries_count(symbol)  # rolls the counter over on a new day
        with self.state_lock:
            self.entries_today[symbol] = self.entries_today.get(symbol, 0) + 1

    def _check_cooldown(self, symbol: str) -> bool:
        now = time.time()
        with self.state_lock:
            last = self.last_entry_attempt.get(symbol, 0.0)
            if now - last < ENTRY_COOLDOWN_SECONDS:
                return False
            self.last_entry_attempt[symbol] = now
            return True

    def _check_risk_gate(self, symbol: str, side: OrderSide, order_notional: float) -> Tuple[bool, str]:
        """Validates holdings, working orders, position count, cash, buying power, exposure, shortability."""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
            )
        except Exception as e:
            return False, f"account state error: {e}"

        cash = float(account.cash)
        buying_power = float(account.buying_power)
        equity = float(account.equity)

        if any(p.symbol == symbol for p in positions):
            return False, "already has an active position"
        if any(o.symbol == symbol for o in open_orders):
            return False, "has a working order pending"

        pending_entries = [o for o in open_orders if (o.client_order_id or "").startswith(ORDER_PREFIX)]
        if len(positions) + len(pending_entries) >= MAX_OPEN_POSITIONS:
            return False, f"max open positions ({MAX_OPEN_POSITIONS}) reached"

        if buying_power < order_notional:
            return False, f"insufficient buying power (${buying_power:,.2f})"
        if side == OrderSide.BUY and cash < order_notional:
            return False, f"insufficient cash (${cash:,.2f})"

        pending_notional = 0.0
        for o in pending_entries:
            try:
                pending_notional += float(o.qty or 0) * float(o.limit_price or 0)
            except (TypeError, ValueError):
                pass
        gross_exposure = sum(abs(float(p.market_value)) for p in positions) + pending_notional
        if gross_exposure + order_notional > equity * MAX_GROSS_EXPOSURE_MULT:
            return False, f"breaches {MAX_GROSS_EXPOSURE_MULT}x gross exposure cap"

        if side == OrderSide.SELL:
            try:
                asset = self.trading_client.get_asset(symbol)
                if not (asset.shortable and asset.easy_to_borrow):
                    return False, "not shortable / not easy to borrow"
            except Exception as e:
                return False, f"asset check failed: {e}"

        return True, ""

    def _reject(self, symbol: str, tag: str, reason: str, alert: bool = False) -> None:
        msg = f"[BLOCKED] {tag} {symbol}: {reason}"
        log_event(msg)
        self.set_status(msg)
        if alert:
            send_discord_alert(msg)

    # ---------- entries ----------
    def execute_entry(self, symbol: str, side: OrderSide, source: str = "auto") -> None:
        """Limit-bracket entry. source = 'auto' (autopilot) or 'manual' (HUD)."""
        tag = "AUTO" if source == "auto" else "MANUAL"
        side_txt = "BUY" if side == OrderSide.BUY else "SHORT"

        if side == OrderSide.SELL and not ENABLE_SHORTS:
            self._reject(symbol, tag, "shorts disabled (ENABLE_SHORTS=False)")
            return
        if self.circuit_breaker_tripped:
            self._reject(symbol, tag, "daily circuit breaker tripped")
            return

        sess = self.session_info()
        if not sess["open"]:
            self._reject(symbol, tag, "market closed")
            return
        if FLATTEN_AT_EOD and sess["mins_to_close"] is not None:
            cutoff = NO_ENTRY_LAST_MINUTES if source == "auto" else EOD_FLATTEN_MINUTES + 1
            if sess["mins_to_close"] <= cutoff:
                self._reject(symbol, tag, f"too close to the bell ({sess['mins_to_close']:.0f}m left)")
                return
        if source == "auto":
            ok, why = self.auto_entry_allowed(symbol, sess)
            if not ok:
                self.set_status(f"{symbol}: auto entry skipped ({why}).")
                return

        if not self._check_cooldown(symbol):
            self.set_status(f"{symbol}: entry cooldown active.")
            return

        with self.state_lock:
            if symbol in self.in_flight_symbols:
                self.set_status(f"{symbol}: order already in-flight.")
                return
            self.in_flight_symbols.add(symbol)

        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DATA_FEED)
            res = self.data_client.get_stock_latest_quote(req)
            if symbol not in res:
                self._reject(symbol, tag, "no quote")
                return
            bid = float(res[symbol].bid_price)
            ask = float(res[symbol].ask_price)
            if bid <= 0 or ask <= 0 or ask < bid:
                self._reject(symbol, tag, f"invalid quote (bid {bid}, ask {ask})")
                return

            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid
            if spread_pct > MAX_SPREAD_PCT:
                self._reject(symbol, tag, f"spread too wide ({spread_pct * 100:.2f}%)")
                return

            limit_price = round(ask if side == OrderSide.BUY else bid, 2)
            equity = float(self.account_snapshot.equity) if self.account_snapshot is not None else 0.0
            if STRATEGY.position_pct_of_equity > 0 and equity <= 0:
                self._reject(symbol, tag, "no account snapshot yet")
                return
            qty, why_size = position_size(limit_price, equity, STRATEGY)   # same sizing as the lab
            if qty <= 0:
                self._reject(symbol, tag, why_size)
                return
            order_notional = qty * limit_price
            hard_cap = position_budget(equity, STRATEGY) * 1.10
            if order_notional > hard_cap:
                self._reject(symbol, tag, f"order ${order_notional:,.2f} exceeds hard cap ${hard_cap:,.2f}",
                             alert=(not STRATEGY.fractional))
                return

            ok, why = self._check_risk_gate(symbol, side, order_notional)
            if not ok:
                self._reject(symbol, tag, why, alert=(source == "manual"))
                return

            # The same function the lab uses: fixed % or volatility-scaled ("atr" mode).
            atr_val = (self.get_signal(symbol) or {}).get("atr")
            tp_price, sl_price = bracket_levels(limit_price, side == OrderSide.BUY, STRATEGY, atr_val)

            client_order_id = f"{ORDER_PREFIX}{source}-{symbol}-{side.value}-{int(time.time() * 1000)}"
            if STRATEGY.fractional:
                # Fractional orders can't carry brackets: plain DAY limit now, and a broker-side stop
                # is attached as soon as it fills (ensure_fractional_protection).
                order_req = LimitOrderRequest(
                    symbol=symbol, qty=qty, side=side, limit_price=limit_price,
                    time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
                )
            else:
                order_req = LimitOrderRequest(
                    symbol=symbol,
                    qty=int(qty),
                    side=side,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY if FLATTEN_AT_EOD else TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=sl_price),
                    client_order_id=client_order_id,
                )
            order = self.trading_client.submit_order(order_req)
            self._count_entry(symbol)
            try:
                self.journal.note_entry(symbol, side == OrderSide.BUY, source, bid, ask, limit_price,
                                        tp_price, sl_price, self.get_signal(symbol))
            except Exception:
                pass

            with self.state_lock:
                self.entry_times[symbol] = time.time()
                if source == "manual":
                    self.manual_symbols.add(symbol)
                else:
                    self.manual_symbols.discard(symbol)

            msg = (f"[{tag} ORDER PLACED] {side_txt} {qty:g}x {symbol} @ ${limit_price:.2f} "
                   f"(TP: ${tp_price:.2f}, SL: ${sl_price:.2f}) | ID: {order.id}")
            log_event(msg)
            send_discord_alert(msg)
            self.set_status(f"{tag} {side_txt}: {qty:g}x {symbol} @ ${limit_price:.2f}")

        except Exception as e:
            msg = f"[ORDER REJECTED] {tag} {symbol}: {e}"
            log_event(msg, logging.WARNING)
            send_discord_alert(msg)
            self.set_status(f"{symbol}: order failed")
        finally:
            with self.state_lock:
                self.in_flight_symbols.discard(symbol)

    # ---------- exits ----------
    def close_position_safely(self, symbol: str, reason: str = "manual close", source: str = "manual") -> None:
        """Cancels the symbol's open orders (incl. bracket legs), then closes the position."""
        self.journal.note_exit_reason(symbol, reason, source)
        with self.state_lock:
            if symbol in self.closing_symbols:
                return
            self.closing_symbols.add(symbol)
        try:
            cancelled = 0
            try:
                orders = self.trading_client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                )
                for o in orders:
                    try:
                        self.trading_client.cancel_order_by_id(o.id)
                        cancelled += 1
                    except Exception as e:
                        log_event(f"[WARN] Cancel failed for {symbol} order {o.id}: {e}", logging.WARNING)
            except Exception as e:
                log_event(f"[WARN] Could not list orders for {symbol}: {e}", logging.WARNING)

            try:
                self.trading_client.get_open_position(symbol)
            except Exception:
                msg = f"[CLOSE] {symbol}: no open position (cancelled {cancelled} order(s))."
                log_event(msg)
                self.set_status(msg)
                return

            # Bracket leg cancels are async; retry while shares are still "held" by orders.
            last_err: Optional[Exception] = None
            for _ in range(4):
                try:
                    self.trading_client.close_position(symbol)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.0)
            if last_err is not None:
                raise last_err

            block_min = MANUAL_CLOSE_BLOCK_MINUTES if source == "manual" else REENTRY_COOLDOWN_MINUTES
            with self.state_lock:
                self.reentry_block_until[symbol] = max(
                    self.reentry_block_until.get(symbol, 0.0), time.time() + block_min * 60
                )
                self.manual_symbols.discard(symbol)
                self.entry_times.pop(symbol, None)

            msg = f"[LIQUIDATED] {symbol} closed ({reason}). Autopilot re-entry blocked {block_min}m."
            log_event(msg)
            send_discord_alert(msg)
            self.set_status(msg)
        except Exception as e:
            msg = f"[ERROR] Liquidation failed for {symbol}: {e}"
            log_event(msg, logging.ERROR)
            send_discord_alert(msg)
            self.set_status(msg)
        finally:
            with self.state_lock:
                self.closing_symbols.discard(symbol)

    def flatten_all_positions(self, reason: str = "manual panic", disable_autopilot: bool = False,
                              timeout_s: float = 45.0) -> bool:
        """Cancel every order, close every position, and VERIFY the account is flat.

        V5.5 fired close_all_positions() once and logged success without checking the result.
        On Sep 28 the bracket legs were cancelled but the closes didn't go through (the legs
        still "held" the shares), so GOOGL and NFLX were carried overnight with no stop.
        This version cancels first, waits for the cancels to land, closes one symbol at a time,
        re-checks, and retries until flat or out of time. Returns True only if verified flat.
        """
        with self.state_lock:
            if self._flattening:
                return False
            self._flattening = True
            if disable_autopilot:
                self.autopilot_enabled = False
            symbols = list(self.positions_snapshot.keys())
        for sym in symbols:
            self.journal.note_exit_reason(sym, reason, "auto")
        deadline = time.time() + timeout_s
        flat = False
        try:
            # 1) Cancel all open orders, then wait until the broker confirms none remain.
            try:
                self.trading_client.cancel_orders()
            except Exception as e:
                log_event(f"[FLATTEN] cancel_orders failed, continuing: {e}", logging.WARNING)
            while time.time() < deadline:
                try:
                    if not self.trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)):
                        break
                except Exception:
                    pass
                time.sleep(0.5)

            # 2) Close positions one by one; retry anything still open that has no pending close.
            attempts: Dict[str, int] = {}
            while time.time() < deadline:
                positions = self.trading_client.get_all_positions()
                if not positions:
                    flat = True
                    break
                pending = {o.symbol for o in self.trading_client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))}
                for pos in positions:
                    if pos.symbol in pending:
                        continue  # a close is already working; don't stack another
                    attempts[pos.symbol] = attempts.get(pos.symbol, 0) + 1
                    try:
                        self.trading_client.close_position(pos.symbol)
                    except Exception as e:
                        log_event(f"[FLATTEN] close {pos.symbol} attempt {attempts[pos.symbol]} failed: {e}",
                                  logging.WARNING)
                time.sleep(2.0)

            now = time.time()
            with self.state_lock:
                for sym in symbols:
                    self.reentry_block_until[sym] = max(
                        self.reentry_block_until.get(sym, 0.0), now + REENTRY_COOLDOWN_MINUTES * 60
                    )
                self.manual_symbols.clear()
                self.entry_times.clear()

            extra = " Autopilot OFF." if disable_autopilot else ""
            if flat:
                msg = f"[FLATTEN] Verified flat: all positions closed, orders cancelled ({reason}).{extra}"
                log_event(msg)
            else:
                left = ", ".join(sorted(p.symbol for p in self.trading_client.get_all_positions())) or "unknown"
                msg = (f"🚨 [FLATTEN INCOMPLETE] Still holding: {left} ({reason}). "
                       f"Retrying automatically; close them manually if this repeats.{extra}")
                log_event(msg, logging.ERROR)
            self.set_status(msg)
            return flat
        except Exception as e:
            msg = f"🚨 [FLATTEN FAILED] ({reason}): {e}. Retrying automatically."
            log_event(msg, logging.ERROR)
            self.set_status(msg)
            return False
        finally:
            with self.state_lock:
                self._flattening = False

    def protect_partial_fill(self, order: Any) -> None:
        """A partly filled entry was cancelled, and Alpaca cancelled its stop/target with it.
        Re-attach a stop-loss + take-profit (OCO) to the shares that did fill; if that's
        impossible, close them. Either way nothing is left unprotected (V5.5 left 13 AAPL
        shares naked for ~2 hours on Oct 1)."""
        symbol = order.symbol
        if STRATEGY.fractional:
            return  # micro mode: ensure_fractional_protection() puts a stop on whatever was filled
        for _ in range(10):  # let the cancel and its bracket legs land
            try:
                if not self.trading_client.get_orders(
                        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])):
                    break
            except Exception:
                pass
            time.sleep(0.5)
        try:
            pos = self.trading_client.get_open_position(symbol)
        except Exception:
            return  # nothing held any more
        held = float(pos.qty)
        qty = abs(int(held))
        if qty == 0:
            return
        is_long = held > 0
        entry = float(order.limit_price or pos.avg_entry_price)
        planned = self.journal.pending.get(symbol) or self.journal.open.get(symbol) or {}
        if planned.get("tp") and planned.get("sl"):
            tp, sl = float(planned["tp"]), float(planned["sl"])   # keep the levels set at entry
        else:
            tp, sl = bracket_levels(entry, is_long, STRATEGY, (self.get_signal(symbol) or {}).get("atr"))
        cid = f"{PROTECT_PREFIX}{symbol}-{int(time.time() * 1000)}"
        last_err: Optional[Exception] = None
        # Alpaca documents OCO without a top-level limit price; some SDK versions demand one.
        for extra in ({}, {"limit_price": tp}):
            try:
                self.trading_client.submit_order(LimitOrderRequest(
                    symbol=symbol, qty=qty,
                    side=OrderSide.SELL if is_long else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY if FLATTEN_AT_EOD else TimeInForce.GTC,
                    order_class=OrderClass.OCO,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                    client_order_id=cid,
                    **extra,
                ))
                log_event(f"[PARTIAL FILL] {symbol}: {qty} of {order.qty} shares filled, rest cancelled. "
                          f"Re-attached stop ${sl:.2f} and target ${tp:.2f} to the {qty} shares.")
                return
            except Exception as e:
                last_err = e
        log_event(f"[PARTIAL FILL] {symbol}: couldn't re-attach a stop ({last_err}). "
                  f"Closing the {qty} shares instead.", logging.WARNING)
        self.close_position_safely(symbol, "partial fill without protection", source="auto")

    def ensure_fractional_protection(self, positions: Dict[str, Any]) -> None:
        """Micro mode, every watchdog cycle: each long position has exactly one broker-side stop for
        its full size, and Sparticus sells it when the take-profit is reached. The stop lives at
        Alpaca, so it protects the position even if this Mac sleeps; the target needs the bot."""
        try:
            open_orders = self.trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        except Exception as e:
            logging.warning(f"[PROTECT] Couldn't list open orders: {e}")
            return
        stops: Dict[str, List[Any]] = {}
        entry_open = set()
        for o in open_orders:
            cid = o.client_order_id or ""
            if cid.startswith(PROTECT_PREFIX):
                stops.setdefault(o.symbol, []).append(o)
            elif cid.startswith(ORDER_PREFIX):
                entry_open.add(o.symbol)
        for sym, p in positions.items():
            with self.state_lock:
                if sym in self.closing_symbols or self._flattening:
                    continue
            try:
                qty = float(p.qty)
                if qty <= 0:
                    continue
                price = float(p.current_price)
                planned = self.journal.open.get(sym) or {}
                if planned.get("tp") and planned.get("sl"):
                    tp, sl = float(planned["tp"]), float(planned["sl"])
                else:
                    tp, sl = bracket_levels(float(p.avg_entry_price), True, STRATEGY,
                                            (self.get_signal(sym) or {}).get("atr"))
                if price >= tp or price <= sl:
                    hit_tp = price >= tp
                    log_event(f"[{'TARGET' if hit_tp else 'STOP'}] {sym} at ${price:.2f} "
                              f"(target ${tp:.2f}, stop ${sl:.2f}). Selling.")
                    threading.Thread(target=self.close_position_safely,
                                     args=(sym, "take profit" if hit_tp else "software stop", "auto"),
                                     daemon=True).start()
                    continue
                if sym in entry_open:
                    continue  # entry still working: Alpaca would reject a sell stop beside it (wash-trade check)
                have = stops.get(sym, [])
                if len(have) == 1 and abs(float(have[0].qty) - qty) < 1e-6 \
                        and abs(float(have[0].stop_price or 0) - round(sl, 2)) < 0.005:
                    continue
                for o in have:
                    self.trading_client.cancel_order_by_id(o.id)
                if have and not self._wait_cancelled([o.id for o in have]):
                    continue  # an old stop filled or is still cancelling: look again next cycle
                self.trading_client.submit_order(StopOrderRequest(
                    symbol=sym, qty=qty, side=OrderSide.SELL, stop_price=round(sl, 2),
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"{PROTECT_PREFIX}{sym}-{int(time.time() * 1000)}",
                ))
                self._protect_warned.pop(sym, None)
                log_event(f"[PROTECT] {sym}: broker-side stop ${sl:.2f} on {qty:g} sh; "
                          f"target ${tp:.2f} handled by Sparticus.")
            except Exception as e:
                if time.time() - self._protect_warned.get(sym, 0.0) > 300:   # at most every 5 minutes
                    self._protect_warned[sym] = time.time()
                    log_event(f"[PROTECT] {sym}: couldn't place its stop ({e}). Sparticus's own stop and "
                              f"the -3% watchdog still apply while the bot runs.", logging.WARNING)

    def _wait_cancelled(self, order_ids: List[Any], tries: int = 10) -> bool:
        """True once every order is cancelled (its shares are free again). False if one filled or is slow."""
        for _ in range(tries):
            statuses = []
            for oid in order_ids:
                try:
                    statuses.append(str(getattr(self.trading_client.get_order_by_id(oid), "status", "")).lower())
                except Exception:
                    statuses.append("")
            if any("filled" in st for st in statuses):
                return False
            if all(any(w in st for w in ("canceled", "cancelled", "expired", "rejected")) for st in statuses):
                return True
            time.sleep(0.2)
        return False

    def post_daily_summary(self) -> None:
        try:
            acct = self.trading_client.get_account()
            equity, last = float(acct.equity), float(acct.last_equity or 0)
            change = equity - last
            pct = change / last * 100 if last else 0.0
            self._entries_count("")  # roll over if needed
            with self.state_lock:
                total = sum(self.entries_today.values())
                busiest = sorted(self.entries_today.items(), key=lambda kv: -kv[1])[:3]
            busy_txt = ", ".join(f"{s} x{n}" for s, n in busiest) or "none"
            try:
                self.refresh_stats()
                t = self.stats_cache.get("today", {})
                a = self.stats_cache.get("all", {})
                edge = (f" | Per trade {stats.fmt_money(t.get('expectancy'))}, win {t.get('win_rate', 0) * 100:.0f}% "
                        f"vs breakeven {(t.get('breakeven_win_rate') or 0) * 100:.0f}% | All time: {a.get('trades', 0)} "
                        f"trades, t-stat {a.get('t_stat', 0):+.2f} ({a.get('verdict', '-')})") if t.get("trades") else ""
            except Exception:
                edge = ""
            inc = (self.stats_cache or {}).get("incubation") or {}
            if inc.get("line"):
                edge += f" | Config {CONFIG_ID}: {inc['line']}"
            log_event(f"📊 [DAILY SUMMARY] {'🟢' if change >= 0 else '🔴'} Day P&L ${change:+,.2f} "
                      f"({pct:+.2f}%) | Equity ${equity:,.2f} | Entries {total} (most: {busy_txt}) | "
                      f"Positions open: {len(self.positions_snapshot)}{edge}")
        except Exception as e:
            log_event(f"[WARN] Daily summary failed: {e}", logging.WARNING)

    def cancel_stale_entries(self) -> None:
        """Cancels Sparticus entry orders that haven't filled within STALE_ENTRY_SECONDS."""
        try:
            orders = self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
            )
        except Exception as e:
            log_event(f"[WARN] Stale-order check failed: {e}", logging.WARNING)
            return

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        for o in orders:
            if not (o.client_order_id or "").startswith(ORDER_PREFIX):
                continue  # bracket legs / other orders
            ts = o.submitted_at or o.created_at
            if ts is None:
                continue
            age = (now_utc - ts).total_seconds()
            if age < STALE_ENTRY_SECONDS:
                continue
            try:
                filled = float(o.filled_qty or 0)
                self.trading_client.cancel_order_by_id(o.id)
                log_event(f"[STALE ENTRY] Cancelled {o.symbol} entry after {age:.0f}s "
                          f"(filled {o.filled_qty}/{o.qty}).")
                if filled > 0:
                    # Cancelling a partly filled bracket also cancels its stop and target.
                    threading.Thread(target=self.protect_partial_fill, args=(o,), daemon=True).start()
                else:
                    with self.state_lock:
                        if o.symbol not in self.positions_snapshot:
                            self.manual_symbols.discard(o.symbol)
                            self.entry_times.pop(o.symbol, None)
            except Exception as e:
                log_event(f"[WARN] Could not cancel stale order {o.id}: {e}", logging.WARNING)


# ==========================================
# 3. AUTOPILOT SCANNER (entries + signal exits)
# ==========================================
def _sleep_to_next_scan() -> None:
    """Wake ~5s after each new minute so the latest 1-min bar is available."""
    now = time.time()
    time.sleep(max(1.0, SCAN_INTERVAL_SECONDS - (now % SCAN_INTERVAL_SECONDS) + 5))


def scanner_worker(engine: SparticusEngineV55, symbols: List[str]) -> None:
    log_event("[AUTOPILOT] Scanner online.")
    min_bars = strategy_min_bars(STRATEGY)

    while True:
        engine.last_scan_ts = time.time()
        try:
            sess = engine.session_info()
            if not sess["open"]:
                engine.set_status("Market closed. Scanner sleeping...")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            bars = engine.data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=now_utc - datetime.timedelta(minutes=BAR_LOOKBACK_MINUTES),
                feed=DATA_FEED,
            ))
            data = getattr(bars, "data", {}) or {}
            positions = engine.get_positions_copy()
            n_long = n_short = n_exit = 0

            for symbol in symbols:
                series = data.get(symbol, [])
                if len(series) < min_bars:
                    engine.update_signal(symbol, {"signal": "NODATA"})
                    continue

                closes = [float(b.close) for b in series]
                bar_atr = atr([float(b.high) for b in series], [float(b.low) for b in series], closes,
                              STRATEGY.atr_period)
                bar_age = (now_utc - series[-1].timestamp).total_seconds()
                stale = bar_age > MAX_BAR_AGE_SECONDS

                signal = compute_signal(closes, STRATEGY)   # the rule the backtester tests
                if signal is None:
                    engine.update_signal(symbol, {"signal": "NODATA"})
                    continue
                fast, slow, rsi = signal.fast, signal.slow, signal.rsi
                crossed_up, crossed_down = signal.crossed_up, signal.crossed_down
                long_signal, short_signal = signal.long_entry, signal.short_entry

                sig_txt = "STALE" if stale else ("LONG" if long_signal else ("SHORT" if short_signal else "-"))
                engine.update_signal(symbol, {
                    "price": closes[-1], "fast": fast, "slow": slow, "rsi": rsi,
                    "signal": sig_txt, "stale": stale,
                    # for the HUD only:
                    "closes": closes[-90:], "crossed_up": crossed_up, "crossed_down": crossed_down,
                    "bar_age": bar_age, "ts": time.time(), "atr": bar_atr,
                })
                if stale or not engine.autopilot_enabled:
                    continue

                # --- Autonomous signal exits (auto-managed positions only) ---
                pos = positions.get(symbol)
                if pos is not None:
                    if (SIGNAL_EXIT_ENABLED and engine.is_auto_managed(symbol)
                            and engine.held_long_enough(symbol)):
                        is_long = float(pos.qty) > 0
                        if is_long and signal.exit_long:
                            n_exit += 1
                            log_event(f"[SIGNAL EXIT] {symbol} long: 5-SMA {fast:.2f} < 20-SMA {slow:.2f}")
                            engine.close_position_safely(symbol, "trend reversal", source="auto")
                        elif not is_long and signal.exit_short:
                            n_exit += 1
                            log_event(f"[SIGNAL EXIT] {symbol} short: 5-SMA {fast:.2f} > 20-SMA {slow:.2f}")
                            engine.close_position_safely(symbol, "trend reversal", source="auto")
                    continue

                # --- Autonomous entries ---
                if not (long_signal or short_signal):
                    continue
                ok, _ = engine.auto_entry_allowed(symbol, sess)
                if not ok:
                    continue
                if long_signal:
                    n_long += 1
                    log_event(f"[SIGNAL] {symbol} LONG @ ${closes[-1]:.2f} | 5-SMA {fast:.2f} > "
                              f"20-SMA {slow:.2f} | RSI {rsi:.1f}")
                    engine.execute_entry(symbol, OrderSide.BUY, "auto")
                else:
                    n_short += 1
                    log_event(f"[SIGNAL] {symbol} SHORT @ ${closes[-1]:.2f} | 5-SMA {fast:.2f} < "
                              f"20-SMA {slow:.2f} | RSI {rsi:.1f}")
                    engine.execute_entry(symbol, OrderSide.SELL, "auto")

            ap = "ON" if engine.autopilot_enabled else "OFF"
            engine.set_status(f"Scan done {datetime.datetime.now().strftime('%H:%M:%S')} | autopilot {ap} | "
                              f"long sig {n_long}, short sig {n_short}, exits {n_exit}, "
                              f"positions {len(positions)}")
        except Exception as e:
            log_event(f"[SCANNER ERROR] {e}", logging.ERROR)

        _sleep_to_next_scan()


# ==========================================
# 4. WATCHDOG (risk, EOD flatten, stale orders)
# ==========================================
_WATCHDOG_ERR_TS: Dict[str, float] = {}


def _guarded(name: str, fn, *args) -> None:
    """Run one watchdog stage. A stage that fails is logged (at most once a minute) and the OTHER
    stages still run: V6.3 lost every stop and the end-of-day flatten to one missing method."""
    try:
        fn(*args)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass
    except Exception as e:
        now = time.time()
        if now - _WATCHDOG_ERR_TS.get(name, 0.0) > 60:
            _WATCHDOG_ERR_TS[name] = now
            log_event(f"[WATCHDOG ERROR] {name}: {type(e).__name__}: {e}", logging.ERROR)


def _check_circuit_breaker(engine: SparticusEngineV55, positions: Dict[str, Any], sess: Dict[str, Any]) -> None:
    if engine.daily_pnl_pct <= MAX_DAILY_DRAWDOWN_PCT and not engine.circuit_breaker_tripped:
        engine.trip_circuit_breaker(engine.daily_pnl_pct)
    elif (engine.circuit_breaker_tripped and positions and sess["open"] and not engine._flattening
          and time.time() - engine.cb_last_flatten > CB_FLATTEN_RETRY_SECONDS):
        # The flatten that the breaker started didn't finish: keep going until the account is flat.
        engine.cb_last_flatten = time.time()
        threading.Thread(target=engine.flatten_all_positions, args=("circuit breaker retry", True),
                         daemon=True).start()


def _check_position_stops(engine: SparticusEngineV55, positions: Dict[str, Any]) -> None:
    for sym, p in positions.items():
        plpc = float(p.unrealized_plpc)
        if plpc <= HARD_WATCHDOG_LOSS_PCT:
            with engine.state_lock:
                already = sym in engine.closing_symbols
            if not already:
                msg = f"⚠️ [WATCHDOG STOP] {sym} at {plpc * 100:.2f}%. Liquidating."
                log_event(msg, logging.WARNING)
                send_discord_alert(msg)
                threading.Thread(target=engine.close_position_safely,
                                 args=(sym, "watchdog stop", "auto"), daemon=True).start()


def _check_overnight(engine: SparticusEngineV55, positions: Dict[str, Any], sess: Dict[str, Any]) -> None:
    if not sess["open"] and positions and FLATTEN_AT_EOD:
        today = engine.today_key()
        if engine.overnight_alert_day != today and engine.eod_flatten_day != today:
            engine.overnight_alert_day = today
            log_event(f"🚨 [OVERNIGHT EXPOSURE] Market closed with open positions: "
                      f"{', '.join(sorted(positions))}. Their bracket legs may have expired. "
                      f"Check them in the Alpaca dashboard.", logging.ERROR)


def _check_carried(engine: SparticusEngineV55, positions: Dict[str, Any], sess: Dict[str, Any]) -> None:
    """V6.4: once a day, just after the open, sell anything left over from an earlier day."""
    if not (CLOSE_CARRIED_AT_OPEN and sess["open"] and positions):
        return
    if (sess.get("mins_since_open") or 0.0) < CARRIED_CHECK_AFTER_MIN:
        return
    today = engine.today_key()
    if engine.carried_check_day == today or time.time() < engine.carried_next_try:
        return
    carried, complete = engine.find_carried_positions(positions)
    if complete:
        engine.carried_check_day = today
    else:
        engine.carried_next_try = time.time() + 60   # a lookup failed: look again in a minute
    if carried:
        log_event(f"🚨 [CARRIED] {', '.join(sorted(carried))} were left open from an earlier day (Sparticus "
                  f"wasn't running at 3:55). Selling them now.", logging.WARNING)
        for sym in carried:
            threading.Thread(target=engine.close_position_safely, args=(sym, "carried overnight", "auto"),
                             daemon=True).start()


def flatten_before_quit(engine: SparticusEngineV55, why: str) -> None:
    """V6.4: Sparticus never holds overnight, so stopping it during market hours closes its positions
    first. (Outside market hours there's nothing to do: the 3:55 flatten already ran.)"""
    if not FLATTEN_ON_QUIT:
        return
    try:
        sess = engine.session_info()
        held = sorted(engine.get_positions_copy())
        if not sess["open"] or not held:
            return
        log_event(f"[SHUTDOWN] {why} during market hours with {len(held)} open position(s): {', '.join(held)}. "
                  f"Closing them before exit (Sparticus never holds overnight).", logging.WARNING)
        if not engine.flatten_all_positions(f"stopped ({why})", disable_autopilot=True):
            log_event("🚨 [SHUTDOWN] Couldn't verify the account is flat. Check positions in the Alpaca "
                      "dashboard NOW.", logging.CRITICAL)
    except Exception as e:
        log_event(f"🚨 [SHUTDOWN] Closing positions on exit failed ({e}). Check positions in the Alpaca "
                  f"dashboard NOW.", logging.CRITICAL)


class StopRequested(Exception):
    """Raised in the main thread when Terminal closes (SIGHUP) or the system asks the bot to stop (SIGTERM)."""


def _on_stop_signal(signum, frame) -> None:
    raise StopRequested({getattr(signal, "SIGHUP", None): "Terminal closed",
                         signal.SIGTERM: "stop signal"}.get(signum, "stop signal"))


def _check_end_of_day(engine: SparticusEngineV55, sess: Dict[str, Any]) -> None:
    today = engine.today_key()
    if (FLATTEN_AT_EOD and sess["mins_to_close"] is not None
            and sess["mins_to_close"] <= EOD_FLATTEN_MINUTES
            and engine.eod_flatten_day != today):
        if engine.flatten_all_positions("end-of-day flatten"):
            engine.eod_flatten_day = today
            engine.post_daily_summary()
    else:
        engine.cancel_stale_entries()


def watchdog_cycle(engine: SparticusEngineV55) -> None:
    """One pass of the watchdog. The snapshot must succeed; every stage after it runs on its own."""
    _, positions = engine.refresh_snapshot()
    sess = engine.session_info()
    engine.cached_session = sess

    _guarded("circuit breaker", _check_circuit_breaker, engine, positions, sess)
    _guarded("position stops", _check_position_stops, engine, positions)
    if STRATEGY.fractional and positions and sess["open"]:
        _guarded("fractional protection", engine.ensure_fractional_protection, positions)
    _guarded("overnight check", _check_overnight, engine, positions, sess)
    _guarded("carried positions", _check_carried, engine, positions, sess)
    if sess["open"]:
        _guarded("end of day", _check_end_of_day, engine, sess)


def watchdog_worker(engine: SparticusEngineV55) -> None:
    while True:
        engine.last_watchdog_ts = time.time()
        try:
            watchdog_cycle(engine)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        except Exception as e:
            log_event(f"[WATCHDOG ERROR] {e}", logging.ERROR)

        time.sleep(WATCHDOG_INTERVAL_SECONDS)


def health_monitor(engine: SparticusEngineV55, threads: Dict[str, threading.Thread], headless: bool) -> None:
    """Alerts when the scanner or watchdog dies or stalls. In --headless mode, exits after
    RESTART_ON_STALL_SECONDS so run_sparticus.sh can start a fresh copy."""
    alerted: Dict[str, bool] = {}
    while True:
        time.sleep(HEALTH_CHECK_SECONDS)
        try:
            now = time.time()
            problems: List[Tuple[str, str, float]] = []
            for name, t in threads.items():
                if not t.is_alive():
                    problems.append((name, "thread has stopped", float("inf")))
            if engine.session_info()["open"]:
                for name, ts, limit in (("scanner", engine.last_scan_ts, SCANNER_STALL_ALERT_SECONDS),
                                        ("watchdog", engine.last_watchdog_ts, WATCHDOG_STALL_ALERT_SECONDS)):
                    if now - ts > limit and threads[name].is_alive():
                        problems.append((name, f"no heartbeat for {(now - ts) / 60:.0f} min", now - ts))
            current = {p[0] for p in problems}
            for name in [n for n in alerted if n not in current]:
                alerted.pop(name)
                log_event(f"[HEALTH] {name} recovered.")
            for name, what, age in problems:
                if not alerted.get(name):
                    alerted[name] = True
                    log_event(f"🚨 [HEALTH] {name}: {what}. Trading may be impaired.", logging.CRITICAL)
                if headless and age >= RESTART_ON_STALL_SECONDS:
                    log_event(f"🚨 [HEALTH] {name} stuck too long. Exiting so run_sparticus.sh restarts "
                              f"the bot.", logging.CRITICAL)
                    DISCORD.shutdown()
                    os._exit(3)
        except Exception as e:
            log_event(f"[HEALTH] Check failed: {e}", logging.WARNING)


# ==========================================
# 5. PYGAME HUD (reads cached state only; no API calls per frame)
# ==========================================
def run_pygame_hud(engine: SparticusEngineV55, symbols: List[str]) -> None:
    """Sparticus "brain" HUD. Reads cached engine state only: no broker calls per frame.

    Left:  brain map (one tile per ticker, coloured by trend strength) + thought stream.
    Right: selected ticker's chart (price, 5/20 SMA, entry/TP/SL), RSI, the decision
           pipeline (every gate the autopilot checks, live), and today's equity curve.
    A drawing error never stops trading: it's logged and the frame is skipped.
    """
    import math
    import pygame

    pygame.init()
    W, H = 1280, 800
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Sparticus {BOT_VERSION} - Brain HUD")

    def mono(size: int, bold: bool = False):
        return pygame.font.SysFont("menlo,dejavusansmono,consolas,couriernew,courier", size, bold=bold)

    F_S, F_M, F_B, F_XL = mono(12), mono(14), mono(14, True), mono(24, True)
    clock = pygame.time.Clock()

    BG, PANEL, EDGE = (11, 13, 18), (19, 23, 31), (44, 51, 64)
    TXT, DIM, FAINT = (210, 214, 222), (125, 133, 146), (38, 44, 56)
    GREEN, RED, YELLOW = (70, 220, 130), (240, 95, 95), (250, 210, 80)
    CYAN, ORANGE, PURPLE = (80, 200, 245), (245, 160, 70), (185, 145, 250)
    NEUTRAL = (36, 41, 51)

    COLS = 7
    GRID_X, GRID_Y, GRID_W, GRID_H, GAP = 16, 96, 546, 372, 6
    rows = max(1, math.ceil(len(symbols) / COLS))
    TILE_W = (GRID_W - (COLS - 1) * GAP) // COLS
    TILE_H = max(24, min(46, (GRID_H - (rows - 1) * GAP) // rows))
    RX, RW = 580, 684  # right column

    selected = 0
    text_cache: Dict[Tuple[str, Tuple[int, int, int], int], Any] = {}
    series_cache: Dict[str, Tuple[float, List[float], List[Optional[float]], List[Optional[float]],
                                  List[Optional[float]]]] = {}
    flashes: Dict[str, Tuple[float, Tuple[int, int, int]]] = {}
    prev_positions: Optional[set] = None
    last_draw_error = 0.0

    def spawn(fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def text(s: str, x: int, y: int, color=TXT, font=None, right: bool = False) -> int:
        font = font or F_M
        key = (s, color, id(font))
        surf = text_cache.get(key)
        if surf is None:
            if len(text_cache) > 4000:
                text_cache.clear()
            surf = text_cache[key] = font.render(s, True, color)
        if right:
            x -= surf.get_width()
        screen.blit(surf, (x, y))
        return surf.get_width()

    def panel(x: int, y: int, w: int, h: int, title: str = "") -> None:
        pygame.draw.rect(screen, PANEL, (x, y, w, h), border_radius=8)
        pygame.draw.rect(screen, EDGE, (x, y, w, h), 1, border_radius=8)
        if title:
            text(title, x + 12, y + 8, DIM, F_S)

    def lerp(a, b, t: float):
        t = max(0.0, min(1.0, t))
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def dashed(color, x1: int, x2: int, y: int) -> None:
        for x in range(x1, x2, 10):
            pygame.draw.line(screen, color, (x, y), (min(x + 5, x2), y), 1)

    def rolling_sma(vals: List[float], n: int) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for i in range(len(vals)):
            out.append(sum(vals[i - n + 1:i + 1]) / n if i >= n - 1 else None)
        return out

    def series_for(sym: str, sig: Dict[str, Any]):
        """Closes + SMA/RSI history, recomputed only when a new scan arrives for this ticker."""
        closes = sig.get("closes") or []
        ts = sig.get("ts", 0.0)
        cached = series_cache.get(sym)
        if cached and cached[0] == ts:
            return cached[1:]
        rsis: List[Optional[float]] = [calculate_rsi(closes[:i + 1], RSI_PERIOD) for i in range(len(closes))]
        out = (closes, rolling_sma(closes, FAST_SMA_PERIOD), rolling_sma(closes, SLOW_SMA_PERIOD), rsis)
        series_cache[sym] = (ts,) + out
        return out

    def line_chart(x: int, y: int, w: int, h: int, lines, lo: float, hi: float) -> None:
        span = (hi - lo) or 1.0
        for vals, color, width in lines:
            pts = [(x + i * (w - 1) / max(1, len(vals) - 1), y + h - (v - lo) / span * h)
                   for i, v in enumerate(vals) if v is not None]
            if len(pts) >= 2:
                pygame.draw.lines(screen, color, False, pts, width)

    def tile_fill(sig: Dict[str, Any]):
        if not sig or sig.get("signal") in ("NODATA", "STALE") or not sig.get("slow"):
            return NEUTRAL
        k = (sig["fast"] - sig["slow"]) / sig["slow"] / 0.002  # +/-0.2% gap = full colour
        return lerp(NEUTRAL, (28, 140, 86), k) if k >= 0 else lerp(NEUTRAL, (165, 48, 60), -k)


    view = {"mode": "brain"}   # [TAB] switches the right-hand side between "brain" and "stats"

    def verdict_colour(v: str):
        return GREEN if v == "likely a real edge" else RED if v.startswith("losing") else YELLOW

    def draw_stats_page(now: float, acc) -> None:
        with engine.state_lock:
            sc = dict(engine.stats_cache)
        cols = [("TODAY", sc.get("today") or {}), (f"CONFIG {CONFIG_ID}", sc.get("config") or {}),
                ("ALL TIME", sc.get("all") or {})]
        allt = sc.get("all") or {}
        top = GRID_Y - 24
        panel(RX, top, RW, 262, "EDGE   is there one, and is it real?")
        if not allt.get("trades"):
            text("No closed trades in the journal yet.", RX + 16, top + 50, TXT, F_B)
            text("Stats appear after the first exit and refresh every minute.", RX + 16, top + 74, DIM, F_M)
            text("Journal file: sparticus_journal.csv (next to the bot).", RX + 16, top + 98, DIM, F_M)
            return
        cx = [RX + 300, RX + 430, RX + 560]
        for (label, _), x in zip(cols, cx):
            text(label, x + 110, top + 28, DIM, F_S, right=True)

        def pct(v):
            return "-" if v is None else f"{v * 100:.0f}%"
        rows = [
            ("Trades", lambda c: f"{c.get('trades', 0)}"),
            ("Net P&L", lambda c: stats.fmt_money(c.get("net_pnl")) if c.get("trades") else "-"),
            ("Per trade", lambda c: f"{stats.fmt_money(c.get('expectancy'))} {c.get('expectancy_bps', 0):+.1f}bp" if c.get("trades") else "-"),
            ("Win rate / breakeven", lambda c: f"{pct(c.get('win_rate'))} / {pct(c.get('breakeven_win_rate'))}" if c.get("trades") else "-"),
            ("Payoff (win / loss)", lambda c: f"{c.get('payoff', 0):.2f}" if c.get("trades") and c.get("payoff") not in (None, float('inf')) else "-"),
            ("Profit factor", lambda c: f"{c.get('profit_factor', 0):.2f}" if c.get("trades") and c.get("profit_factor") != float('inf') else "-"),
            ("t-stat (2+ = not luck)", lambda c: f"{c.get('t_stat', 0):+.2f}" if c.get("trades") else "-"),
        ]
        for i, (label, fn) in enumerate(rows):
            y = top + 50 + i * 24
            text(label, RX + 16, y, TXT, F_M)
            for (_, c), x in zip(cols, cx):
                val = fn(c)
                col = TXT
                if label in ("Net P&L", "Per trade") and c.get("trades"):
                    col = GREEN if (c.get("net_pnl") or 0) >= 0 else RED
                text(val, x + 110, y, col, F_M, right=True)
        y = top + 50 + len(rows) * 24 + 8
        v = str(allt.get("verdict", ""))
        need = allt.get("trades_needed")
        extra = f"  (needs ~{need:,} trades to prove)" if need and need > allt.get("trades", 0) else ""
        text(f"Verdict, all time: {v}{extra}", RX + 16, y, verdict_colour(v), F_B)

        # ---- leaks ----
        ly = top + 272
        panel(RX, ly, RW, 196, "WHERE MONEY LEAKS   all time")
        def bps(k):
            v_ = allt.get(k)
            return "-" if v_ is None else f"{v_:+.1f} bps"
        leak_rows = [("Spread at entry", bps("avg_spread_bps")), ("Slippage vs signal price", bps("avg_slippage_bps")),
                     ("Best move while open (avg)", bps("avg_mfe_bps")), ("Worst move while open (avg)", bps("avg_mae_bps")),
                     ("Share of best move kept", pct(allt.get("kept_of_best_move"))),
                     ("Median hold", "-" if allt.get("median_hold_min") is None else f"{allt['median_hold_min']:.0f} min")]
        for i, (label, val) in enumerate(leak_rows):
            y = ly + 30 + i * 24
            text(label, RX + 16, y, TXT, F_M)
            text(val, RX + 370, y, DIM, F_M, right=True)
        text("HOW TRADES ENDED", RX + 400, ly + 30, DIM, F_S)
        for i, (reason, (n, avg)) in enumerate(list((allt.get("exits") or {}).items())[:6]):
            y = ly + 50 + i * 22
            text(f"{reason}", RX + 400, y, TXT, F_M)
            text(f"{n}", RX + 560, y, DIM, F_M, right=True)
            text(stats.fmt_money(avg), RX + RW - 16, y, GREEN if avg >= 0 else RED, F_M, right=True)

        # ---- by hour ----
        hy = ly + 206
        panel(RX, hy, RW, 128, "P&L BY ENTRY HOUR   this config (clues to test in the lab, not rules)")
        hours = sc.get("by_hour") or []
        if hours:
            peak_ = max(abs(r[2]) for r in hours) or 1.0
            bw = (RW - 40) // max(len(hours), 1)
            base = hy + 70
            for i, (label, n, net, avg, wr) in enumerate(hours):
                x = RX + 20 + i * bw
                h_ = int(abs(net) / peak_ * 26)
                pygame.draw.rect(screen, GREEN if net >= 0 else RED,
                                 (x + 6, base - h_ if net >= 0 else base, bw - 12, max(h_, 1)))
                text(stats.fmt_money(net), x + bw // 2, hy + 24, GREEN if net >= 0 else RED, F_S, right=False)
                text(label[:5], x + bw // 2 - 18, hy + 104, DIM, F_S)
            pygame.draw.line(screen, EDGE, (RX + 16, base), (RX + RW - 16, base), 1)

        # ---- risk ----
        ry_ = hy + 138
        panel(RX, ry_, RW, H - ry_ - 34, "RISK & INCUBATION   this config")
        allt = sc.get("config") or allt
        risk = [f"Max drawdown {stats.fmt_money(allt.get('max_drawdown'))}",
                f"Worst day {stats.fmt_money(allt.get('worst_day'))}",
                f"Longest losing streak {allt.get('longest_losing_streak', 0)}",
                f"Sizing {f'{STRATEGY.position_pct_of_equity:.0%} of equity' if STRATEGY.position_pct_of_equity > 0 else stats.fmt_money(STRATEGY.target_dollars_per_trade).lstrip('+')}"]
        text("   ".join(risk), RX + 16, ry_ + 24, TXT, F_S)
        inc = sc.get("incubation") or {}
        inc_col = {"kill": RED, "below": RED, "on_track": GREEN, "above": YELLOW}.get(inc.get("status"), DIM)
        head, _, tail = str(inc.get("line", "Incubation: waiting for the first trade on this config")).partition(" | ")
        text(head[:88], RX + 16, ry_ + 41, inc_col, F_S)
        if tail:
            text(tail[:88], RX + 16, ry_ + 57, DIM, F_S)
        upd = sc.get("updated")
        if upd:
            text(f"updated {now - upd:.0f}s ago", RX + RW - 14, ry_ + 8, DIM, F_S, right=True)

    # ------------------------------------------------------------------ frame
    def draw_frame(now: float) -> None:
        nonlocal prev_positions
        positions = engine.get_positions_copy()
        held = set(positions)
        if prev_positions is not None:
            for s in held - prev_positions:
                flashes[s] = (now, GREEN)
            for s in prev_positions - held:
                flashes[s] = (now, ORANGE)
        prev_positions = held
        sess = engine.cached_session or {}
        acc = engine.account_snapshot
        sym = symbols[selected]
        sig = engine.get_signal(sym) or {}
        pos = positions.get(sym)

        screen.fill(BG)

        # ---- header ----
        text("SPARTICUS", 16, 10, PURPLE, F_XL)
        if acc is not None:
            pnl_pct = engine.daily_pnl_pct
            day_usd = float(acc.equity) - float(acc.last_equity or 0)
            col = GREEN if day_usd >= 0 else RED
            text(f"${float(acc.equity):,.2f}", 190, 10, TXT, F_XL)
            text(f"{day_usd:+,.2f} ({pnl_pct * 100:+.2f}%) today", 400, 16, col, F_B)
            text(f"cash ${float(acc.cash):,.0f}   buying power ${float(acc.buying_power):,.0f}", W - 16, 18,
                 DIM, F_S, right=True)
        else:
            text("fetching account...", 190, 16, DIM, F_B)

        mode = ("PAPER" if PAPER_TRADING else "LIVE $$$") + (" MICRO" if STRATEGY.fractional else "")
        if sess.get("open"):
            m = int(sess.get("mins_to_close") or 0)
            market = f"market OPEN, {m // 60}h{m % 60:02d}m to close"
        else:
            market = "market CLOSED"
        ap_col = GREEN if engine.autopilot_enabled else YELLOW
        x = text(f"{mode}  |  {market}  |  ", 16, 48, DIM, F_M)
        x += text(f"autopilot {'ON' if engine.autopilot_enabled else 'OFF'}", 16 + x, 48, ap_col, F_B)
        cb = engine.circuit_breaker_tripped
        text(f"  |  breaker {'TRIPPED' if cb else 'armed'}", 16 + x, 48, RED if cb else DIM, F_M)

        scan_age, wd_age = now - engine.last_scan_ts, now - engine.last_watchdog_ts
        hx = W - 16
        for label, age, ok_s in (("watchdog", wd_age, 15), ("scanner", scan_age, 90)):
            col = GREEN if age < ok_s else (YELLOW if age < ok_s * 4 else RED)
            w = text(f"{label} {age:.0f}s", hx, 48, col, F_S, right=True)
            pygame.draw.circle(screen, col, (hx - w - 9, 55), 4)
            hx -= w + 30
        with engine.state_lock:
            n_entries = sum(engine.entries_today.values())
        text(f"positions {len(positions)}/{MAX_OPEN_POSITIONS}   entries today {n_entries}", 16, 70, DIM, F_S)

        # ---- brain map ----
        panel(GRID_X - 8, GRID_Y - 24, GRID_W + 16, GRID_H + 32, "BRAIN MAP   colour = trend strength (5 vs 20 SMA)")
        pulse = 0.5 + 0.5 * math.sin(now * 5.0)
        for i, s in enumerate(symbols):
            r, c = divmod(i, COLS)
            tx, ty = GRID_X + c * (TILE_W + GAP), GRID_Y + r * (TILE_H + GAP)
            sg = engine.get_signal(s) or {}
            p = positions.get(s)
            pygame.draw.rect(screen, tile_fill(sg), (tx, ty, TILE_W, TILE_H), border_radius=6)
            if p is not None:
                pl = float(p.unrealized_plpc)
                pygame.draw.rect(screen, GREEN if pl >= 0 else RED, (tx, ty, TILE_W, TILE_H), 2, border_radius=6)
            elif sg.get("signal") in ("LONG", "SHORT"):
                pygame.draw.rect(screen, lerp(NEUTRAL, CYAN, pulse), (tx, ty, TILE_W, TILE_H), 2, border_radius=6)
            if s in flashes:
                t0, fcol = flashes[s]
                fade = 1.0 - (now - t0) / 4.0
                if fade <= 0:
                    flashes.pop(s, None)
                else:
                    pygame.draw.rect(screen, lerp(BG, fcol, fade), (tx - 3, ty - 3, TILE_W + 6, TILE_H + 6),
                                     3, border_radius=8)
            if i == selected:
                pygame.draw.rect(screen, YELLOW, (tx - 2, ty - 2, TILE_W + 4, TILE_H + 4), 2, border_radius=7)
            dim_tile = sg.get("signal") in ("NODATA", "STALE") or not sg
            text(s, tx + 6, ty + 4, DIM if dim_tile else TXT, F_B)
            flags = engine.symbol_flags(s).replace("LOCK", "L").replace("MAN", "M").replace("CD", "C")
            if flags:
                text(flags.replace(" ", ""), tx + TILE_W - 5, ty + 5, YELLOW, F_S, right=True)
            if TILE_H >= 34:
                if p is not None:
                    text(f"{float(p.unrealized_plpc) * 100:+.2f}%", tx + 6, ty + TILE_H - 17,
                         GREEN if float(p.unrealized_pl) >= 0 else RED, F_S)
                elif sg.get("rsi") is not None:
                    text(f"rsi {sg['rsi']:.0f}", tx + 6, ty + TILE_H - 17, DIM, F_S)

        # ---- thought stream ----
        ts_y = GRID_Y + GRID_H + 20
        panel(GRID_X - 8, ts_y, GRID_W + 16, H - ts_y - 34, "THOUGHT STREAM")
        tag_colours = (("[SIGNAL EXIT]", ORANGE), ("[EXIT DETECTED]", ORANGE), ("[SIGNAL]", CYAN),
                       ("[FILLED]", GREEN), ("ORDER PLACED", GREEN), ("[FLATTEN", PURPLE), ("[PARTIAL", YELLOW),
                       ("[BLOCKED]", DIM), ("[STALE", DIM), ("ERROR", RED), ("WATCHDOG STOP", RED),
                       ("[HEALTH]", RED), ("[CIRCUIT BREAKER]", RED), ("SUMMARY", PURPLE))
        events = list(RECENT_EVENTS)
        max_lines = (H - ts_y - 34 - 30) // 17
        for j, ev in enumerate(reversed(events[-max_lines:])):
            col = next((c for tag, c in tag_colours if tag in ev), TXT)
            plain = "".join(ch for ch in ev if ord(ch) < 0x250).replace("  ", " ")  # emoji show as boxes
            text(plain[:74], GRID_X + 4, ts_y + 28 + j * 17, col, F_S)

        if view["mode"] == "stats":
            draw_stats_page(now, acc)
        else:
            # ---- selected ticker chart ----
            panel(RX, GRID_Y - 24, RW, 300)
            closes, s5, s20, rsis = series_for(sym, sig) if sig.get("closes") else ([], [], [], [])
            hdr = f"{sym}"
            text(hdr, RX + 12, GRID_Y - 16, YELLOW, F_XL)
            if sig.get("price"):
                text(f"${sig['price']:.2f}", RX + 110, GRID_Y - 10, TXT, F_B)
                text("5-SMA", RX + 230, GRID_Y - 10, CYAN, F_S)
                text("20-SMA", RX + 280, GRID_Y - 10, ORANGE, F_S)
                text(f"last {len(closes)} min", RX + RW - 12, GRID_Y - 10, DIM, F_S, right=True)
            cx, cy, cw, ch = RX + 12, GRID_Y + 22, RW - 80, 240
            if len(closes) >= 2:
                vals = [v for v in closes + s5 + s20 if v is not None]
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.12, closes[-1] * 0.0008)
                lo, hi = lo - pad, hi + pad
                for k in range(5):
                    gy = cy + int(ch * k / 4)
                    pygame.draw.line(screen, FAINT, (cx, gy), (cx + cw, gy), 1)
                    text(f"{hi - (hi - lo) * k / 4:.2f}", cx + cw + 6, gy - 7, DIM, F_S)
                line_chart(cx, cy, cw, ch, [(closes, TXT, 2), (s5, CYAN, 1), (s20, ORANGE, 1)], lo, hi)
                if pos is not None:
                    entry = float(pos.avg_entry_price)
                    is_long = float(pos.qty) > 0
                    planned = engine.journal.open.get(sym) or {}
                    if planned.get("tp") and planned.get("sl"):
                        tp, sl = float(planned["tp"]), float(planned["sl"])
                    else:
                        tp, sl = bracket_levels(entry, is_long, STRATEGY, sig.get("atr"))
                    notes = []
                    for name, level, col in (("entry", entry, YELLOW), ("TP", tp, GREEN), ("SL", sl, RED)):
                        if lo <= level <= hi:
                            ly = int(cy + ch - (level - lo) / (hi - lo) * ch)
                            dashed(col, cx, cx + cw, ly)
                            text(f"{name} {level:.2f}", cx + 4, ly - 15, col, F_S)
                        else:
                            notes.append((f"{name} {level:.2f} {'above' if level > hi else 'below'} chart", col))
                    for k, (n, col) in enumerate(notes):
                        text(n, cx + cw - 4, cy + 4 + k * 15, col, F_S, right=True)
            else:
                text("waiting for the first scan of this ticker...", cx, cy + ch // 2, DIM, F_M)

            # ---- RSI ----
            ry, rh = GRID_Y + 284, 78
            panel(RX, ry, RW, rh, f"RSI {sig['rsi']:.1f}" if sig.get("rsi") is not None else "RSI")
            rx, rw, rpy, rph = RX + 12, RW - 80, ry + 24, rh - 32
            band_top = rpy + rph - MAX_RSI_ENTRY / 100 * rph
            band_bot = rpy + rph - MIN_RSI_THRESHOLD / 100 * rph
            pygame.draw.rect(screen, (24, 46, 38), (rx, int(band_top), rw, int(band_bot - band_top)))
            for lvl in (MIN_RSI_THRESHOLD, MAX_RSI_ENTRY):
                ly = int(rpy + rph - lvl / 100 * rph)
                text(f"{lvl:.0f}", rx + rw + 6, ly - 7, DIM, F_S)
            if rsis:
                line_chart(rx, rpy, rw, rph, [(rsis, PURPLE, 2)], 0.0, 100.0)

            # ---- decision pipeline ----
            dy, dh = ry + rh + 10, 196
            panel(RX, dy, RW, dh, "DECISION PIPELINE   what the autopilot thinks about this ticker right now")
            # Each gate: (label, dot colour, live detail). Entry gates are green/red (pass/fail).
            # Holding gates use grey for "not triggered", so red never means "fine".
            gates: List[Tuple[str, Tuple[int, int, int], str]] = []
            verdict, vcol = "", DIM
            if not sig or sig.get("signal") == "NODATA":
                verdict = "NO DATA YET: not enough 1-minute bars for this ticker."
            elif pos is not None:
                is_long = float(pos.qty) > 0
                auto = engine.is_auto_managed(sym)
                with engine.state_lock:
                    t_in = engine.entry_times.get(sym)
                hold_left = max(0.0, MIN_HOLD_SECONDS - (now - t_in)) if t_in else 0.0
                reversed_ = (sig["fast"] < sig["slow"]) if is_long else (sig["fast"] > sig["slow"])
                plpc = float(pos.unrealized_plpc)
                gates = [
                    ("Managed by autopilot", GREEN if auto else YELLOW,
                     "yes" if auto else "manual or locked: no signal exits"),
                    ("Minimum hold passed", GREEN if hold_left <= 0 else DIM,
                     f"{hold_left / 60:.1f} min left" if hold_left else "yes"),
                    ("Exit trigger: trend reversed", ORANGE if reversed_ else DIM,
                     f"5-SMA {sig['fast']:.2f} vs 20-SMA {sig['slow']:.2f}"),
                    ("Watchdog stop", GREEN if plpc > HARD_WATCHDOG_LOSS_PCT else RED,
                     f"{plpc * 100:+.2f}% (stops at {HARD_WATCHDOG_LOSS_PCT * 100:.0f}%)"),
                ]
                if auto and reversed_ and hold_left <= 0:
                    verdict, vcol = "WILL EXIT at the next scan: the trend has reversed.", ORANGE
                elif auto and reversed_:
                    verdict, vcol = f"TREND REVERSED: exits once the minimum hold ends ({hold_left / 60:.1f} min).", ORANGE
                elif not auto:
                    verdict, vcol = "HOLDING (manual): bracket TP/SL and end-of-day flatten still apply.", YELLOW
                else:
                    verdict, vcol = "HOLDING: exits when the 5-SMA crosses back below the 20-SMA.", GREEN
            else:
                fast, slow, rsi = sig.get("fast"), sig.get("slow"), sig.get("rsi")
                fresh = not sig.get("stale")
                up = bool(fast and slow and fast > slow)
                cross = bool(sig.get("crossed_up")) or not REQUIRE_FRESH_CROSS
                rsi_ok = rsi is not None and MIN_RSI_THRESHOLD <= rsi <= MAX_RSI_ENTRY
                allowed, why = (engine.auto_entry_allowed(sym, sess) if sess else (False, "session unknown"))
                checks = [
                    ("Fresh data", fresh, f"last bar {sig.get('bar_age', 0):.0f}s old",
                     "the latest price bar is too old"),
                    ("Uptrend: 5-SMA above 20-SMA", up, f"{fast:.2f} vs {slow:.2f}" if fast and slow else "-",
                     "no uptrend yet (5-SMA is below the 20-SMA)"),
                    (f"Fresh cross (last {CROSS_LOOKBACK_BARS} bars)", cross,
                     "just crossed" if sig.get("crossed_up") else "no recent cross",
                     "it's trending, but the cross wasn't recent; it only buys fresh crosses"),
                    (f"RSI between {MIN_RSI_THRESHOLD:.0f} and {MAX_RSI_ENTRY:.0f}", rsi_ok,
                     f"{rsi:.1f}" if rsi is not None else "-",
                     "RSI too hot to chase" if rsi is not None and rsi > MAX_RSI_ENTRY else "momentum too weak (RSI)"),
                    ("Allowed to enter now", allowed, "yes" if allowed else why, f"blocked: {why}"),
                ]
                gates = [(label, GREEN if ok else RED, detail) for label, ok, detail, _ in checks]
                failed = next((c for c in checks if not c[1]), None)
                if failed is None:
                    verdict, vcol = "WOULD BUY at the next scan (spread and buying power checked at order time).", GREEN
                else:
                    verdict, vcol = f"WAITING: {failed[3]}.", DIM
            for k, (label, dot, detail) in enumerate(gates):
                gy = dy + 30 + k * 26
                pygame.draw.circle(screen, dot, (RX + 22, gy + 8), 6)
                text(label, RX + 38, gy, TXT, F_M)
                text(detail[:40], RX + RW - 14, gy, DIM, F_M, right=True)
            text(verdict[:84], RX + 14, dy + dh - 28, vcol, F_B)

            # ---- equity curve ----
            ey = dy + dh + 10
            eh = H - ey - 34
            with engine.state_lock:
                hist = list(engine.equity_history)
            panel(RX, ey, RW, eh, "TODAY'S EQUITY")
            if acc is not None and len(hist) >= 2:
                base = float(acc.last_equity or hist[0][1])
                vals = [e for _, e in hist]
                lo, hi = min(vals + [base]), max(vals + [base])
                pad = max((hi - lo) * 0.15, base * 0.0005)
                lo, hi = lo - pad, hi + pad
                ex, ew, epy, eph = RX + 12, RW - 80, ey + 26, eh - 34
                by = int(epy + eph - (base - lo) / (hi - lo) * eph)
                dashed(DIM, ex, ex + ew, by)
                text("prev close", ex + ew + 62, by - 18, DIM, F_S, right=True)
                line_chart(ex, epy, ew, eph, [(vals, GREEN if vals[-1] >= base else RED, 2)], lo, hi)
            else:
                text("collecting equity points (one every 15 s)...", RX + 14, ey + 40, DIM, F_M)

        # ---- footer ----
        keys = "[TAB] stats  [arrows/click] select  [B] buy  [C] close  [L] lock  [A] autopilot  [Shift+X] panic"
        if ENABLE_SHORTS:
            keys = keys.replace("[C] close", "[S] short  [C] close")
        kw = text(keys, 16, H - 24, DIM, F_S)
        room = max(10, int((W - 16 - kw - 40) / 7.3))
        text(engine.get_status()[:room], W - 16, H - 24, CYAN, F_S, right=True)

    # ------------------------------------------------------------------ loop
    running = True
    while running:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                c, r = (mx - GRID_X) // (TILE_W + GAP), (my - GRID_Y) // (TILE_H + GAP)
                if 0 <= c < COLS and r >= 0 and r * COLS + c < len(symbols):
                    selected = r * COLS + c
            elif event.type == pygame.KEYDOWN:
                s = symbols[selected]
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                if event.key == pygame.K_TAB:
                    view["mode"] = "stats" if view["mode"] == "brain" else "brain"
                elif event.key == pygame.K_LEFT:
                    selected = (selected - 1) % len(symbols)
                elif event.key == pygame.K_RIGHT:
                    selected = (selected + 1) % len(symbols)
                elif event.key == pygame.K_UP:
                    selected = (selected - COLS) % len(symbols)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + COLS) % len(symbols)
                elif event.key == pygame.K_b:
                    spawn(engine.execute_entry, s, OrderSide.BUY, "manual")
                elif event.key == pygame.K_s:
                    if ENABLE_SHORTS:
                        spawn(engine.execute_entry, s, OrderSide.SELL, "manual")
                    else:
                        engine.set_status("Shorts disabled. Set ENABLE_SHORTS = True to use [S].")
                elif event.key == pygame.K_c:
                    spawn(engine.close_position_safely, s, "manual close", "manual")
                elif event.key == pygame.K_l:
                    engine.toggle_lock(s)
                elif event.key == pygame.K_a:
                    engine.toggle_autopilot()
                elif event.key == pygame.K_x and shift:
                    spawn(engine.flatten_all_positions, "manual panic", True)
        try:
            draw_frame(now)
        except Exception as e:
            if now - last_draw_error > 60:
                last_draw_error = now
                log_event(f"[HUD] Drawing error, trading unaffected: {e}", logging.WARNING)
        pygame.display.flip()
        clock.tick(20)

    pygame.quit()


# ==========================================
# 6. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Sparticus {BOT_VERSION} trading engine")
    parser.add_argument("--headless", action="store_true", help="Run without the pygame HUD")
    parser.add_argument("--no-autopilot", action="store_true",
                        help="Start with autopilot OFF (watch the HUD, press [A] to enable it)")
    args = parser.parse_args()

    acquire_single_instance_lock()

    mode = "PAPER" if PAPER_TRADING else "LIVE"
    autopilot_start = AUTOPILOT_ON_START and not args.no_autopilot
    DISCORD.self_test()
    log_event(f"Starting Sparticus {BOT_VERSION} ({mode}, feed={DATA_FEED.value}, "
              f"autopilot={'ON' if autopilot_start else 'OFF'}, shorts={ENABLE_SHORTS})...")
    log_event(f"🚀 Sparticus {BOT_VERSION} online ({mode}). Every action is mirrored to this channel.")

    symbols_universe = list(DEFAULT_UNIVERSE)  # edit the watchlist in sparticus_strategy.py

    engine = SparticusEngineV55()
    engine.autopilot_enabled = autopilot_start
    start_positions: Dict[str, Any] = {}
    try:
        _, start_positions = engine.refresh_snapshot()
        if start_positions:
            held = ", ".join(f"{s} {float(p.qty):g} sh (P&L ${float(p.unrealized_pl):+,.2f})"
                             for s, p in sorted(start_positions.items()))
            log_event(f"[STARTUP] Already holding: {held}. These were opened before this start; "
                      f"the watchdog stop, signal exits and end-of-day flatten will manage them.",
                      logging.WARNING)
    except Exception as e:
        log_event(f"[WARN] Initial account snapshot failed: {e}", logging.WARNING)

    for sym, pos_ in start_positions.items():
        engine.journal.note_fill(sym, pos_)   # positions opened before this start: tracked as "carried"

    if STRATEGY.fractional:
        tradable, skipped = fractionable_only(engine.trading_client, symbols_universe)
        symbols_universe[:] = tradable
        log_event(f"[MICRO] Micro mode: fractional shares, {STRATEGY.position_pct_of_equity:.0%} of equity per "
                  f"position, up to {MAX_OPEN_POSITIONS} positions, long only. Scanning {len(tradable)} tickers"
                  + (f"; skipped (not fractionable): {', '.join(skipped)}." if skipped else "."))

    if not symbols_universe:
        log_event("🚨 [CONFIG] The watchlist is empty (micro mode found no fractionable tickers). "
                  "Fix DEFAULT_UNIVERSE in sparticus_strategy.py.", logging.CRITICAL)
        DISCORD.shutdown()
        sys.exit(4)

    log_event(f"[CONFIG] {STRATEGY.strategy} config {CONFIG_ID}. Lab baseline: "
              f"{'found' if engine.expectations else 'none yet (run backtest.py)'}.")
    killed = incubation.read_kill(KILL_FILE)
    if killed and killed.get("config_id") == CONFIG_ID:
        engine.killed = True
        engine.autopilot_enabled = False
        log_event(f"🚨 [KILL SWITCH] Still active for config {CONFIG_ID} since {killed.get('when')}: "
                  f"{killed.get('reason')} Autopilot stays OFF. Delete {KILL_FILE} to re-arm.", logging.CRITICAL)
    elif killed:
        log_event(f"[KILL SWITCH] {KILL_FILE} is for an older config ({killed.get('config_id')}); "
                  f"config {CONFIG_ID} starts a fresh incubation.", logging.WARNING)

    def stats_worker() -> None:
        while True:
            try:
                engine.refresh_stats()
            except Exception as e:
                logging.warning(f"[STATS] refresh failed: {e}")
            time.sleep(STATS_REFRESH_SECONDS)
    threading.Thread(target=stats_worker, daemon=True, name="stats").start()

    workers = {
        "watchdog": threading.Thread(target=watchdog_worker, args=(engine,), daemon=True, name="watchdog"),
        "scanner": threading.Thread(target=scanner_worker, args=(engine, symbols_universe), daemon=True,
                                    name="scanner"),
    }
    for t in workers.values():
        t.start()
    threading.Thread(target=health_monitor, args=(engine, workers, args.headless), daemon=True,
                     name="health").start()

    def _warn_open_positions_on_exit() -> None:
        try:
            held = sorted(p.symbol for p in engine.trading_client.get_all_positions())
        except Exception:
            held = sorted(engine.get_positions_copy())
        if held:
            log_event(f"🚨 [SHUTDOWN] Sparticus stopped with open positions: {', '.join(held)}. Nothing will "
                      f"close them at 3:55 and their brackets expire at 4:00 PM ET. Restart Sparticus "
                      f"(it sells leftovers after the open) or close them in the Alpaca dashboard.",
                      logging.CRITICAL)
    atexit.register(_warn_open_positions_on_exit)  # runs before the Discord flush (atexit is LIFO)

    for _sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if _sig is not None:
            signal.signal(_sig, _on_stop_signal)
    why = "HUD window closed"
    try:
        if args.headless:
            log_event("[HEADLESS] Running without HUD. Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            run_pygame_hud(engine, symbols_universe)
    except KeyboardInterrupt:
        why = "Ctrl+C"
    except StopRequested as e:
        why = str(e)
    for _sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):   # let the flatten finish
        if _sig is not None:
            signal.signal(_sig, signal.SIG_IGN)
    flatten_before_quit(engine, why)
