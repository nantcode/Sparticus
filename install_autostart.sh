#!/bin/bash
# macOS only: start Sparticus automatically every weekday morning, so no trading day gets missed.
#
#   ./install_autostart.sh          start at 9:20 AM (your Mac's clock) Monday to Friday
#   ./install_autostart.sh 9 15     pick another time: hour minute
#   ./uninstall_autostart.sh        turn it off again
#
# Run it from the same Terminal you normally start Sparticus in (with your .venv active), so it
# remembers which Python has alpaca-py installed. It runs headless (no window); Discord and the
# log file work as usual. If Sparticus is already running at that time, the second copy just exits.

cd "$(dirname "$0")" || exit 1
[ "$(uname)" = "Darwin" ] || { echo "Autostart uses macOS launchd; this isn't a Mac."; exit 1; }

HOUR="${1:-9}"
MINUTE="${2:-20}"
DIR="$(pwd)"
PY="$(command -v python3)"
LABEL="com.sparticus.autostart"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if ! "$PY" -c "import alpaca, dotenv" 2>/dev/null; then
  echo "The python3 in this Terminal ($PY) can't import alpaca-py."
  echo "Activate the environment you run Sparticus with (for example: source .venv/bin/activate), then run this again."
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"
"$PY" - "$PLIST" "$DIR" "$PY" "$HOUR" "$MINUTE" "$LABEL" <<'PYEOF'
import plistlib, sys
plist, folder, py, hour, minute, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
job = {
    "Label": label,
    "ProgramArguments": ["/bin/bash", f"{folder}/run_sparticus.sh"],
    "WorkingDirectory": folder,
    "EnvironmentVariables": {"SPARTICUS_PYTHON": py,
                             "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"},
    "StartCalendarInterval": [{"Weekday": day, "Hour": hour, "Minute": minute} for day in range(1, 6)],
    "StandardOutPath": f"{folder}/sparticus_autostart.log",
    "StandardErrorPath": f"{folder}/sparticus_autostart.log",
}
with open(plist, "wb") as f:
    plistlib.dump(job, f)
PYEOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
if ! launchctl bootstrap "gui/$(id -u)" "$PLIST"; then
  echo "launchctl couldn't load $PLIST. Nothing was started."
  exit 1
fi

WAKE_MIN=$(( MINUTE >= 5 ? MINUTE - 5 : 0 ))
printf "Autostart is ON: Sparticus starts at %d:%02d Monday to Friday (headless).\n" "$HOUR" "$MINUTE"
echo "Output: sparticus_autostart.log in this folder. Turn it off any time: ./uninstall_autostart.sh"
echo
echo "The Mac has to be on, logged in and awake at that time (plugged in, lid open). To wake it"
echo "automatically every weekday, run this once (it asks for your Mac password):"
printf "   sudo pmset repeat wakeorpoweron MTWRF %02d:%02d:00\n" "$HOUR" "$WAKE_MIN"
