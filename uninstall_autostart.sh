#!/bin/bash
# Turns off the weekday autostart. If autostart launched the copy of Sparticus that's running now,
# that copy stops too (during market hours it closes its positions first).

LABEL="com.sparticus.autostart"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null; then
  echo "Stopped the autostart job."
else
  echo "Autostart wasn't loaded."
fi
rm -f "$PLIST"
echo "Autostart is OFF. (To stop waking the Mac too: sudo pmset repeat cancel)"
