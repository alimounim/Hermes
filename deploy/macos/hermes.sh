#!/usr/bin/env bash
# Run Hermes on this Mac as background services (launchd), so FaceTime calls
# are answered without terminal windows.
#
#   deploy/macos/hermes.sh on       install (first time) and start both services
#   deploy/macos/hermes.sh off      stop both and free the FaceTime line
#   deploy/macos/hermes.sh status   is everything running?
#   deploy/macos/hermes.sh logs     follow both logs (Ctrl-C to stop watching)
#   deploy/macos/hermes.sh remove   stop and uninstall the services
#
# Services start again at login and restart within seconds after a crash.
# While running, the Mac is kept from idle-sleeping (caffeinate). Closing the
# lid still sleeps a MacBook unless an external display is attached.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="$HOME/Library/Logs/Hermes"
UID_NUM="$(id -u)"
AGENT_LABEL=com.hermes.agent
CHERT_LABEL=com.hermes.chert
CHERT_VERSION="0.1.0-beta.21"
ENV_FILE="$REPO/.env"
[[ -f "$REPO/.env.local" ]] && ENV_FILE="$REPO/.env.local"
NODE_DIR="$(dirname "$(command -v node || echo /usr/local/bin/node)")"

plist() { echo "$AGENTS/$1.plist"; }

write_plists() {
  mkdir -p "$AGENTS" "$LOGS"
  cat >"$(plist $AGENT_LABEL)" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$AGENT_LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$REPO/.venv/bin/python</string><string>agent.py</string><string>start</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key><dict>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>PATH</key><string>$NODE_DIR:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$LOGS/agent.log</string>
  <key>StandardErrorPath</key><string>$LOGS/agent.log</string>
</dict></plist>
EOF
  cat >"$(plist $CHERT_LABEL)" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$CHERT_LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$NODE_DIR/npx</string><string>--yes</string>
    <string>@trychert/facetime@$CHERT_VERSION</string>
    <string>dev</string><string>--agent</string><string>hermes-agent</string>
    <string>--env-file</string><string>$ENV_FILE</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key><dict>
    <key>HOME</key><string>$HOME</string>
    <key>PATH</key><string>$NODE_DIR:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$LOGS/chert.log</string>
  <key>StandardErrorPath</key><string>$LOGS/chert.log</string>
</dict></plist>
EOF
}

loaded() { launchctl print "gui/$UID_NUM/$1" >/dev/null 2>&1; }

start_one() {
  loaded "$1" && launchctl bootout "gui/$UID_NUM/$1" 2>/dev/null || true
  launchctl bootstrap "gui/$UID_NUM" "$(plist "$1")"
}

stop_one() {
  loaded "$1" && launchctl bootout "gui/$UID_NUM/$1" 2>/dev/null || true
}

port_busy() {
  local pid
  pid=$(lsof -t -nP -iTCP:8081 -sTCP:LISTEN 2>/dev/null | head -1)
  [[ -n "$pid" ]] && ! loaded $AGENT_LABEL && echo "$pid"
}

status() {
  for l in $AGENT_LABEL $CHERT_LABEL; do
    if loaded "$l"; then
      pid=$(launchctl print "gui/$UID_NUM/$l" | awk '/^\tpid =/{print $3}')
      echo "$l: running${pid:+ (pid $pid)}"
    else
      echo "$l: stopped"
    fi
  done
  echo "logs: $LOGS"
}

case "${1:-status}" in
  on)
    if pid=$(port_busy); then
      echo "Another Hermes agent is already running (pid $pid). Stop it first (Ctrl-C in its terminal)." >&2
      exit 1
    fi
    if pgrep -f "chert dev" >/dev/null && ! loaded $CHERT_LABEL; then
      echo "A 'chert dev' is already running in a terminal. Stop it first (Ctrl-C twice)." >&2
      exit 1
    fi
    write_plists
    start_one $AGENT_LABEL
    sleep 3
    start_one $CHERT_LABEL
    echo "Hermes is on. Call the line on FaceTime Video."
    status
    ;;
  off)
    stop_one $CHERT_LABEL
    stop_one $AGENT_LABEL
    echo "Hermes is off. The FaceTime line is free."
    ;;
  status) status ;;
  logs) mkdir -p "$LOGS"; touch "$LOGS/agent.log" "$LOGS/chert.log"; tail -n 40 -f "$LOGS/agent.log" "$LOGS/chert.log" ;;
  remove)
    stop_one $CHERT_LABEL
    stop_one $AGENT_LABEL
    rm -f "$(plist $AGENT_LABEL)" "$(plist $CHERT_LABEL)"
    echo "Removed the Hermes services."
    ;;
  *) echo "usage: $0 on|off|status|logs|remove" >&2; exit 2 ;;
esac
