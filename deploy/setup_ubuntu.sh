#!/usr/bin/env bash
# One-time setup of a Hermes call server on Ubuntu 24.04 (Python 3.12).
#
#   curl -fsSL https://raw.githubusercontent.com/alimounim/Hermes/main/deploy/setup_ubuntu.sh | sudo bash
#   (or copy this file to the server and run: sudo bash setup_ubuntu.sh)
#
# Installs Python + Node, creates a `hermes` user, clones the repo, installs
# dependencies and the two systemd services. It does NOT touch API keys: put
# them in /home/hermes/Hermes/.env.local afterwards (see deploy/README.md).
# Safe to run again: it updates the code and reinstalls dependencies.
set -euo pipefail

REPO_URL="${HERMES_REPO:-https://github.com/alimounim/Hermes.git}"
BRANCH="${HERMES_BRANCH:-main}"
APP_USER=hermes
APP_HOME=/home/$APP_USER
APP_DIR=$APP_HOME/Hermes

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

PYV=$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])' 2>/dev/null || echo 0)
if [[ $PYV -lt 311 || $PYV -gt 313 ]]; then
  echo "Hermes needs Python 3.11-3.13 (Ubuntu 24.04 has 3.12). This server has $(python3 -V 2>&1)." >&2
  exit 1
fi

echo "==> System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
# libgl1/libglib2.0-0: OpenCV + MediaPipe on a headless server.
apt-get install -y -q git curl ca-certificates python3 python3-venv python3-pip \
  libgl1 libglib2.0-0 libportaudio2

if ! command -v node >/dev/null || [[ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]]; then
  echo "==> Node.js 22 (for the Chert bridge)"
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -q nodejs
fi

echo "==> App user and code"
id -u $APP_USER >/dev/null 2>&1 || useradd --create-home --shell /bin/bash $APP_USER
if [[ -d $APP_DIR/.git ]]; then
  sudo -u $APP_USER git -C $APP_DIR fetch -q origin
  sudo -u $APP_USER git -C $APP_DIR checkout -q $BRANCH
  sudo -u $APP_USER git -C $APP_DIR pull -q --ff-only origin $BRANCH
else
  sudo -u $APP_USER git clone -q --branch $BRANCH $REPO_URL $APP_DIR
fi

echo "==> Python environment (a few minutes)"
sudo -u $APP_USER python3 -m venv $APP_DIR/.venv
sudo -u $APP_USER $APP_DIR/.venv/bin/pip install -q --upgrade pip
sudo -u $APP_USER $APP_DIR/.venv/bin/pip install -q -r $APP_DIR/requirements.txt
# Voice-activity model for the agent (needs no API keys).
sudo -u $APP_USER bash -c "cd $APP_DIR && .venv/bin/python agent.py download-files"

echo "==> systemd services"
install -m 644 $APP_DIR/deploy/hermes-agent.service /etc/systemd/system/hermes-agent.service
install -m 644 $APP_DIR/deploy/hermes-chert.service /etc/systemd/system/hermes-chert.service
systemctl daemon-reload

if [[ ! -f $APP_DIR/.env.local ]]; then
  sudo -u $APP_USER cp $APP_DIR/.env.example $APP_DIR/.env.local
  chmod 600 $APP_DIR/.env.local
  echo
  echo "Next: put your keys in $APP_DIR/.env.local, then follow deploy/README.md"
  echo "(sign Chert in, then: systemctl enable --now hermes-agent hermes-chert)."
else
  systemctl restart hermes-agent || true
  systemctl is-enabled -q hermes-chert && systemctl restart hermes-chert || true
  echo
  echo "Updated and restarted. Check with: systemctl status hermes-agent hermes-chert"
fi
