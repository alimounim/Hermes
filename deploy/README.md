# Run Hermes without terminal windows

## On your Mac (simplest)

`deploy/macos/hermes.sh` runs the agent and the Chert bridge as macOS
background services: they start at login, restart within seconds after a
crash, and keep the Mac from idle-sleeping while they run. Stop any
`agent.py` or `chert dev` you started in a terminal first.

```bash
deploy/macos/hermes.sh on       # start (installs the services the first time)
deploy/macos/hermes.sh status   # running?
deploy/macos/hermes.sh logs     # follow the logs (~/Library/Logs/Hermes)
deploy/macos/hermes.sh off      # stop and free the FaceTime line
```

Keep the Mac plugged in, online and open: closing a MacBook's lid still
puts it to sleep unless an external display is attached. The Chert sign-in
lasts 30 days; if the bridge log says you are signed out, run
`npx @trychert/facetime@0.1.0-beta.21 login`, then `hermes.sh on` again.

## On a Linux server

This keeps the FaceTime line answering without your laptop. Two services run on
one Linux server:

| Service | What it does |
|---|---|
| `hermes-agent` | `python agent.py start`: recognizes signs, speaks, draws the caption board |
| `hermes-chert` | `chert dev`: routes your FaceTime line to the agent |

## 1. Get a server

An **Ubuntu 24.04** machine (22.04 is too old: its Python 3.10 cannot install the packages) with **2 CPUs and 4 GB RAM** or more
(one call uses about 1 GB). For example a DigitalOcean "Basic 4 GB" droplet,
a Hetzner CX22/CPX21, or an AWS t3.medium. No inbound ports are needed: both
services only make outgoing connections.

## 2. Install

SSH in and run:

```bash
curl -fsSL https://raw.githubusercontent.com/alimounim/Hermes/main/deploy/setup_ubuntu.sh | sudo bash
```

This installs Python, Node, a `hermes` user, the code in `/home/hermes/Hermes`,
its packages and both services. It takes a few minutes.

## 3. Add your keys

```bash
sudo -u hermes nano /home/hermes/Hermes/.env.local
```

Fill in the same values as your laptop's `.env` / `.env.local`:
`GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
`LIVEKIT_API_SECRET` and `AGENT_NAME=hermes-agent`. Type them in on the server;
don't send them through chat or email.

## 4. Sign the server in to Chert

```bash
sudo -u hermes npx --yes @trychert/facetime@0.1.0-beta.21 login --device
```

It prints an address and a code: open the address on your phone or laptop and
approve. The sign-in lasts **30 days**; repeat this step monthly (step 6 shows
how to tell).

Then check everything:

```bash
cd /home/hermes/Hermes && sudo -u hermes npx --yes @trychert/facetime@0.1.0-beta.21 doctor --env-file .env.local
```

## 5. Start

**Stop `agent.py` and `chert dev` on your laptop first**: only one session can
hold the line, and two agents with the same name split the calls.

```bash
sudo systemctl enable --now hermes-agent hermes-chert
```

Both start again on their own after a crash or a reboot. Call the line on
FaceTime Video to test.

## 6. Day to day

| Task | Command |
|---|---|
| Is it running? | `systemctl status hermes-agent hermes-chert` |
| Live log | `journalctl -u hermes-agent -u hermes-chert -f` |
| Update to the latest code | rerun the install command from step 2 |
| Free the line (stop answering) | `sudo systemctl stop hermes-chert` |
| Take it back | `sudo systemctl start hermes-chert` |
| Chert sign-in expired | step 4, then `sudo systemctl restart hermes-chert` |

The trial line is shared with other teams and answers one call at a time. If
someone else holds it, `hermes-chert` keeps retrying every 15 seconds and the
log shows `line_in_use`.

`HERMES_IDLE_PROCESSES` (default 1) sets how many agent processes wait warm for
a call. Raise it only on a server with more memory.
