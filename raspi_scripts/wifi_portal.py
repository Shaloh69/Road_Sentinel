#!/usr/bin/env python3
"""
Road Sentinel — WiFi provisioning portal for headless Raspberry Pis.

The Pis sit at the Busay installation with no monitor or keyboard. If the
site WiFi changes (new router, new password, moved network), the Pi silently
drops off and there is no way to fix it without physically bringing a screen
to it. This turns a phone into that screen.

Behaviour
---------
1. Watch the WiFi connection. While it is healthy, do nothing.
2. If it fails FAIL_THRESHOLD consecutive checks (default 5), stop trying and
   raise an access point: SSID "RoadSentinel-Setup", password "roadsentinel".
3. Connect a phone to that AP. A captive-portal prompt appears automatically
   (the same "Sign in to network" sheet a hotel WiFi shows). If it does not,
   browse to http://10.42.0.1
4. The page lists nearby networks. Pick one, enter the password, submit.
5. The Pi tears down the AP, joins the chosen network, and confirms. On
   failure it brings the portal back up so you can retry.
6. The onboard LED reports state throughout, so the Pi is diagnosable at a
   glance with no network at all.

LED patterns (onboard ACT LED)
------------------------------
    connected     brief blink every 5s   "alive, online"
    connecting    1 Hz even blink
    portal/AP     4 Hz fast blink        "waiting for you to configure me"
    failed        two quick blinks, pause

Why this and not comitup / RaspiWiFi
------------------------------------
Both are good and more general, but they take over WiFi management wholesale,
which would fight with the Tailscale + systemd setup already on these Pis.
This is deliberately narrower: it stays out of the way entirely while the
network is fine, and only intervenes after a sustained, repeated failure.
It also adds the two things neither offers — a failure-count trigger rather
than boot-time-only setup, and LED status signalling.

Requires NetworkManager (standard on Raspberry Pi OS Bookworm). Python
stdlib only, so there is nothing to pip install. Must run as root, for
nmcli and the LED sysfs nodes.

Usage:
    sudo python3 wifi_portal.py                  # run the watcher
    sudo python3 wifi_portal.py --portal-now     # force the portal up
    sudo python3 wifi_portal.py --status         # print state and exit
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# ── Configuration ──────────────────────────────────────────────────────────────

AP_SSID = os.environ.get("PORTAL_AP_SSID", "RoadSentinel-Setup")
AP_PASSWORD = os.environ.get("PORTAL_AP_PASSWORD", "roadsentinel")
AP_CONN_NAME = "roadsentinel-portal"

# NetworkManager's shared mode always hands out 10.42.0.1 to the host.
AP_ADDRESS = "10.42.0.1"
PORTAL_PORT = 80

FAIL_THRESHOLD = int(os.environ.get("PORTAL_FAIL_THRESHOLD", "5"))
CHECK_INTERVAL = int(os.environ.get("PORTAL_CHECK_INTERVAL", "20"))
PORTAL_TIMEOUT = int(os.environ.get("PORTAL_TIMEOUT", "900"))  # 15 min

# Reaching the internet is not required — associating with the LAN is enough,
# since the Pi's job is to reach the server, which may itself be LAN-local.
CONNECTIVITY_HOSTS = [("1.1.1.1", 53), ("8.8.8.8", 53)]

log = logging.getLogger("wifi-portal")


# ── Shell helpers ──────────────────────────────────────────────────────────────

def run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a command, never raise. Returns (rc, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:  # noqa: BLE001 - never let a shell call kill the watcher
        return 1, "", str(exc)


def wifi_device() -> str:
    """First WiFi interface NetworkManager knows about."""
    rc, out, _ = run(["nmcli", "-t", "-f", "DEVICE,TYPE", "device"])
    if rc == 0:
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[1] == "wifi":
                return parts[0]
    return "wlan0"


# ── Onboard LED ────────────────────────────────────────────────────────────────

class StatusLed:
    """
    Drives the Pi's onboard ACT LED to signal network state.

    Writing to /sys/class/leds requires taking the LED off its default
    trigger (mmc0/heartbeat). We restore the trigger on exit so the Pi's
    normal disk-activity indication comes back.
    """

    CANDIDATES = ("ACT", "led0", "activity", "PWR", "led1")

    def __init__(self) -> None:
        self.path: str | None = None
        self._orig_trigger: str | None = None
        base = "/sys/class/leds"
        if os.path.isdir(base):
            available = os.listdir(base)
            for name in self.CANDIDATES:
                if name in available:
                    self.path = os.path.join(base, name)
                    break
            else:
                if available:
                    self.path = os.path.join(base, available[0])

        if self.path:
            self._orig_trigger = self._read_trigger()
            self._write("trigger", "none")
            log.info("Status LED: %s (was trigger=%s)", self.path, self._orig_trigger)
        else:
            log.warning("No onboard LED found — status signalling disabled")

        self._state = "connecting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _read_trigger(self) -> str | None:
        try:
            with open(os.path.join(self.path or "", "trigger")) as fh:  # type: ignore[arg-type]
                m = re.search(r"\[(\w+)\]", fh.read())
                return m.group(1) if m else None
        except OSError:
            return None

    def _write(self, node: str, value: str) -> None:
        if not self.path:
            return
        try:
            with open(os.path.join(self.path, node), "w") as fh:
                fh.write(value)
        except OSError:
            pass

    def _on(self, on: bool) -> None:
        self._write("brightness", "1" if on else "0")

    def set(self, state: str) -> None:
        self._state = state

    def _loop(self) -> None:
        while not self._stop.is_set():
            st = self._state
            if st == "portal":                      # fast 4 Hz — needs attention
                self._on(True); time.sleep(0.12)
                self._on(False); time.sleep(0.12)
            elif st == "connecting":                # even 1 Hz
                self._on(True); time.sleep(0.5)
                self._on(False); time.sleep(0.5)
            elif st == "failed":                    # double blink, then pause
                for _ in range(2):
                    self._on(True); time.sleep(0.1)
                    self._on(False); time.sleep(0.15)
                time.sleep(1.2)
            else:                                   # connected — quiet heartbeat
                self._on(True); time.sleep(0.06)
                self._on(False); time.sleep(5.0)

    def close(self) -> None:
        self._stop.set()
        self._on(False)
        if self._orig_trigger:
            self._write("trigger", self._orig_trigger)


# ── Connectivity ───────────────────────────────────────────────────────────────

def wifi_is_up(dev: str) -> bool:
    """Associated with a network AND able to open a socket outward."""
    rc, out, _ = run(["nmcli", "-t", "-f", "DEVICE,STATE", "device"])
    associated = any(
        line.split(":")[0] == dev and line.split(":")[1] == "connected"
        for line in out.splitlines()
        if ":" in line
    )
    if not associated:
        return False

    for host, port in CONNECTIVITY_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=4):
                return True
        except OSError:
            continue
    # Associated but no route out. Still "up" for our purposes — the server
    # may be LAN-local, and flapping into portal mode over a transient
    # internet outage would be worse than leaving it alone.
    return True


def scan_networks(dev: str) -> list[dict]:
    run(["nmcli", "device", "wifi", "rescan"], timeout=25)
    time.sleep(2)
    rc, out, _ = run(["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"])
    seen: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) < 3:
            continue
        ssid = parts[0].strip()
        if not ssid or ssid in seen:
            continue
        try:
            signal = int(parts[1])
        except ValueError:
            signal = 0
        seen[ssid] = {
            "ssid": ssid,
            "signal": signal,
            "secure": bool(parts[2].strip()),
        }
    return sorted(seen.values(), key=lambda n: -n["signal"])


def saved_connections() -> list[str]:
    rc, out, _ = run(["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show"])
    return [
        ln.split(":")[0]
        for ln in out.splitlines()
        if ":" in ln and ln.split(":")[1].endswith("wireless")
    ]


# ── Access point ───────────────────────────────────────────────────────────────

def enable_captive_dns() -> None:
    """
    Point every DNS lookup at the Pi so phones pop the captive-portal sheet.

    NetworkManager runs its own dnsmasq for shared connections and reads
    extra config from this directory, so we do not have to manage dnsmasq
    ourselves.
    """
    d = "/etc/NetworkManager/dnsmasq-shared.d"
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "roadsentinel-captive.conf"), "w") as fh:
            fh.write(f"address=/#/{AP_ADDRESS}\n")
    except OSError as exc:
        log.warning("Could not write captive DNS config: %s", exc)


def start_ap(dev: str) -> bool:
    log.info("Starting access point '%s' on %s", AP_SSID, dev)
    enable_captive_dns()
    run(["nmcli", "connection", "delete", AP_CONN_NAME], timeout=15)

    rc, _, err = run([
        "nmcli", "connection", "add",
        "type", "wifi", "ifname", dev, "con-name", AP_CONN_NAME,
        "autoconnect", "no", "ssid", AP_SSID,
        "802-11-wireless.mode", "ap",
        "802-11-wireless.band", "bg",
        "ipv4.method", "shared",
        "wifi-sec.key-mgmt", "wpa-psk",
        "wifi-sec.psk", AP_PASSWORD,
    ], timeout=30)
    if rc != 0:
        log.error("Failed to create AP profile: %s", err)
        return False

    rc, _, err = run(["nmcli", "connection", "up", AP_CONN_NAME], timeout=45)
    if rc != 0:
        log.error("Failed to bring AP up: %s", err)
        return False

    log.info("AP up — SSID '%s', password '%s', portal http://%s",
             AP_SSID, AP_PASSWORD, AP_ADDRESS)
    return True


def stop_ap() -> None:
    run(["nmcli", "connection", "down", AP_CONN_NAME], timeout=20)
    run(["nmcli", "connection", "delete", AP_CONN_NAME], timeout=20)
    try:
        os.remove("/etc/NetworkManager/dnsmasq-shared.d/roadsentinel-captive.conf")
    except OSError:
        pass


def join_network(dev: str, ssid: str, password: str) -> tuple[bool, str]:
    """Leave AP mode and try to join the given network."""
    stop_ap()
    time.sleep(3)

    cmd = ["nmcli", "device", "wifi", "connect", ssid, "ifname", dev]
    if password:
        cmd += ["password", password]
    rc, out, err = run(cmd, timeout=75)
    if rc == 0:
        return True, out or "connected"

    msg = err or out or "unknown error"
    log.error("Join '%s' failed: %s", ssid, msg)
    return False, msg


# ── Portal web app ─────────────────────────────────────────────────────────────

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Road Sentinel — WiFi Setup</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:20px; background:#0B0E14; color:#E8EAED;
         font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .card {{ max-width:460px; margin:0 auto; background:#141922;
           border:1px solid #232A38; border-radius:14px; padding:22px; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .sub {{ color:#8A93A6; font-size:13px; margin-bottom:20px; }}
  label {{ display:block; font-size:13px; color:#8A93A6; margin:14px 0 6px; }}
  select,input {{ width:100%; padding:12px; font-size:16px; border-radius:9px;
                  border:1px solid #232A38; background:#0B0E14; color:#E8EAED; }}
  button {{ width:100%; margin-top:20px; padding:14px; font-size:16px; font-weight:600;
            border:0; border-radius:9px; background:#F2B33D; color:#12151C; }}
  .msg {{ padding:12px; border-radius:9px; margin-bottom:16px; font-size:14px; }}
  .err {{ background:rgba(229,72,77,.15); border:1px solid rgba(229,72,77,.4); color:#E5484D; }}
  .ok  {{ background:rgba(61,220,151,.15); border:1px solid rgba(61,220,151,.4); color:#3DDC97; }}
  .sig {{ color:#8A93A6; font-size:12px; }}
  .foot {{ margin-top:18px; color:#8A93A6; font-size:12px; text-align:center; }}
</style>
<div class="card">
  <h1>Road Sentinel</h1>
  <div class="sub">{host} &middot; WiFi setup</div>
  {message}
  <form method="POST" action="/connect">
    <label for="ssid">Network</label>
    <select id="ssid" name="ssid">{options}</select>
    <label for="password">Password</label>
    <input id="password" name="password" type="password"
           placeholder="Leave blank if open" autocomplete="off">
    <button type="submit">Connect</button>
  </form>
  <div class="foot">Saved networks: {saved}</div>
</div>
"""

# The probe URLs each mobile OS hits to decide whether it is behind a captive
# portal. Answering them with a redirect is what makes the sign-in sheet
# appear on its own instead of the user having to find the IP.
PROBE_PATHS = (
    "/generate_204", "/gen_204",                       # Android
    "/hotspot-detect.html", "/library/test/success.html",  # iOS / macOS
    "/connecttest.txt", "/ncsi.txt", "/redirect",      # Windows
    "/canonical.html", "/success.txt",                 # Firefox / others
)


class PortalState:
    def __init__(self, dev: str, led: StatusLed) -> None:
        self.dev = dev
        self.led = led
        self.networks: list[dict] = []
        self.message = ""
        self.message_kind = ""
        self.connected = threading.Event()
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    state: PortalState = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):  # quieter journal
        log.debug("http: " + fmt, *args)

    def _send(self, code: int, body: bytes, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _page(self) -> bytes:
        st = self.state
        opts = []
        for n in st.networks:
            lock = " \U0001F512" if n["secure"] else ""
            opts.append(
                '<option value="{v}">{t}{lock} &nbsp;({sig}%)</option>'.format(
                    v=html.escape(n["ssid"], quote=True),
                    t=html.escape(n["ssid"]),
                    lock=lock,
                    sig=n["signal"],
                )
            )
        if not opts:
            opts.append("<option>(no networks found — rescanning)</option>")

        msg = ""
        if st.message:
            cls = "ok" if st.message_kind == "ok" else "err"
            msg = f'<div class="msg {cls}">{html.escape(st.message)}</div>'

        saved = ", ".join(saved_connections()[:4]) or "none"
        return PAGE.format(
            host=html.escape(socket.gethostname()),
            options="".join(opts),
            message=msg,
            saved=html.escape(saved),
        ).encode()

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in PROBE_PATHS:
            # 302 to ourselves -> the OS decides it is behind a portal and
            # opens the sheet. Answering 204 here would tell it the internet
            # is fine and no portal would ever appear.
            self._send(302, b"", extra={"Location": f"http://{AP_ADDRESS}/"})
            return
        if path == "/status":
            self._send(200, json.dumps({
                "hostname": socket.gethostname(),
                "networks": self.state.networks,
                "saved": saved_connections(),
            }).encode(), "application/json")
            return
        self._send(200, self._page())

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/connect":
            self._send(404, b"not found")
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        ssid = (form.get("ssid", [""])[0]).strip()
        password = form.get("password", [""])[0]

        if not ssid:
            self.state.message = "Pick a network first."
            self.state.message_kind = "err"
            self._send(200, self._page())
            return

        # Respond BEFORE switching networks: joining tears down the AP, which
        # drops this very connection. If we tried to reply afterwards the
        # phone would just see a dead socket and the user would not know
        # whether it worked.
        body = ("<!doctype html><meta charset=utf-8>"
                "<meta name=viewport content='width=device-width,initial-scale=1'>"
                "<body style='background:#0B0E14;color:#E8EAED;font-family:sans-serif;"
                "padding:40px;text-align:center'>"
                f"<h2 style='color:#F2B33D'>Connecting to {html.escape(ssid)}…</h2>"
                "<p>This network will disappear now — that is expected.</p>"
                "<p>Watch the Pi's LED:<br><b>slow blink</b> = connecting, "
                "<b>brief blink every 5s</b> = connected, "
                "<b>fast blink</b> = failed, portal is back.</p>"
                "</body>").encode()
        self._send(200, body)
        try:
            self.wfile.flush()
        except OSError:
            pass

        threading.Thread(
            target=self._do_join, args=(ssid, password), daemon=True
        ).start()

    def _do_join(self, ssid: str, password: str) -> None:
        st = self.state
        time.sleep(1.5)  # let the response actually reach the phone
        st.led.set("connecting")
        ok, detail = join_network(st.dev, ssid, password)
        if ok and wifi_is_up(st.dev):
            log.info("Joined '%s'", ssid)
            st.led.set("connected")
            st.connected.set()
        else:
            log.error("Could not join '%s': %s", ssid, detail)
            st.message = f"Could not connect to {ssid}: {detail}"
            st.message_kind = "err"
            st.led.set("portal")
            start_ap(st.dev)      # bring the portal back so they can retry
            st.networks = scan_networks(st.dev)


def run_portal(dev: str, led: StatusLed) -> bool:
    """Hold the AP + portal open until configured or PORTAL_TIMEOUT. True if joined."""
    st = PortalState(dev, led)
    st.networks = scan_networks(dev)

    if not start_ap(dev):
        led.set("failed")
        return False
    led.set("portal")

    Handler.state = st
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", PORTAL_PORT), Handler)
    except OSError as exc:
        log.error("Cannot bind port %d: %s", PORTAL_PORT, exc)
        stop_ap()
        return False

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log.info("Portal ready — join '%s' (password: %s), browse http://%s",
             AP_SSID, AP_PASSWORD, AP_ADDRESS)

    joined = st.connected.wait(timeout=PORTAL_TIMEOUT)
    httpd.shutdown()

    if not joined:
        log.warning("Portal timed out after %ds — retrying saved networks",
                    PORTAL_TIMEOUT)
        stop_ap()
    return joined


# ── Watcher ────────────────────────────────────────────────────────────────────

def watch(led: StatusLed) -> None:
    dev = wifi_device()
    log.info("Watching %s — portal after %d consecutive failures",
             dev, FAIL_THRESHOLD)
    failures = 0

    while True:
        if wifi_is_up(dev):
            if failures:
                log.info("WiFi recovered after %d failure(s)", failures)
            failures = 0
            led.set("connected")
            time.sleep(CHECK_INTERVAL)
            continue

        failures += 1
        led.set("connecting" if failures < FAIL_THRESHOLD else "failed")
        log.warning("WiFi down (%d/%d)", failures, FAIL_THRESHOLD)

        # Give NetworkManager a nudge before escalating — most outages are
        # transient and reconnect on their own.
        if failures < FAIL_THRESHOLD:
            run(["nmcli", "device", "connect", dev], timeout=45)
            time.sleep(CHECK_INTERVAL)
            continue

        log.error("WiFi failed %d times — raising setup portal", failures)
        if run_portal(dev, led):
            failures = 0
            led.set("connected")
        else:
            # Portal timed out. Fall back to normal retries rather than
            # holding the AP open forever — the real network may well come
            # back on its own, and an AP nobody connects to helps nobody.
            failures = 0
            run(["nmcli", "device", "connect", dev], timeout=45)
        time.sleep(CHECK_INTERVAL)


def main() -> int:
    ap = argparse.ArgumentParser(description="Road Sentinel WiFi provisioning portal")
    ap.add_argument("--portal-now", action="store_true",
                    help="Raise the portal immediately, skipping the failure count")
    ap.add_argument("--status", action="store_true",
                    help="Print WiFi status and exit")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [wifi-portal] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dev = wifi_device()

    if args.status:
        print(f"device:    {dev}")
        print(f"connected: {wifi_is_up(dev)}")
        print(f"saved:     {', '.join(saved_connections()) or 'none'}")
        for n in scan_networks(dev)[:10]:
            print(f"  {n['signal']:3d}%  {n['ssid']}{' (secure)' if n['secure'] else ''}")
        return 0

    if os.geteuid() != 0:
        print("Must run as root (nmcli + LED sysfs).", file=sys.stderr)
        return 1

    led = StatusLed()
    try:
        if args.portal_now:
            return 0 if run_portal(dev, led) else 1
        watch(led)
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        led.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
