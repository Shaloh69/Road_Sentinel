# WiFi Provisioning Portal

Re-provision a headless Pi's WiFi from a phone. No monitor, no keyboard, no
SD card removal.

The Pis sit at the Busay installation site. If the WiFi changes — new router,
new password, network renamed — the Pi drops off and there is no way to fix
it without physically carrying a screen and keyboard to it. This turns any
phone into that screen.

## How it works

1. A background service checks the WiFi every 20s. While it is healthy it
   does nothing at all.
2. After **5 consecutive failures** it stops retrying and raises an access
   point.
3. Connect a phone to it:

   | | |
   |---|---|
   | SSID | `RoadSentinel-Setup` |
   | Password | `roadsentinel` |
   | Portal | `http://10.42.0.1` |

4. A captive-portal sheet opens by itself — the same "Sign in to network"
   prompt hotel WiFi shows. If it does not, browse to `http://10.42.0.1`.
5. Pick a network from the scanned list, enter the password, submit.
6. The Pi tears down the AP and joins. If it fails, the portal comes back so
   you can retry.

## LED status

The onboard ACT LED reports state, so the Pi is diagnosable at a glance even
with no network at all:

| Pattern | Meaning |
|---|---|
| Brief blink every 5s | Connected — healthy |
| Even 1 Hz blink | Connecting / retrying |
| **Fast 4 Hz blink** | **Portal is up, waiting for you** |
| Double-blink, pause | Connection failed |

The service takes the LED off its default `mmc0` trigger while running and
restores it on exit, so normal disk-activity indication comes back.

## Install

```bash
cd ~/roadsentinel-repo/raspi_scripts
bash setup_wifi_portal.sh
```

Requires NetworkManager, standard on Raspberry Pi OS Bookworm.

## Commands

```bash
sudo systemctl status roadsentinel-wifi-portal
sudo journalctl -u roadsentinel-wifi-portal -f

sudo python3 /opt/roadsentinel/wifi_portal.py --status       # what it sees
sudo python3 /opt/roadsentinel/wifi_portal.py --portal-now   # force portal up
```

`--portal-now` is the one to use for testing — it skips the failure count and
raises the AP immediately.

## Tuning

Edit the `Environment=` lines in
`/etc/systemd/system/roadsentinel-wifi-portal.service`, then:

```bash
sudo systemctl daemon-reload && sudo systemctl restart roadsentinel-wifi-portal
```

| Variable | Default | Meaning |
|---|---|---|
| `PORTAL_FAIL_THRESHOLD` | 5 | Consecutive failures before the portal opens |
| `PORTAL_CHECK_INTERVAL` | 20 | Seconds between checks |
| `PORTAL_TIMEOUT` | 900 | Seconds to hold the portal before resuming retries |
| `PORTAL_AP_SSID` | RoadSentinel-Setup | AP name |
| `PORTAL_AP_PASSWORD` | roadsentinel | AP password (8+ chars, WPA2 minimum) |

## Design notes

**Why not comitup or RaspiWiFi.** Both are more general and well-established,
but both take over WiFi management wholesale, which would fight with the
Tailscale and systemd setup already on these Pis. This stays completely out
of the way while the network is healthy and only intervenes after sustained,
repeated failure. It also adds the two things neither offers: a failure-count
trigger rather than boot-time-only setup, and LED status signalling.

**Why the probe URLs return 302 and not 204.** Each mobile OS hits a known
URL to decide whether it is behind a captive portal — Android
`/generate_204`, iOS `/hotspot-detect.html`, Windows `/ncsi.txt`. Answering
204 tells the phone the internet is fine and **no portal sheet ever appears**;
the user would have to know to type the IP. Answering 302 is what makes it
open on its own. There is a test covering this.

**Why DNS is hijacked.** `/etc/NetworkManager/dnsmasq-shared.d/` resolves
every lookup to the Pi. NetworkManager runs its own dnsmasq for shared
connections and reads that directory, so dnsmasq does not have to be managed
separately.

**Why the HTTP response is sent before switching networks.** Joining the new
network tears down the AP, which drops the phone's connection. Replying
afterwards would leave the user staring at a dead socket with no idea whether
it worked. The page is sent first, then the join runs on a background thread —
which is why the confirmation page explains what the LED will do next.

**Why a timeout rather than holding the AP open forever.** If nobody connects
within 15 minutes, the service goes back to ordinary retries. The real
network may well return on its own, and an access point nobody is connecting
to helps no one while actively preventing reconnection.

**Failure-count trigger, not boot-time.** Provisioning at boot only would not
help here — the common case is a Pi that has been running fine for weeks and
then loses its network. That is exactly when nobody is standing next to it.

## Recommended: add a hotspot fallback

Worth doing while the Pi still has a connection. NetworkManager saves
multiple networks and picks whichever is in range:

```bash
sudo nmcli device wifi connect "YourPhoneHotspot" password "yourpassword"
sudo nmcli connection modify "YourPhoneHotspot" connection.autoconnect-priority 5
```

Then any future WiFi problem is recoverable by just turning on your phone's
hotspot — no portal needed.

## Security

The AP is WPA2-protected, not open. Anyone joining it can set the Pi's WiFi
credentials, so treat the AP password as a real credential and change it from
the default before deploying somewhere public:

```
Environment=PORTAL_AP_PASSWORD=<something-better>
```

The portal binds port 80 only while the AP is up and is torn down immediately
after a successful join — it is not listening during normal operation.
