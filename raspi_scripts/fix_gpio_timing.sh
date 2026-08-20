#!/usr/bin/env bash
# Road Sentinel — Pi 4 LED GPIO-timing conflict checker/fixer.
#
# Context: Pi 4's HUB75 display intermittently shows garbled/scrambled text
# (not consistently blank — sometimes fine, sometimes garbage). That pattern
# is the signature of a GPIO signal-timing problem, not a static config error:
# Pi 4's CPU clocks GPIO writes faster than older Pis, and if the panel's
# shift registers can't latch data reliably, output corrupts intermittently,
# worse under load. display_manager.py's --led-slowdown-gpio default was
# raised from 4 to 6 in Phase 0 (raspi_scripts/display_manager.py) to help
# with this, but slowdown alone doesn't fix the other common causes below.
#
# This script only CHECKS and reports by default. Pass --fix to actually
# apply the two safe, reversible OS-level fixes (sound module blacklist,
# cmdline.txt isolcpus). Run this ON the Raspberry Pi 4 itself (via SSH/
# Tailscale/roadsentinel-agent), not from a dev machine.
#
# Usage:
#   bash fix_gpio_timing.sh          # report only
#   bash fix_gpio_timing.sh --fix    # apply blacklist + isolcpus, then prompt reboot

set -uo pipefail

FIX=false
[ "${1:-}" = "--fix" ] && FIX=true

echo "================================================================"
echo " Road Sentinel — Pi 4 GPIO Timing Diagnostic"
echo "================================================================"
echo

# ── 1. Onboard sound module (snd_bcm2835) ─────────────────────────────────────
# Shares hardware with the HUB75 driver's PWM/DMA and must be blacklisted.
echo "[1/4] Onboard sound module (snd_bcm2835)..."
if lsmod 2>/dev/null | grep -q snd_bcm2835; then
    echo "  ✗ snd_bcm2835 is currently LOADED — this conflicts with the LED matrix driver"
    if $FIX; then
        BLACKLIST_FILE="/etc/modprobe.d/blacklist-rgb-matrix.conf"
        if [ ! -f "$BLACKLIST_FILE" ]; then
            echo "blacklist snd_bcm2835" | sudo tee "$BLACKLIST_FILE" > /dev/null
            echo "  → Blacklisted in $BLACKLIST_FILE (takes effect after reboot)"
        else
            echo "  → Already blacklisted in $BLACKLIST_FILE (takes effect after reboot)"
        fi
    else
        echo "  → Re-run with --fix to blacklist it (requires reboot to take effect)"
    fi
else
    echo "  ✓ snd_bcm2835 not loaded"
fi
echo

# ── 2. 1-Wire overlay (dtoverlay=w1-gpio) ─────────────────────────────────────
# Often added for temperature sensors; can conflict with the same GPIO pins.
echo "[2/4] 1-Wire overlay (dtoverlay=w1-gpio)..."
CONFIG_TXT=""
for candidate in /boot/firmware/config.txt /boot/config.txt; do
    [ -f "$candidate" ] && CONFIG_TXT="$candidate" && break
done
if [ -n "$CONFIG_TXT" ] && grep -q "^dtoverlay=w1-gpio" "$CONFIG_TXT" 2>/dev/null; then
    echo "  ✗ 1-Wire overlay is ACTIVE in $CONFIG_TXT — may conflict with LED matrix GPIO pins"
    echo "  → Not auto-removed (you may need it for a sensor) — comment it out manually"
    echo "    if you don't need 1-Wire, then reboot: sudo sed -i 's/^dtoverlay=w1-gpio/#dtoverlay=w1-gpio/' $CONFIG_TXT"
else
    echo "  ✓ No 1-Wire overlay found${CONFIG_TXT:+ in $CONFIG_TXT}"
fi
echo

# ── 3. CPU core isolation (isolcpus) ──────────────────────────────────────────
# Keeps one core free of the scheduler for consistent display-refresh timing.
echo "[3/4] CPU core isolation (isolcpus)..."
CMDLINE_TXT=""
for candidate in /boot/firmware/cmdline.txt /boot/cmdline.txt; do
    [ -f "$candidate" ] && CMDLINE_TXT="$candidate" && break
done
if [ -n "$CMDLINE_TXT" ] && grep -q "isolcpus=" "$CMDLINE_TXT" 2>/dev/null; then
    echo "  ✓ isolcpus already set in $CMDLINE_TXT: $(grep -o 'isolcpus=[0-9,]*' "$CMDLINE_TXT")"
elif [ -n "$CMDLINE_TXT" ]; then
    echo "  – isolcpus not set in $CMDLINE_TXT"
    if $FIX; then
        sudo cp "$CMDLINE_TXT" "${CMDLINE_TXT}.bak-$(date +%s)"
        sudo sed -i 's/$/ isolcpus=3/' "$CMDLINE_TXT"
        echo "  → Added isolcpus=3 to $CMDLINE_TXT (backup saved, takes effect after reboot)"
        echo "    Only helps if display_manager.py is later pinned to core 3 (not done automatically)"
    else
        echo "  → Re-run with --fix to reserve core 3 for display timing (requires reboot)"
    fi
else
    echo "  ? Could not find cmdline.txt — skipping"
fi
echo

# ── 4. Panel input logic chips ────────────────────────────────────────────────
# 74HCT245/74AHCT245 are 3.3V-compatible; 74HC245 is NOT and causes exactly
# this kind of intermittent corruption. This can only be checked by reading
# the chip markings on the physical HUB75 adapter board — not detectable
# from software.
echo "[4/4] Panel input logic chips (manual check — cannot be detected from software)"
echo "  Look at the small ICs on the HUB75 adapter/breakout board:"
echo "    74HCT245 / 74AHCT245  → OK, 3.3V-compatible with the Pi's GPIO"
echo "    74HC245               → NOT compatible — known cause of intermittent"
echo "                            corruption on Pi 4/5; the ₱149 adapter boards"
echo "                            referenced in raspi_scripts/hub75_piomatter_notes.md"
echo "                            should already use the HCT/AHCT variant, but"
echo "                            verify against the actual board in hand."
echo

echo "================================================================"
if $FIX; then
    echo " Fixes applied where possible. Reboot required for changes to take effect:"
    echo "   sudo reboot"
else
    echo " Report only. Re-run with --fix to apply the safe, reversible fixes above."
fi
echo " After reboot, re-test with:"
echo "   sudo python3 raspi_scripts/display_manager.py --pi 4 --test"
echo "================================================================"
