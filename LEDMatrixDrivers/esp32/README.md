# ESP32 sign driver — Road Sentinel Pi 4

The ESP32 port of the HUB75Sign driver. Drives two chained 64x32 P5 outdoor
panels as one 128x32 sign, taking state from the Raspberry Pi 4 over USB serial.

- **Wiring and first-run checks:** [WIRING.md](WIRING.md)
- **How the driver works, and why it is not a library:** [../README.md](../README.md)
- **What was already ruled out, with photos:** [DEBUG_LOG.md](DEBUG_LOG.md)

## Build and flash

The board is cabled to the Pi, so build there — no swapping cables between a
laptop and the installation site.

```bash
ssh roadsentinel@100.98.53.95
cd ~/LEDMatrixDrivers/esp32

~/.pio-venv/bin/pio run                                      # compile
~/.pio-venv/bin/pio run -t upload --upload-port /dev/ttyUSB0 # flash
~/.pio-venv/bin/pio device monitor -b 115200                 # watch
```

`./flash.sh` wraps this and stops the bridge service first.

**Close the monitor before starting the bridge.** Both use the same USB serial
port and two processes cannot hold it at once. The failure mode is a silent
board with no error message, which is a confusing thing to debug.

**Upload speed is pinned to 230400 on purpose.** 921600 fails reproducibly
while the panel is drawing a full-brightness frame — the current draw browns
the board out at the end of the write, reporting `Failed to leave compressed
flash mode`. Do not raise it.

## Board-specific code

One file: [`src/hub75_esp32.cpp`](src/hub75_esp32.cpp). It configures the pins,
runs the FM6124 init sequence and refreshes the panel from a task pinned to
core 1 — core 0 carries the WiFi/BT stacks even when unused, and their
interrupts would show as visible refresh jitter.

Measured refresh rate: **250 fps**, reported live by the `INFO` command.

Everything else lives in `../shared/HUB75Sign/` and is identical to the STM32
build.

## Status

Working and confirmed on the physical sign: `SAFE`, `VEHICLE INCOMING` /
`SLOW DOWN`, and `STOP` all render legibly, with the falling-sand test showing
individual addressable pixels.
