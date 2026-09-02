/*
 * ESP32 port of the HUB75 driver — Road Sentinel Pi 4 sign.
 *
 * The only board-specific code in the project. Everything above it (panel
 * mapping, framebuffer, text, modes, protocol) is shared with the STM32 port.
 *
 * Measured on hardware: 250 frames per second, reported by INFO.
 */

#include "hub75.h"
#include "soc/gpio_struct.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace hub75 {

// ── Pin bit masks ──────────────────────────────────────────────────────────
// Every pin is below GPIO 32, so the single 32-bit W1TS/W1TC registers cover
// all of them and one store moves six data lines at once. That is the whole
// reason a bit-banged refresh is fast enough here.
static const uint32_t M_R1  = 1UL << PIN_R1;
static const uint32_t M_G1  = 1UL << PIN_G1;
static const uint32_t M_B1  = 1UL << PIN_B1;
static const uint32_t M_R2  = 1UL << PIN_R2;
static const uint32_t M_G2  = 1UL << PIN_G2;
static const uint32_t M_B2  = 1UL << PIN_B2;
static const uint32_t M_A   = 1UL << PIN_A;
static const uint32_t M_B   = 1UL << PIN_B;
static const uint32_t M_C   = 1UL << PIN_C;
static const uint32_t M_CLK = 1UL << PIN_CLK;
static const uint32_t M_LAT = 1UL << PIN_LAT;
static const uint32_t M_OE  = 1UL << PIN_OE;

static const uint32_t M_DATA = M_R1|M_G1|M_B1|M_R2|M_G2|M_B2;
static const uint32_t M_ADDR = M_A|M_B|M_C;

#define GPIO_SET(m) GPIO.out_w1ts = (m)
#define GPIO_CLR(m) GPIO.out_w1tc = (m)

// Every (channel1, channel2) colour pair precomputed into the GPIO word that
// drives the six data lines. Both channels are 3 bits, so there are only 64
// combinations and the whole table is 256 bytes.
//
// This replaces six conditional branches per pixel with one shift, one OR and
// one load. At ~2000 pixels per frame that removes ~12000 branches per frame,
// and branches are exactly what a tight bit-banged loop cannot afford.
static uint32_t dataBits[64];

static void buildDataTable() {
  for (int c1 = 0; c1 < 8; c1++) {
    for (int c2 = 0; c2 < 8; c2++) {
      uint32_t b = 0;
      if (c1 & 0x4) b |= M_R1;
      if (c1 & 0x2) b |= M_G1;
      if (c1 & 0x1) b |= M_B1;
      if (c2 & 0x4) b |= M_R2;
      if (c2 & 0x2) b |= M_G2;
      if (c2 & 0x1) b |= M_B2;
      dataBits[(c1 << 3) | c2] = b;
    }
  }
}

static volatile uint8_t  bright   = DEFAULT_BRIGHTNESS;
static volatile uint32_t onTimeUs = 200;
static volatile uint32_t fps      = 0;

void setBrightness(uint8_t b) {
  bright = b;
  // Maps 0-255 onto the lit window per address. The 8us floor keeps a very
  // low setting dim but still visible rather than snapping to black.
  onTimeUs = 8 + ((uint32_t)b * 400) / 255;
}
uint8_t  brightness()      { return bright; }
uint32_t framesPerSecond() { return fps; }

// ── FM6124 register init ───────────────────────────────────────────────────
// Copied from the reference implementation rather than invented — it is the
// one piece of the old stack that demonstrably worked, and getting it wrong
// looks exactly like a dead panel, which would send diagnosis the wrong way.
//
// REG1 sets global drive current, REG2 holds the single bit that enables
// output at all. The latch must rise a specific number of clocks before the
// row ends; that count is how the chip knows this is a register write and not
// pixel data.
static void fm6124init() {
  const bool REG1[16] = {0,0,0,0,0, 1,1,1,1,1,1, 0,0,0,0,0};
  const bool REG2[16] = {0,0,0,0,0, 0,0,0,0,1,0, 0,0,0,0,0};

  GPIO_SET(M_OE);
  GPIO_CLR(M_LAT);

  for (int l = 0; l < REGISTER_LEN; l++) {
    if (REG1[l % 16]) GPIO_SET(M_DATA); else GPIO_CLR(M_DATA);
    if (l > REGISTER_LEN - 12) GPIO_SET(M_LAT);
    GPIO_SET(M_CLK); GPIO_CLR(M_CLK);
  }
  GPIO_CLR(M_LAT);

  for (int l = 0; l < REGISTER_LEN; l++) {
    if (REG2[l % 16]) GPIO_SET(M_DATA); else GPIO_CLR(M_DATA);
    if (l > REGISTER_LEN - 13) GPIO_SET(M_LAT);
    GPIO_SET(M_CLK); GPIO_CLR(M_CLK);
  }
  GPIO_CLR(M_LAT);

  GPIO_CLR(M_DATA);
  for (int l = 0; l < REGISTER_LEN; l++) { GPIO_SET(M_CLK); GPIO_CLR(M_CLK); }
  GPIO_SET(M_LAT);
  GPIO_CLR(M_LAT);
}

// ── Refresh ────────────────────────────────────────────────────────────────
static void refreshTask(void *) {
  uint32_t frames = 0;
  uint32_t tMark  = millis();

  for (;;) {
    for (uint8_t a = 0; a < SCAN_ADDRESSES; a++) {
      const uint16_t *m1 = panelmap::ch1[a];
      const uint16_t *m2 = panelmap::ch2[a];

      // Both channels load in the same pass because they share the clock.
      for (int p = 0; p < REGISTER_LEN; p++) {
        uint32_t bits = dataBits[(fbuf::pixels[m1[p]] << 3) | fbuf::pixels[m2[p]]];
        GPIO_CLR(M_DATA);
        if (bits) GPIO_SET(bits);
        GPIO_SET(M_CLK);
        GPIO_CLR(M_CLK);
      }

      // Blank BEFORE switching rows. Without this the previous row stays lit
      // while the address changes, which shows as ghosting one row down.
      GPIO_SET(M_OE);

      GPIO_CLR(M_ADDR);
      uint32_t addrBits = 0;
      if (a & 0x1) addrBits |= M_A;
      if (a & 0x2) addrBits |= M_B;
      if (a & 0x4) addrBits |= M_C;
      if (addrBits) GPIO_SET(addrBits);

      GPIO_SET(M_LAT);
      GPIO_CLR(M_LAT);

      GPIO_CLR(M_OE);                     // OE is active LOW: low = lit
      delayMicroseconds(onTimeUs);
      GPIO_SET(M_OE);
    }

    frames++;
    uint32_t now = millis();
    if (now - tMark >= 1000) { fps = frames; frames = 0; tMark = now; }

    // Yield once per frame so the idle task runs and the watchdog stays fed.
    // The tick costs refresh duty, which is why the per-address on-time above
    // is generous — brightness is recovered there rather than by spinning.
    vTaskDelay(1);
  }
}

void begin() {
  for (uint8_t p : {PIN_R1,PIN_G1,PIN_B1,PIN_R2,PIN_G2,PIN_B2,
                    PIN_A,PIN_B,PIN_C,PIN_CLK,PIN_LAT,PIN_OE}) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
  GPIO_SET(M_OE);            // start blanked

  fbuf::clear();
  panelmap::build();
  buildDataTable();
  setBrightness(bright);
  fm6124init();

  // Core 1. Core 0 carries the WiFi/BT stacks in the Arduino framework even
  // when unused, and their interrupts would show up as visible refresh jitter.
  xTaskCreatePinnedToCore(refreshTask, "hub75", 4096, nullptr, 5, nullptr, 1);
}

} // namespace hub75
