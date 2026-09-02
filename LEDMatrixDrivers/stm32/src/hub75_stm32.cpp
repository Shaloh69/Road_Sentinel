/*
 * STM32 port of the HUB75 driver — Road Sentinel Pi 5 sign.
 *
 * WeAct Black Pill V3.0, STM32F411CEU6 at 100MHz, Arduino framework.
 *
 * Structurally identical to the ESP32 port: same panel mapping, same
 * framebuffer, same FM6124 init, same refresh shape. Only three things differ,
 * and each is a real hardware difference rather than a stylistic one:
 *
 *   1. GPIO writes go through BSRR instead of W1TS/W1TC.
 *   2. There is no FreeRTOS and no second core, so refreshing runs from a
 *      hardware timer interrupt rather than a pinned task.
 *   3. USB CDC provides the serial link to the Pi, not a UART bridge chip.
 *
 * ── Why a timer interrupt rather than a loop ───────────────────────────────
 *
 * The ESP32 pins its refresh to core 1 and lets core 0 handle everything else.
 * The F411 is single-core, so a busy refresh loop would starve command
 * handling, and refreshing from loop() would stall the display whenever the
 * main code did anything slow. A timer ISR gives the display a guaranteed
 * slice regardless of what the foreground is doing, which is what keeps the
 * refresh rate — and therefore the brightness — steady.
 *
 * One address is shifted per interrupt rather than a whole frame. Shifting all
 * eight would hold the ISR for milliseconds and make USB CDC unreliable.
 */

#include "hub75.h"
#include <HardwareTimer.h>

namespace hub75 {

// ── Pin bit masks ──────────────────────────────────────────────────────────
// All six data lines and all three address lines are on GPIOB, so one BSRR
// write moves them together. Control lines are on GPIOA.
//
// STM32's BSRR is neater than the ESP32's pair of registers: the low 16 bits
// set, the high 16 bits reset, so a set and a clear can be one atomic store.
#define PB_BIT(p)  (1UL << ((p) & 0x0F))

static const uint32_t B_R1 = PB_BIT(PIN_R1);
static const uint32_t B_G1 = PB_BIT(PIN_G1);
static const uint32_t B_B1 = PB_BIT(PIN_B1);
static const uint32_t B_R2 = PB_BIT(PIN_R2);
static const uint32_t B_G2 = PB_BIT(PIN_G2);
static const uint32_t B_B2 = PB_BIT(PIN_B2);
static const uint32_t B_A  = PB_BIT(PIN_A);
static const uint32_t B_B  = PB_BIT(PIN_B);
static const uint32_t B_C  = PB_BIT(PIN_C);

static const uint32_t A_CLK = PB_BIT(PIN_CLK);
static const uint32_t A_LAT = PB_BIT(PIN_LAT);
static const uint32_t A_OE  = PB_BIT(PIN_OE);

static const uint32_t B_DATA = B_R1|B_G1|B_B1|B_R2|B_G2|B_B2;
static const uint32_t B_ADDR = B_A|B_B|B_C;

// BSRR: low half sets, high half resets. One store, no read-modify-write, and
// atomic against interrupts.
#define GPIOB_SET(m) (GPIOB->BSRR = (m))
#define GPIOB_CLR(m) (GPIOB->BSRR = ((m) << 16))
#define GPIOA_SET(m) (GPIOA->BSRR = (m))
#define GPIOA_CLR(m) (GPIOA->BSRR = ((m) << 16))

// Every (channel1, channel2) colour pair precomputed into a COMPLETE BSRR
// word - set bits in the low half, reset bits in the high half.
//
// This is where STM32 beats the ESP32. The ESP32 needs a clear-then-set pair
// of stores; here the whole thing is one 32-bit write of a value computed at
// startup. The inner loop becomes one load, one store, and the clock pulse.
static uint32_t bsrrData[64];

static void buildDataTable() {
  for (int c1 = 0; c1 < 8; c1++) {
    for (int c2 = 0; c2 < 8; c2++) {
      uint32_t b = 0;
      if (c1 & 0x4) b |= B_R1;
      if (c1 & 0x2) b |= B_G1;
      if (c1 & 0x1) b |= B_B1;
      if (c2 & 0x4) b |= B_R2;
      if (c2 & 0x2) b |= B_G2;
      if (c2 & 0x1) b |= B_B2;
      bsrrData[(c1 << 3) | c2] = b | ((B_DATA & ~b) << 16);
    }
  }
}

static volatile uint8_t  bright   = DEFAULT_BRIGHTNESS;
static volatile uint32_t onTicks  = 200;
static volatile uint32_t fps      = 0;
static volatile uint8_t  curAddr  = 0;

static HardwareTimer *refreshTimer = nullptr;

void setBrightness(uint8_t b) {
  bright = b;
  // The lit window per address, in microseconds. Same curve as the ESP32 port
  // so both signs look the same at the same setting — worth keeping identical,
  // since the two are mounted on the same road.
  onTicks = 8 + ((uint32_t)b * 400) / 255;
}
uint8_t  brightness()      { return bright; }
uint32_t framesPerSecond() { return fps; }

// ── FM6124 register init ───────────────────────────────────────────────────
// Byte-for-byte the same sequence as the ESP32 port; only the register writes
// differ. REG1 sets global drive current, REG2 holds the bit that enables
// output at all. The latch must rise a specific number of clocks before the
// row ends — that count is how the chip distinguishes a register write from
// pixel data.
static void fm6124init() {
  const bool REG1[16] = {0,0,0,0,0, 1,1,1,1,1,1, 0,0,0,0,0};
  const bool REG2[16] = {0,0,0,0,0, 0,0,0,0,1,0, 0,0,0,0,0};

  GPIOA_SET(A_OE);
  GPIOA_CLR(A_LAT);

  for (int l = 0; l < REGISTER_LEN; l++) {
    if (REG1[l % 16]) GPIOB_SET(B_DATA); else GPIOB_CLR(B_DATA);
    if (l > REGISTER_LEN - 12) GPIOA_SET(A_LAT);
    GPIOA_SET(A_CLK); GPIOA_CLR(A_CLK);
  }
  GPIOA_CLR(A_LAT);

  for (int l = 0; l < REGISTER_LEN; l++) {
    if (REG2[l % 16]) GPIOB_SET(B_DATA); else GPIOB_CLR(B_DATA);
    if (l > REGISTER_LEN - 13) GPIOA_SET(A_LAT);
    GPIOA_SET(A_CLK); GPIOA_CLR(A_CLK);
  }
  GPIOA_CLR(A_LAT);

  GPIOB_CLR(B_DATA);
  for (int l = 0; l < REGISTER_LEN; l++) { GPIOA_SET(A_CLK); GPIOA_CLR(A_CLK); }
  GPIOA_SET(A_LAT);
  GPIOA_CLR(A_LAT);
}

// ── Refresh ISR: one address per call ──────────────────────────────────────
static void refreshOneAddress() {
  static uint32_t frames = 0;
  static uint32_t tMark  = 0;

  const uint8_t a = curAddr;
  const uint16_t *m1 = panelmap::ch1[a];
  const uint16_t *m2 = panelmap::ch2[a];

  for (int p = 0; p < REGISTER_LEN; p++) {
    GPIOB->BSRR = bsrrData[(fbuf::pixels[m1[p]] << 3) | fbuf::pixels[m2[p]]];
    GPIOA_SET(A_CLK);
    GPIOA_CLR(A_CLK);
  }

  // Blank BEFORE switching rows, or the previous row stays lit through the
  // address change and ghosts one row down.
  GPIOA_SET(A_OE);

  uint32_t addrBits = 0;
  if (a & 0x1) addrBits |= B_A;
  if (a & 0x2) addrBits |= B_B;
  if (a & 0x4) addrBits |= B_C;
  GPIOB->BSRR = addrBits | ((B_ADDR & ~addrBits) << 16);

  GPIOA_SET(A_LAT);
  GPIOA_CLR(A_LAT);

  GPIOA_CLR(A_OE);                // active LOW: low = lit
  delayMicroseconds(onTicks);
  GPIOA_SET(A_OE);

  if (++curAddr >= SCAN_ADDRESSES) {
    curAddr = 0;
    frames++;
    uint32_t now = millis();
    if (now - tMark >= 1000) { fps = frames; frames = 0; tMark = now; }
  }
}

void begin() {
  for (uint32_t p : {PIN_R1,PIN_G1,PIN_B1,PIN_R2,PIN_G2,PIN_B2,
                     PIN_A,PIN_B,PIN_C,PIN_CLK,PIN_LAT,PIN_OE}) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
  GPIOA_SET(A_OE);           // start blanked

  fbuf::clear();
  panelmap::build();
  buildDataTable();
  setBrightness(bright);
  fm6124init();

  // TIM3 is a general-purpose timer with no role in the Arduino core's own
  // timekeeping, so claiming it does not disturb millis()/micros().
  //
  // 8 addresses at 1kHz gives 125 full frames per second. That is well clear
  // of visible flicker while leaving the vast majority of CPU time to the
  // foreground, which on this single-core part has to handle USB CDC.
  refreshTimer = new HardwareTimer(TIM3);
  refreshTimer->setOverflow(1000, HERTZ_FORMAT);
  refreshTimer->attachInterrupt(refreshOneAddress);
  refreshTimer->resume();
}

} // namespace hub75
