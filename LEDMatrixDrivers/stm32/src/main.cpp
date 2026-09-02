/*
 * Road Sentinel STM32 sign (Pi 5) entry point.
 *
 * Deliberately trivial: all behaviour lives in shared/HUB75Sign/sign_app.cpp,
 * which is identical for both boards. The only board-specific code in this
 * project is hub75_stm32.cpp — the driver itself.
 */

#include <Arduino.h>
#include "sign_app.h"

void setup() { signapp::begin(); }
void loop()  { signapp::tick();  }
