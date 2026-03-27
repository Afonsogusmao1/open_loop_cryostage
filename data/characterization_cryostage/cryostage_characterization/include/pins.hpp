#pragma once

// SPI (VSPI default no ESP32)
static constexpr int PIN_SCK = 18;
static constexpr int PIN_MISO = 19;
static constexpr int PIN_MOSI = 23;

// Chip Select do MAX31865
static constexpr int PIN_CS_RTD = 27;

// Chip Selects dos MAX31856 (termopares)
// Escolha pinos livres (evitar 25/26/32/33 e 16/17/21/22 usados pelos BTS).
static constexpr int PIN_CS_TC3 = 13;  // z=3 mm
static constexpr int PIN_CS_TC7 = 14;  // z=7 mm
static constexpr int PIN_CS_TC12 = 4;  // z=12 mm
static constexpr int PIN_CS_TAMB = 15; // ambient thermocouple (Tamb)

// RTD constants
static constexpr float RNOMINAL = 100.0f; // PT100
static constexpr float RREF = 431.0f;     // resistor de referência do teu módulo
