#include "HX711.h"

#define DOUT 8  
#define CLK 7

HX711 scale;

// Your calibration factor (adjust if needed)
const float force_a = 0.00001652;  // Newtons per raw unit

long raw_zero = 0;
unsigned long last_sample_time = 0;
const unsigned long sample_interval = 12;  // ≈12.5 ms → 80 Hz

void setup() {
  Serial.begin(38400);         
  scale.begin(DOUT, CLK);
  delay(1000);                 // Wait for HX711 to stabilize
  scale.tare();                // Zero the scale
  raw_zero = scale.read();     // Store baseline
  Serial.println("time_ms,force_N");  // CSV header
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_time >= sample_interval) {
    last_sample_time = now;

    long raw = scale.read();
    long net = raw - raw_zero;

    float force = force_a * net;
    if (abs(force) < 0.01) force = 0.0;  // Noise filter

    Serial.print(now);
    Serial.print(",");
    Serial.println(force, 4);  // 4 decimal places
  }
}
