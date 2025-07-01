#include "HX711.h"

HX711 scale;

void setup() {
  Serial.begin(38400);
  scale.begin(8, 7);
  delay(1000);
  scale.tare(); // Zero the sensor with no weight
  Serial.println("Start placing weights...");
}

void loop() {
  Serial.println(scale.read());  // Raw output
  delay(500);
}
