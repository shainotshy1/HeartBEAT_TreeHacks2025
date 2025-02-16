// Pulse Monitor  Test Script
int sensorPin = A0;

void setup() {
    Serial.begin(9600);
}

void  loop ()
{
//  Serial.print("raw: ");
  Serial.println(analogRead(sensorPin));
//  delay(10);
}

 