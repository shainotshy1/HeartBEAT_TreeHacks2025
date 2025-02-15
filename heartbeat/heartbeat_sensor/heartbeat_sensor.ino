const int PULSE_SENSOR_PIN = A0;  // Analog pin connected to pulse sensor
const int LED_PIN = 13;         // On-board LED for visual feedback
const int SAMPLING_RATE = 100;  // Sample every 10ms (100Hz)

volatile int BPM;                // Holds heart rate value
volatile int Signal;             // Holds the incoming raw data
volatile int IBI = 600;          // Holds the time between beats (Inter-Beat Interval)
volatile boolean Pulse = false;  // "True" when heartbeat is detected
volatile boolean QS = false;     // Becomes true when Arduino finds a beat

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);  // High baud rate for faster data transmission
  
  // Configure Timer2 interrupt for precise sampling
  cli();                // Disable interrupts
  TCCR2A = 0x02;       // DISABLE PWM ON DIGITAL PINS 3 AND 11, AND GO INTO CTC MODE
  TCCR2B = 0x06;       // DON'T FORCE COMPARE, 256 PRESCALER
  OCR2A = 0X7C;        // SET THE TOP OF THE COUNT TO 124 FOR 500Hz SAMPLE RATE
  TIMSK2 = 0x02;       // ENABLE INTERRUPT ON MATCH BETWEEN TIMER2 AND OCR2A
  sei();               // Enable interrupts
}

void loop() {
  if (QS) {  // Quantified Self flag is true when arduino finds a heartbeat
    Serial.print("BPM:");
    Serial.println(BPM);
    Serial.print("Signal:");
    Serial.println(Signal);
    QS = false;  // Reset the Quantified Self flag for next time
  }
  delay(20);  // Short delay to prevent flooding the serial port
}

// Timer2 interrupt service routine
ISR(TIMER2_COMPA_vect) {
  cli();                // Disable interrupts while we do this
  Signal = analogRead(PULSE_SENSOR_PIN);  // Read the Pulse Sensor
  sei();                // Re-enable interrupts
  
  // Simple peak detection algorithm
  static int N = 0;
  static int lastBeatTime = 0;
  static int thresh = 512;
  static int P = 512;
  static int T = 512;
  static int firstBeat = true;
  static int secondBeat = false;
  
  unsigned long now = millis();
  
  if (Signal < thresh && N > (IBI/5)*3) {
    if (Signal < T) {
      T = Signal;
    }
  }
  
  if (Signal > thresh && Signal > P) {
    P = Signal;
  }
  
  N++;
  
  if (N > 250) {
    thresh = (P + T) / 2;
    P = thresh;
    T = thresh;
    N = 0;
  }
  
  if (Signal > thresh && Pulse == false && N > (IBI/5)*3) {
    Pulse = true;
    digitalWrite(LED_PIN, HIGH);
    IBI = now - lastBeatTime;
    lastBeatTime = now;
    
    if (secondBeat) {
      secondBeat = false;
      for (int i = 0; i <= 9; i++) {
        int rate = 60000 / IBI;
        BPM = rate;
      }
    }
    
    if (firstBeat) {
      firstBeat = false;
      secondBeat = true;
    }
  }
  
  if (Signal < thresh && Pulse == true) {
    digitalWrite(LED_PIN, LOW);
    Pulse = false;
    thresh = (P + T) / 2;
    P = thresh;
    T = thresh;
  }
}
