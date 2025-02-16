import serial 
import time 
# read .env file
import os
from dotenv import load_dotenv

load_dotenv()

PORT = os.getenv("PORT")
BAUDRATE = os.getenv("BAUDRATE")

arduino = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=1) 

def read_sensor(): 
    time.sleep(0.05) 
    data = arduino.readline() 
    data = data.decode('utf-8').strip()
    print(data)

while True:
    read_sensor()

