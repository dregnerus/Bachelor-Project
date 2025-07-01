import serial
import time

# Set your COM port and baud rate (match Arduino)
ser = serial.Serial('COM10', 38400)  # Replace 'COM3' with your port
filename = "force_data.csv"

with open(filename, 'w') as f:
    print("Logging started. Press Ctrl+C to stop.\n")
    while True:
        try:
            line = ser.readline().decode().strip()
            print(line)
            f.write(line + '\n')
        except KeyboardInterrupt:
            print("\nLogging stopped.")
            break
