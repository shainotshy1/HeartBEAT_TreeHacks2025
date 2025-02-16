import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Set up the display window
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Real-Time Graph Plot")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Set up the graph parameters
max_points = 100  # Max number of points in the graph
data = []  # List to hold the incoming values
x_scale = width / max_points  # X axis scale to fit data
y_scale = height / 2  # Y axis scale (we'll assume values fit in the range -1 to 1 for simplicity)

# Main loop
running = True
clock = pygame.time.Clock()

from heartbeat.heartbeat_sensor.heartbeat_sensors import ArduinoHeartbeatSensor
hs = ArduinoHeartbeatSensor(serial_port='/dev/cu.usbmodem1401', baud_rate=9600)

while running:
    screen.fill(BLACK)  # Clear the screen

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Simulate incoming data (replace with actual sensor data)
    # incoming_value = random.uniform(-1, 1)  # Generate a random value for demonstration
    incoming_value = hs.read_single_signal()
    incoming_value = incoming_value / 1024  # Normalize the value to be in the range -1 to 1
    data.append(incoming_value)

    # Limit the size of the data list to the max number of points
    if len(data) > max_points:
        data.pop(0)  # Remove the first element (oldest data)

    # Plot the data
    for i in range(1, len(data)):
        x1 = (i - 1) * x_scale
        y1 = height // 2 - data[i - 1] * y_scale
        x2 = i * x_scale
        y2 = height // 2 - data[i] * y_scale
        pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 2)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
