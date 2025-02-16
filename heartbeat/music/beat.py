import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Keystroke Sound Player")

# Pygame Mixer settings
SAMPLE_RATE = 44100  # Sample rate in Hz
BITS = 16
pygame.mixer.pre_init(SAMPLE_RATE, -BITS, 1)
pygame.mixer.init()

C_MAJOR_SCALE = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
LOW_C = 130.81  # C3

# Generate a Sine Wave Sound
def generate_sine_wave(frequency=440, duration=1.0, sample_rate=SAMPLE_RATE):
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate  # Time values
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate sine wave
    wave = (wave * (2**(BITS - 1) - 1)).astype(np.int16)  # Convert to 16-bit PCM
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))  # Stereo sound

# Create a sound object for a 440Hz (A4) tone
# sound = generate_sine_wave(440)

# Dictionary to track key states
key_held = {}

# Main loop
running = True
while running:
    screen.fill((255, 255, 255))  # Clear screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # When a key is pressed
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and event.key not in key_held:
                frequency = C_MAJOR_SCALE[0]  # Start with C3
                sound = generate_sine_wave(frequency)
                sound.play(-1)  # Start looping the sound
                key_held[event.key] = True  # Mark key as held

        # When a key is released
        if event.type == pygame.KEYUP:
            if event.key in key_held:
                sound.stop()  # Stop the sound
                del key_held[event.key]  # Remove from held keys

    pygame.display.flip()  # Update display

pygame.quit()
