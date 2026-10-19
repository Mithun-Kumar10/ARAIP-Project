import random
import time


def read_distance_cm():
    """Simulated ultrasonic sensor: returns a distance in cm."""
    return random.randint(5, 100)


if __name__ == "__main__":
    for _ in range(10):
        print(f"Distance: {read_distance_cm()} cm")
        time.sleep(0.5)