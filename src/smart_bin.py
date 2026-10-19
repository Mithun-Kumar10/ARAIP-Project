import random
import time

OPEN_THRESHOLD_CM = 20   # hand closer than this opens the lid
AUTO_CLOSE_DELAY_S = 3   # seconds before the lid closes again


def read_distance_cm():
    """Simulated ultrasonic sensor: returns a distance in cm."""
    return random.randint(5, 100)


def open_lid():
    print("Lid OPEN")


def close_lid():
    print("Lid CLOSED")


def main():
    lid_open = False
    opened_at = 0
    open_count = 0

    for _ in range(40):
        distance = read_distance_cm()
        now = time.time()
        print(f"Distance: {distance} cm")

        if distance < OPEN_THRESHOLD_CM and not lid_open:
            open_lid()
            lid_open = True
            opened_at = now
            open_count += 1
        elif lid_open and distance >= OPEN_THRESHOLD_CM and now - opened_at >= AUTO_CLOSE_DELAY_S:
            close_lid()
            lid_open = False

        time.sleep(0.5)

    print(f"Lid opened {open_count} times")


if __name__ == "__main__":
    main()