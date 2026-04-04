"""
Servo tester — test each joint one at a time.
Controls:
  j/k  = previous/next servo
  a/d  = decrease/increase pulse width by 50us
  c    = go to center (1500)
  0    = turn off current servo (stop signal)
  q    = quit (turns off all servos)
"""

import pigpio
import sys
import termios
import tty
from config import JOINTS, CENTER_PW, STEP_PW


def getch():
    """Read a single keypress."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    pi = pigpio.pi()
    if not pi.connected:
        print("ERROR: pigpiod not running. Run: sudo pigpiod")
        return

    joint_names = list(JOINTS.keys())
    current_idx = 0
    pulse_widths = {name: CENTER_PW for name in joint_names}

    # Start with all servos off
    for name in joint_names:
        pi.set_servo_pulsewidth(JOINTS[name]["gpio"], 0)

    def display():
        name = joint_names[current_idx]
        pw = pulse_widths[name]
        gpio = JOINTS[name]["gpio"]
        print(f"\r\033[K  [{current_idx+1}/5] {name:>10} (GPIO {gpio:2d})  "
              f"pulse: {pw:4d}us  |  a/d=move  c=center  0=off  j/k=switch  q=quit", end="", flush=True)

    print("Servo Tester — plug in charger, then press any key to start")
    print("WARNING: servos will move! Keep fingers clear.\n")
    getch()

    # Send center to first servo
    name = joint_names[current_idx]
    pi.set_servo_pulsewidth(JOINTS[name]["gpio"], CENTER_PW)
    display()

    while True:
        key = getch()

        if key == "q":
            break
        elif key == "k":
            # Turn off current servo before switching
            pi.set_servo_pulsewidth(JOINTS[joint_names[current_idx]]["gpio"], 0)
            current_idx = (current_idx + 1) % len(joint_names)
            name = joint_names[current_idx]
            pi.set_servo_pulsewidth(JOINTS[name]["gpio"], pulse_widths[name])
        elif key == "j":
            pi.set_servo_pulsewidth(JOINTS[joint_names[current_idx]]["gpio"], 0)
            current_idx = (current_idx - 1) % len(joint_names)
            name = joint_names[current_idx]
            pi.set_servo_pulsewidth(JOINTS[name]["gpio"], pulse_widths[name])
        elif key == "d":
            name = joint_names[current_idx]
            pw = min(pulse_widths[name] + STEP_PW, JOINTS[name]["max_pw"])
            pulse_widths[name] = pw
            pi.set_servo_pulsewidth(JOINTS[name]["gpio"], pw)
        elif key == "a":
            name = joint_names[current_idx]
            pw = max(pulse_widths[name] - STEP_PW, JOINTS[name]["min_pw"])
            pulse_widths[name] = pw
            pi.set_servo_pulsewidth(JOINTS[name]["gpio"], pw)
        elif key == "c":
            name = joint_names[current_idx]
            pulse_widths[name] = CENTER_PW
            pi.set_servo_pulsewidth(JOINTS[name]["gpio"], CENTER_PW)
        elif key == "0":
            name = joint_names[current_idx]
            pi.set_servo_pulsewidth(JOINTS[name]["gpio"], 0)

        display()

    # Cleanup — turn off all servos
    for name in joint_names:
        pi.set_servo_pulsewidth(JOINTS[name]["gpio"], 0)
    pi.stop()
    print("\nAll servos off. Done.")


if __name__ == "__main__":
    main()
