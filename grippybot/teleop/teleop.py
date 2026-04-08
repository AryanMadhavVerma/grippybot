"""
Keyboard teleoperation for Grippy Bot over SSH.

Controls:
  Q/A  — base +/-
  W/S  — shoulder +/-
  E/D  — elbow +/-
  R/F  — wrist +/-
  Space — gripper toggle
  H    — home position
  T    — toggle recording
  Esc  — quit
"""

import curses
import time
from grippybot.hardware import ServoDriver, Camera
from grippybot.teleop.recorder import DataRecorder
from grippybot.config import JOINTS

STEP_DEG = 5  # degrees per keypress

# Key mapping: key -> (joint_name, direction)
KEY_MAP = {
    ord('q'): ("base", +1),
    ord('a'): ("base", -1),
    ord('w'): ("shoulder", +1),
    ord('s'): ("shoulder", -1),
    ord('e'): ("elbow", +1),
    ord('d'): ("elbow", -1),
    ord('r'): ("wrist", +1),
    ord('f'): ("wrist", -1),
}

JOINT_NAMES = [name for name in JOINTS if name != "gripper"]


def draw_ui(stdscr, driver, recorder, fps=0.0):
    """Draw current joint state on screen."""
    stdscr.clear()
    stdscr.addstr(0, 0, "=== Grippy Bot Teleop ===")
    stdscr.addstr(1, 0, "Q/A=base  W/S=shoulder  E/D=elbow  R/F=wrist  Space=gripper  H=home  Esc=quit")

    rec_status = "OFF"
    if recorder.recording:
        rec_status = f"REC episode {recorder.episode_dir} | step {recorder.step_count}"
    stdscr.addstr(2, 0, f"Step: {STEP_DEG} deg  |  Recording: {rec_status}  |  T=toggle rec  |  FPS: {fps:.1f}")
    stdscr.addstr(3, 0, "-" * 70)

    row = 4
    for name in JOINTS:
        if name == "gripper":
            state = "OPEN" if driver.is_gripper_open() else "CLOSED"
            pw = driver.get_pw(name)
            stdscr.addstr(row, 0, f"  {name:>10}:  {state}  (pw: {pw})")
        else:
            angle = driver.get_angle(name)
            pw = driver.get_pw(name)
            j = JOINTS[name]
            angle_str = f"{angle:+6.1f} deg" if angle is not None else "   OFF"
            stdscr.addstr(row, 0, f"  {name:>10}:  {angle_str}  (pw: {pw})  [{j['min_deg']} to {j['max_deg']}]")
        row += 1

    stdscr.refresh()


def _main(stdscr):
    curses.curs_set(0)
    stdscr.timeout(100)  # 10Hz

    driver = ServoDriver()
    camera = Camera()
    recorder = DataRecorder()

    driver.home()
    last_tick = time.time()
    fps = 0.0
    draw_ui(stdscr, driver, recorder, fps)

    try:
        while True:
            now = time.time()
            dt = now - last_tick
            fps = 1.0 / dt if dt > 0 else 0.0
            last_tick = now

            key = stdscr.getch()

            # Track delta for this step
            delta = {name: 0 for name in JOINT_NAMES}
            delta["gripper"] = 0

            if key == 27:  # Esc
                if recorder.recording:
                    recorder.stop()
                break
            elif key == ord('t'):  # toggle recording
                if recorder.recording:
                    recorder.stop()
                else:
                    recorder.start()
            elif key == ord('h'):
                driver.home()
            elif key == ord(' '):
                if driver.is_gripper_open():
                    driver.gripper_close()
                    delta["gripper"] = -1
                else:
                    driver.gripper_open()
                    delta["gripper"] = 1
            elif key in KEY_MAP:
                joint, direction = KEY_MAP[key]
                current = driver.get_angle(joint)
                if current is None:
                    current = 0.0
                new_angle = current + direction * STEP_DEG
                driver.set_angle(joint, new_angle)
                actual = driver.get_angle(joint)
                delta[joint] = round(actual - current, 1)

            # Record at every tick if recording
            if recorder.recording:
                frame = camera.capture_frame()
                joint_angles = {name: driver.get_angle(name) or 0.0 for name in JOINT_NAMES}
                gripper_state = 1 if driver.is_gripper_open() else 0
                recorder.record_step(frame, joint_angles, gripper_state, delta)

            draw_ui(stdscr, driver, recorder, fps)

    finally:
        driver.close()
        camera.close()


def main():
    curses.wrapper(_main)


if __name__ == "__main__":
    main()
