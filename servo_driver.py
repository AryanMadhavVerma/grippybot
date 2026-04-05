"""
Servo driver — pigpio wrapper for Grippy Bot.

Converts angles (degrees from home position) to pulse widths.
Piecewise linear mapping: different us/degree ratio on each side of home.
"""

import time
import pigpio
from config import JOINTS

MOVE_DELAY = 0.3  # seconds between sequential servo moves


class ServoDriver:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod not running. Run: sudo pigpiod")
        self._current_pw = {}
        for name in JOINTS:
            self.pi.set_servo_pulsewidth(JOINTS[name]["gpio"], 0)
            self._current_pw[name] = 0

    def angle_to_pw(self, name, angle):
        """Convert angle (degrees from home) to pulse width."""
        j = JOINTS[name]
        angle = max(j["min_deg"], min(j["max_deg"], angle))
        if angle >= 0:
            pw = j["home_pw"] + angle * (j["max_pw"] - j["home_pw"]) / j["max_deg"]
        else:
            pw = j["home_pw"] + angle * (j["home_pw"] - j["min_pw"]) / (-j["min_deg"])
        return int(pw)

    def pw_to_angle(self, name, pw):
        """Convert pulse width back to angle (degrees from home)."""
        j = JOINTS[name]
        if pw >= j["home_pw"]:
            if j["max_pw"] == j["home_pw"]:
                return 0.0
            return (pw - j["home_pw"]) / (j["max_pw"] - j["home_pw"]) * j["max_deg"]
        else:
            if j["home_pw"] == j["min_pw"]:
                return 0.0
            return (pw - j["home_pw"]) / (j["home_pw"] - j["min_pw"]) * (-j["min_deg"])

    def set_angle(self, name, angle):
        """Move a joint to an angle (degrees from home)."""
        pw = self.angle_to_pw(name, angle)
        self.set_pw(name, pw)

    def set_pw(self, name, pw):
        """Set raw pulse width for a joint."""
        j = JOINTS[name]
        pw = max(j["min_pw"], min(j["max_pw"], int(pw)))
        self.pi.set_servo_pulsewidth(j["gpio"], pw)
        self._current_pw[name] = pw

    def get_angle(self, name):
        """Get current commanded angle (degrees from home). None if servo is off."""
        pw = self._current_pw[name]
        if pw == 0:
            return None
        return round(self.pw_to_angle(name, pw), 1)

    def get_pw(self, name):
        """Get current commanded pulse width."""
        return self._current_pw[name]

    def get_all_angles(self):
        """Get dict of all joint angles."""
        return {name: self.get_angle(name) for name in JOINTS if name != "gripper"}

    def home(self):
        """Move all joints to home position sequentially."""
        for name in JOINTS:
            self.set_pw(name, JOINTS[name]["home_pw"])
            time.sleep(MOVE_DELAY)

    def gripper_open(self):
        """Open the gripper."""
        self.set_pw("gripper", JOINTS["gripper"]["max_pw"])

    def gripper_close(self):
        """Close the gripper."""
        self.set_pw("gripper", JOINTS["gripper"]["min_pw"])

    def is_gripper_open(self):
        """Check if gripper is open."""
        return self._current_pw["gripper"] == JOINTS["gripper"]["max_pw"]

    def disable(self, name=None):
        """Stop sending signal. If name is None, disables all."""
        targets = [name] if name else list(JOINTS.keys())
        for n in targets:
            self.pi.set_servo_pulsewidth(JOINTS[n]["gpio"], 0)
            self._current_pw[n] = 0

    def close(self):
        """Disable all servos and disconnect."""
        self.disable()
        self.pi.stop()
