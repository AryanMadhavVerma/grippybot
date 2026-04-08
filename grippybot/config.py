# Hardware constants for Grippy Bot
#
# Coordinate system (DH convention, right-hand rule):
#   X+ = forward (toward monitor)
#   Y+ = left
#   Z+ = up
#   Positive rotation = right-hand rule around each axis
#
# Base rotates around Z axis. Shoulder, elbow, wrist rotate around Y axis.
# All angles measured from home position = 0 degrees.

JOINTS = {
    "base":     {"gpio": 17, "min_pw": 500,  "max_pw": 2500,
                 "home_pw": 1500, "min_deg": -90, "max_deg": 90},
    "shoulder": {"gpio": 27, "min_pw": 550,  "max_pw": 1500,
                 "home_pw": 550,  "min_deg": -5,  "max_deg": 90},
    "elbow":    {"gpio": 22, "min_pw": 500,  "max_pw": 2500,
                 "home_pw": 1900, "min_deg": -120, "max_deg": 70},
    "wrist":    {"gpio": 23, "min_pw": 500,  "max_pw": 2450,
                 "home_pw": 1650, "min_deg": -95,  "max_deg": 90},
    "gripper":  {"gpio": 24, "min_pw": 1450, "max_pw": 1900,
                 "home_pw": 1900, "min_deg": 0, "max_deg": 1},  # binary: open=1900, close=1450
}

CENTER_PW = 1500  # microseconds
STEP_PW = 50      # microseconds per step in test
