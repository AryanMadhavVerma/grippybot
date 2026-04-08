# Hardware

Raspberry Pi reference implementation for servo control and camera capture.

**This is our setup — bring your own.** The interface is what matters, not the specific board. Contributors can add support for Arduino, Jetson Nano, USB cameras, etc.

## Interface Contract

### ServoDriver
Any servo driver should implement:
- `set_angle(name, angle)` — move joint to angle (degrees from home)
- `get_angle(name) -> float` — get current commanded angle
- `home()` — move all joints to home position
- `gripper_open()` / `gripper_close()` — binary gripper control
- `close()` — cleanup

### Camera
Any camera should implement:
- `capture_frame() -> numpy array (H, W, 3) RGB`
- `close()` — cleanup

## Our Setup (Pi Reference)
- **Board**: Raspberry Pi 4
- **Servos**: 5x SG90 via pigpio (hardware-timed PWM)
- **Camera**: Pi Camera Module (5MP, 640x480)
- **Power**: External 5V 3A DC supply for servos (NOT from Pi USB)

## GPIO Wiring
| Joint    | GPIO | Signal Wire Color |
|----------|------|-------------------|
| Base     | 17   | varies            |
| Shoulder | 27   | varies            |
| Elbow    | 22   | varies            |
| Wrist    | 23   | varies            |
| Gripper  | 24   | varies            |

All servo power (red) and ground (brown) wires go to the breadboard, powered by the external 5V supply. Pi GND (pin 6) connects to the breadboard ground rail.

## Pi Setup

Setting up a fresh Pi from scratch (flashing OS, installing pigpiod, cloning the repo, copying files to/from Pi)? See **[PI_SETUP.md](PI_SETUP.md)**.

## Calibration
Run `scripts/servo_test.py` to find safe pulse width ranges per joint, then update `config.py`.
