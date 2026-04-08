# Raspberry Pi Setup

How to set up a fresh Raspberry Pi for grippybot (servo control, camera, teleop, inference server).

## 1. Flash the OS

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash **Raspberry Pi OS (64-bit, Lite)** to an SD card.

In the imager settings (gear icon), set:
- Hostname: `raspi.local` (or whatever you prefer)
- Enable SSH (password or key-based)
- Set Wi-Fi credentials
- Set username/password

## 2. First boot + SSH in

Insert the SD card, power on the Pi, and SSH in:
```bash
ssh pi@raspi.local
```

Update packages:
```bash
sudo apt update && sudo apt upgrade -y
```

## 3. Enable camera and pigpio

```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Install pigpio (hardware-timed PWM for servos)
sudo apt install -y pigpio python3-pigpio

# Start pigpiod (must run before any servo code)
sudo pigpiod

# Auto-start pigpiod on boot
sudo systemctl enable pigpiod
```

## 4. Clone and install grippybot

```bash
# Install pip/venv if not already present
sudo apt install -y python3-pip python3-venv

# Clone the repo
git clone https://github.com/AryanMadhavVerma/grippybot.git
cd grippybot

# Create venv and install with Pi extras
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pi]"
```

## 5. Verify hardware

Test the camera:
```bash
python -c "from grippybot.hardware import Camera; c = Camera(); print(c.capture_frame().shape); c.close()"
# Should print: (480, 640, 3)
```

Test a servo (one at a time, keep fingers clear):
```bash
python scripts/servo_test.py
```

## 6. Calibrate servos

With the arm assembled, run the servo tester to find safe min/max pulse widths for each joint:
```bash
python scripts/servo_test.py
```

Record the values and update `grippybot/config.py` on your development machine. Then push and pull, or edit directly on the Pi.

## Copying files to/from the Pi

### Copy training data from Pi to your Mac/GPU machine:
```bash
# From your Mac:
scp -r pi@raspi.local:~/grippybot/data/ ./data/
```

### Copy a trained checkpoint to the Pi:
```bash
# From your Mac:
scp checkpoints/act_final.pt pi@raspi.local:~/grippybot/checkpoints/
```

### Sync the whole repo (alternative to git):
```bash
# From your Mac — push code changes to Pi:
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'data' --exclude 'checkpoints' \
  ./ pi@raspi.local:~/grippybot/
```

## Running on the Pi

### Teleoperate (collect demos):
```bash
source .venv/bin/activate
grippybot-teleop
# Press T to start/stop recording. Episodes saved to data/
```

### Inference server (Pi captures + executes, Mac runs model):
```bash
source .venv/bin/activate
grippybot-server --port 5555
# Then on Mac: grippybot-client --host raspi.local --port 5555
```

### Run evaluation directly on Pi (slow, CPU-only):
```bash
grippybot-eval --mode robot --checkpoint checkpoints/act_final.pt --device cpu
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pigpiod not running` | Run `sudo pigpiod` |
| Camera not detected | Run `sudo raspi-config` > Interface Options > Camera > Enable, then reboot |
| `ImportError: pigpio` | Make sure you installed with `pip install -e ".[pi]"` |
| SSH connection refused | Check Pi is on the same network, try `ping raspi.local` |
| Servos jittering | Check power supply — servos need 5V 3A external, not Pi USB power |
| `Address already in use` on port 5555 | Previous server didn't clean up: `kill $(lsof -t -i:5555)` |
