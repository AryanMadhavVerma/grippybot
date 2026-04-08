# Scripts

Standalone utilities that aren't part of the grippybot package. Install the package first: `pip install -e .`

## servo_test.py
Interactive servo calibration tool. **Run this first** when building a new arm to find the safe pulse width ranges for each joint.

```bash
# On Pi:
python scripts/servo_test.py
```

Controls: `j/k` = switch servo, `a/d` = adjust pulse width, `c` = center, `0` = off, `q` = quit.

Record the min/max pulse widths where each servo starts jittering, then update `grippybot/config.py`.

## convert_dataset.py
Converts episode data to LeRobot format. Reference only — not needed for the default training pipeline.
