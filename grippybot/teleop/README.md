# Teleop

Keyboard teleoperation over SSH with integrated data recording.

## Usage
```bash
# On Pi (over SSH):
grippybot-teleop
```

## Keybindings
| Key     | Action              |
|---------|---------------------|
| Q / A   | Base +/-            |
| W / S   | Shoulder +/-        |
| E / D   | Elbow +/-           |
| R / F   | Wrist +/-           |
| Space   | Gripper toggle      |
| H       | Home position       |
| T       | Toggle recording    |
| Esc     | Quit                |

Step size: 5 degrees per keypress. Loop runs at 10Hz.

## Recording Workflow
1. Position the object in the workspace
2. Press `T` to start recording
3. Teleoperate the arm to pick up the object
4. Press `T` to stop — episode saved to `data/episode_XXX/`
5. Repeat with object in different positions

Each episode saves JPEG frames + `episode.json` with joint angles, gripper state, and timestamps.
