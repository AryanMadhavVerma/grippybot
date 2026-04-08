"""
Data recorder — synchronized logging of frames + joint state during teleop.

Each episode is saved as a directory with PNG frames and a JSON metadata file.
"""

import os
import json
import time
from PIL import Image


class DataRecorder:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.recording = False
        self.episode_dir = None
        self.steps = []
        self.step_count = 0
        self.start_time = 0

    def _next_episode_dir(self):
        """Find next episode number and create directory."""
        os.makedirs(self.data_dir, exist_ok=True)
        existing = [d for d in os.listdir(self.data_dir) if d.startswith("episode_")]
        nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
        next_num = max(nums) + 1 if nums else 0
        ep_dir = os.path.join(self.data_dir, f"episode_{next_num:03d}")
        os.makedirs(ep_dir)
        return ep_dir

    def start(self):
        """Start recording a new episode."""
        self.episode_dir = self._next_episode_dir()
        self.steps = []
        self.step_count = 0
        self.start_time = time.time()
        self.recording = True

    def stop(self):
        """Stop recording and save metadata."""
        if not self.recording:
            return
        self.recording = False
        meta_path = os.path.join(self.episode_dir, "episode.json")
        with open(meta_path, "w") as f:
            json.dump({"steps": self.steps}, f, indent=2)

    def record_step(self, frame, joint_angles, gripper_state, delta):
        """Record one timestep: save frame and log state.

        Args:
            frame: numpy array (H, W, 3) RGB image
            joint_angles: dict of joint angles {"base": 0.0, ...}
            gripper_state: 1 for open, 0 for closed
            delta: dict of angle deltas applied this step
        """
        if not self.recording:
            return

        frame_name = f"frame_{self.step_count:04d}.jpg"
        frame_path = os.path.join(self.episode_dir, frame_name)
        Image.fromarray(frame).save(frame_path, quality=95)

        self.steps.append({
            "timestamp": round(time.time() - self.start_time, 3),
            "joint_angles": joint_angles,
            "gripper_state": gripper_state,
            "delta": delta,
            "frame": frame_name,
        })
        self.step_count += 1
