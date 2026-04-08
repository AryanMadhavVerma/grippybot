"""
Convert our episode data to LeRobot dataset format.

Reads data/episode_XXX/episode.json + frame_NNNN.jpg
and creates a LeRobot dataset using their API.
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA_DIR = "data"
REPO_ID = "grippy-learns/pick-tissue"
FPS = 6  # our actual capture rate (~5-8Hz, use 6 as average)

# Define features matching our data
FEATURES = {
    "observation.image": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["base", "shoulder", "elbow", "wrist", "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (5,),
        "names": ["base", "shoulder", "elbow", "wrist", "gripper"],
    },
}


def load_episode(ep_dir):
    """Load episode.json and return steps."""
    with open(os.path.join(ep_dir, "episode.json")) as f:
        data = json.load(f)
    return data["steps"]


def main():
    # Find all episodes
    ep_dirs = sorted(
        [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if d.startswith("episode_")]
    )
    print(f"Found {len(ep_dirs)} episodes")

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=Path("lerobot_dataset"),
        robot_type="grippy_bot",
        use_videos=False,  # store as images, not video
        image_writer_processes=0,
        image_writer_threads=4,
    )

    for ep_dir in ep_dirs:
        steps = load_episode(ep_dir)
        ep_name = os.path.basename(ep_dir)
        print(f"Converting {ep_name}: {len(steps)} steps")

        for i, step in enumerate(steps):
            # Load image
            frame_path = os.path.join(ep_dir, step["frame"])
            img = np.array(Image.open(frame_path))

            # Current state: 4 joints + gripper
            state = np.array([
                step["joint_angles"]["base"],
                step["joint_angles"]["shoulder"],
                step["joint_angles"]["elbow"],
                step["joint_angles"]["wrist"],
                float(step["gripper_state"]),
            ], dtype=np.float32)

            # Action = next state (what joints should be at next timestep)
            # For last step, action = current state (hold position)
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                action = np.array([
                    next_step["joint_angles"]["base"],
                    next_step["joint_angles"]["shoulder"],
                    next_step["joint_angles"]["elbow"],
                    next_step["joint_angles"]["wrist"],
                    float(next_step["gripper_state"]),
                ], dtype=np.float32)
            else:
                action = state.copy()

            frame = {
                "observation.image": img,
                "observation.state": state,
                "action": action,
                "task": "pick up tissue",
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  Saved {ep_name}")

    dataset.finalize()
    print(f"\nDone! Dataset saved to lerobot_dataset/")
    print(f"Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
