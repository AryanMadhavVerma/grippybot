"""
PyTorch Dataset for ACT training.
Loads episode.json + JPEGs directly — no LeRobot dependency.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# ImageNet normalization (for pretrained ResNet18)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

JOINT_ORDER = ["base", "shoulder", "elbow", "wrist"]


def step_to_state(step):
    """Extract 5-dim state vector from an episode step."""
    return np.array([
        step["joint_angles"]["base"],
        step["joint_angles"]["shoulder"],
        step["joint_angles"]["elbow"],
        step["joint_angles"]["wrist"],
        float(step["gripper_state"]),
    ], dtype=np.float32)


class ACTDataset(Dataset):
    def __init__(self, data_dir, chunk_size=50, stats=None):
        self.data_dir = data_dir
        self.chunk_size = chunk_size

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Load all episodes into memory (JSON is small)
        self.episodes = []
        ep_dirs = sorted([
            d for d in os.listdir(data_dir) if d.startswith("episode_")
        ])
        for ep_dir in ep_dirs:
            ep_path = os.path.join(data_dir, ep_dir, "episode.json")
            with open(ep_path) as f:
                steps = json.load(f)["steps"]
            self.episodes.append({
                "steps": steps,
                "dir": os.path.join(data_dir, ep_dir),
            })

        # Build flat index: (episode_idx, step_idx)
        # Exclude last step of each episode (no next action)
        self.index = []
        for ep_idx, ep in enumerate(self.episodes):
            for step_idx in range(len(ep["steps"]) - 1):
                self.index.append((ep_idx, step_idx))

        print(f"ACTDataset: {len(self.episodes)} episodes, {len(self.index)} samples")

        # Compute or load normalization stats
        if stats is None:
            self.stats = self.compute_stats(data_dir)
        else:
            self.stats = stats

        self.state_mean = torch.tensor(self.stats["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(self.stats["state_std"], dtype=torch.float32)

    @staticmethod
    def compute_stats(data_dir):
        """Compute mean and std of state vectors across all episodes."""
        all_states = []
        ep_dirs = sorted([
            d for d in os.listdir(data_dir) if d.startswith("episode_")
        ])
        for ep_dir in ep_dirs:
            ep_path = os.path.join(data_dir, ep_dir, "episode.json")
            with open(ep_path) as f:
                steps = json.load(f)["steps"]
            for step in steps:
                all_states.append(step_to_state(step))

        all_states = np.stack(all_states)
        stats = {
            "state_mean": all_states.mean(axis=0).tolist(),
            "state_std": all_states.std(axis=0).tolist(),
        }
        # Prevent division by zero
        stats["state_std"] = [max(s, 1e-6) for s in stats["state_std"]]
        print(f"Stats: mean={stats['state_mean']}, std={stats['state_std']}")
        return stats

    def normalize_state(self, state):
        """Normalize a state tensor."""
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state):
        """Denormalize a state tensor."""
        return state * self.state_std + self.state_mean

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, step_idx = self.index[idx]
        ep = self.episodes[ep_idx]
        steps = ep["steps"]

        # Load image
        frame_path = os.path.join(ep["dir"], steps[step_idx]["frame"])
        image = Image.open(frame_path).convert("RGB")
        image = self.image_transform(image)  # [3, 480, 640]

        # Current state
        state = torch.tensor(step_to_state(steps[step_idx]), dtype=torch.float32)

        # Build action chunk: next chunk_size states
        actions = []
        action_is_pad = []
        for i in range(self.chunk_size):
            future_idx = step_idx + 1 + i
            if future_idx < len(steps):
                actions.append(step_to_state(steps[future_idx]))
                action_is_pad.append(False)
            else:
                # Pad with last available state
                actions.append(step_to_state(steps[-1]))
                action_is_pad.append(True)

        actions = torch.tensor(np.stack(actions), dtype=torch.float32)  # [50, 5]
        action_is_pad = torch.tensor(action_is_pad, dtype=torch.bool)  # [50]

        # Normalize
        state_norm = self.normalize_state(state)
        actions_norm = self.normalize_state(actions)  # same normalization

        return {
            "image": image,            # [3, 480, 640]
            "state": state_norm,        # [5]
            "actions": actions_norm,    # [50, 5]
            "action_is_pad": action_is_pad,  # [50]
        }


if __name__ == "__main__":
    # Quick test
    ds = ACTDataset("data", chunk_size=50)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"State shape: {sample['state'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Pad mask shape: {sample['action_is_pad'].shape}")
    print(f"Pad mask sum: {sample['action_is_pad'].sum()} (of {len(sample['action_is_pad'])})")
