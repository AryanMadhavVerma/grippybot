"""
ACT Policy Evaluation — offline validation and real robot inference.

Modes:
  grippybot-eval --mode offline --episode 5     # replay episode, compare predictions vs actual
  grippybot-eval --mode robot                    # run on real robot (Pi camera + servos)
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from grippybot.model.act import ACTPolicy
from grippybot.model.dataset import IMAGENET_MEAN, IMAGENET_STD, step_to_state
from grippybot.model.ensemble import TemporalEnsemble


# ── Model Loading ───────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """Load trained ACT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    stats = ckpt["stats"]

    model = ACTPolicy(
        d_model=config["d_model"],
        latent_dim=config["latent_dim"],
        chunk_size=config["chunk_size"],
        state_dim=config["state_dim"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    state_mean = torch.tensor(stats["state_mean"], device=device)
    state_std = torch.tensor(stats["state_std"], device=device)

    print(f"Model loaded: step {ckpt['step']}, {sum(p.numel() for p in model.parameters()):,} params")
    print(f"State mean: {stats['state_mean']}")
    print(f"State std: {stats['state_std']}")

    return model, state_mean, state_std, config


def preprocess_image(image_pil, device):
    """PIL Image → normalized tensor [1, 3, 480, 640]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image_pil).unsqueeze(0).to(device)


@torch.no_grad()
def predict(model, image_tensor, state_tensor, state_mean, state_std):
    """Run model inference. Returns denormalized actions [chunk_size, 5]."""
    state_norm = (state_tensor - state_mean) / state_std
    pred_norm, _, _ = model(image_tensor, state_norm.unsqueeze(0))  # [1, chunk, 5]
    pred = pred_norm[0] * state_std + state_mean  # denormalize
    return pred.cpu().numpy()


# ── Offline Validation ──────────────────────────────────────────────────────

def evaluate_offline(model, state_mean, state_std, config, device,
                     data_dir="data", episode_idx=0, use_ensemble=True):
    """Replay a training episode, compare model predictions to actual actions."""
    # Load episode
    ep_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith("episode_")])
    ep_dir = os.path.join(data_dir, ep_dirs[episode_idx])
    with open(os.path.join(ep_dir, "episode.json")) as f:
        steps = json.load(f)["steps"]

    print(f"\nOffline evaluation: {ep_dirs[episode_idx]} ({len(steps)} steps)")
    print(f"Temporal ensembling: {'ON' if use_ensemble else 'OFF'}")
    print("-" * 70)

    chunk_size = config["chunk_size"]
    ensemble = TemporalEnsemble(chunk_size, config["state_dim"]) if use_ensemble else None

    errors = []
    joint_names = ["base", "shoulder", "elbow", "wrist", "gripper"]

    for t in range(len(steps) - 1):
        # Load image and state at time t
        image = Image.open(os.path.join(ep_dir, steps[t]["frame"])).convert("RGB")
        image_tensor = preprocess_image(image, device)
        state = torch.tensor(step_to_state(steps[t]), device=device)

        # Predict
        pred_actions = predict(model, image_tensor, state, state_mean, state_std)

        if use_ensemble:
            ensemble.add_chunk(pred_actions)
            action = ensemble.get_action()
        else:
            action = pred_actions[0]  # just take first predicted action

        # Actual next state
        actual = step_to_state(steps[t + 1])

        # Error
        error = np.abs(action - actual)
        errors.append(error)

        if t % 20 == 0 or t == len(steps) - 2:
            print(f"  step {t:3d} | pred: [{', '.join(f'{a:6.1f}' for a in action)}] | "
                  f"actual: [{', '.join(f'{a:6.1f}' for a in actual)}] | "
                  f"err: [{', '.join(f'{e:5.2f}' for e in error)}]")

    errors = np.stack(errors)
    mean_err = errors.mean(axis=0)
    max_err = errors.max(axis=0)

    print(f"\n{'Joint':<12} {'Mean Error':>10} {'Max Error':>10}")
    print("-" * 35)
    for i, name in enumerate(joint_names):
        print(f"{name:<12} {mean_err[i]:>10.2f}\u00b0 {max_err[i]:>10.2f}\u00b0")
    print(f"{'OVERALL':<12} {mean_err[:4].mean():>10.2f}\u00b0 {max_err[:4].max():>10.2f}\u00b0")
    print(f"\nGripper accuracy: {(errors[:, 4] < 0.5).mean() * 100:.1f}%")

    return mean_err


# ── Real Robot Inference ────────────────────────────────────────────────────

def evaluate_robot(model, state_mean, state_std, config, device, fps=6):
    """Run policy on real robot. Requires Pi with camera + servos."""
    from grippybot.hardware import ServoDriver, Camera

    chunk_size = config["chunk_size"]
    ensemble = TemporalEnsemble(chunk_size, config["state_dim"])

    driver = ServoDriver()
    camera = Camera()
    driver.home()
    time.sleep(2)

    joint_names = ["base", "shoulder", "elbow", "wrist"]
    print("\nRunning policy on robot. Press Ctrl+C to stop.")
    print("-" * 50)

    step = 0
    try:
        while True:
            t_start = time.time()

            # Capture image
            frame = camera.capture_frame()  # numpy HWC RGB
            image = Image.fromarray(frame)
            image_tensor = preprocess_image(image, device)

            # Read current joint state
            angles = driver.get_all_angles()
            gripper = 1.0 if driver.is_gripper_open() else 0.0
            state = torch.tensor([
                angles.get("base", 0.0) or 0.0,
                angles.get("shoulder", 0.0) or 0.0,
                angles.get("elbow", 0.0) or 0.0,
                angles.get("wrist", 0.0) or 0.0,
                gripper,
            ], dtype=torch.float32, device=device)

            # Predict
            pred_actions = predict(model, image_tensor, state, state_mean, state_std)
            ensemble.add_chunk(pred_actions)
            action = ensemble.get_action()

            # Execute
            driver.set_angle("base", float(action[0]))
            driver.set_angle("shoulder", float(action[1]))
            driver.set_angle("elbow", float(action[2]))
            driver.set_angle("wrist", float(action[3]))
            if action[4] > 0.5:
                driver.gripper_open()
            else:
                driver.gripper_close()

            if step % 5 == 0:
                print(f"  step {step:3d} | action: [{', '.join(f'{a:6.1f}' for a in action)}]")

            step += 1

            # Match training FPS
            elapsed = time.time() - t_start
            sleep_time = max(0, 1.0 / fps - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        driver.home()
        time.sleep(1)
        driver.close()
        camera.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "robot"], default="offline")
    parser.add_argument("--checkpoint", default="checkpoints/act_final.pt")
    parser.add_argument("--episode", type=int, default=0, help="Episode index for offline mode")
    parser.add_argument("--no_ensemble", action="store_true", help="Disable temporal ensembling")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    model, state_mean, state_std, config = load_model(args.checkpoint, device)

    if args.mode == "offline":
        evaluate_offline(
            model, state_mean, state_std, config, device,
            data_dir=args.data_dir, episode_idx=args.episode,
            use_ensemble=not args.no_ensemble,
        )
    elif args.mode == "robot":
        evaluate_robot(model, state_mean, state_std, config, device)


if __name__ == "__main__":
    main()
