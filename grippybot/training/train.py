"""
Training loop for ACT policy. Pure PyTorch — no LeRobot, no Accelerate.
Explicit model.to(device) for guaranteed GPU usage.

Usage:
    grippybot-train                        # auto-detect device
    grippybot-train --device cuda          # force CUDA
    grippybot-train --steps 1000           # short test run
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grippybot.model.act import ACTPolicy
from grippybot.model.dataset import ACTDataset

# ── Config ──────────────────────────────────────────────────────────────────

D_MODEL = 256
LATENT_DIM = 32
CHUNK_SIZE = 50
STATE_DIM = 5
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 1
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
BETA = 10.0  # KL weight
NUM_WORKERS = 4
LOG_FREQ = 100
SAVE_FREQ = 10000
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"


def compute_loss(pred_actions, target_actions, pad_mask, mu, logvar, beta=BETA):
    """L1 reconstruction + KL divergence."""
    # L1 loss, masked for padded positions
    l1 = F.l1_loss(pred_actions, target_actions, reduction="none")  # [B, chunk, 5]
    mask = (~pad_mask).unsqueeze(-1).float()  # [B, chunk, 1]
    l1 = (l1 * mask).sum() / mask.sum() / STATE_DIM

    # KL divergence: push q(z|x) toward N(0,1)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = l1 + beta * kl
    return loss, l1.item(), kl.item()


def save_checkpoint(model, optimizer, step, stats, path, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if config is None:
        config = {
            "d_model": D_MODEL,
            "latent_dim": LATENT_DIM,
            "chunk_size": CHUNK_SIZE,
            "state_dim": STATE_DIM,
            "n_encoder_layers": N_ENCODER_LAYERS,
            "n_decoder_layers": N_DECODER_LAYERS,
        }
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "stats": stats,
        "config": config,
    }, path)
    print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Dataset
    print("Loading dataset...")
    stats = ACTDataset.compute_stats(args.data_dir)
    dataset = ACTDataset(args.data_dir, chunk_size=CHUNK_SIZE, stats=stats)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"DataLoader: {len(loader)} batches per epoch")

    # Model — explicitly on device
    print("Creating model...")
    model_config = {
        "d_model": D_MODEL,
        "latent_dim": LATENT_DIM,
        "chunk_size": CHUNK_SIZE,
        "state_dim": STATE_DIM,
        "n_encoder_layers": N_ENCODER_LAYERS,
        "n_decoder_layers": N_DECODER_LAYERS,
    }
    model = ACTPolicy(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Verify model is on correct device
    param_device = next(model.parameters()).device
    print(f"Model device: {param_device}")
    assert str(param_device).startswith(str(device)), f"Model on {param_device}, expected {device}!"

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop
    print(f"\nStarting training: {args.steps} steps, batch_size={args.batch_size}")
    print(f"Epochs needed: ~{args.steps * args.batch_size / len(dataset):.0f}")
    print("-" * 60)

    model.train()
    step = 0
    start_time = time.time()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            image = batch["image"].to(device)              # [B, 3, 480, 640]
            state = batch["state"].to(device)               # [B, 5]
            actions = batch["actions"].to(device)           # [B, 50, 5]
            pad_mask = batch["action_is_pad"].to(device)    # [B, 50]

            pred_actions, mu, logvar = model(image, state, actions)
            loss, l1, kl = compute_loss(pred_actions, actions, pad_mask, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            if step % LOG_FREQ == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                eta_hours = (args.steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
                print(f"step {step:6d}/{args.steps} | loss {loss.item():.4f} | "
                      f"L1 {l1:.4f} | KL {kl:.6f} | "
                      f"{steps_per_sec:.1f} step/s | ETA {eta_hours:.1f}h")

            if step > 0 and step % SAVE_FREQ == 0:
                path = os.path.join(args.checkpoint_dir, f"act_step_{step}.pt")
                save_checkpoint(model, optimizer, step, stats, path, config=model_config)

            step += 1

    # Final checkpoint
    path = os.path.join(args.checkpoint_dir, "act_final.pt")
    save_checkpoint(model, optimizer, step, stats, path, config=model_config)
    print(f"\nTraining complete! {step} steps in {(time.time()-start_time)/3600:.1f} hours")


if __name__ == "__main__":
    main()
