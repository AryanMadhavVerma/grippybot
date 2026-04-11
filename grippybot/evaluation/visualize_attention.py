"""
Visualize cross-attention weights from ACT's Action Decoder.

Shows which parts of the image (and state/z tokens) each action query
attends to — revealing what the model "looks at" when predicting each
future timestep.

Usage:
  python -m grippybot.evaluation.visualize_attention --episode 0 --steps 0 50 100
  python -m grippybot.evaluation.visualize_attention --episode 5 --steps 0  # single step
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from grippybot.evaluation.evaluate import load_model, preprocess_image
from grippybot.model.dataset import IMAGENET_MEAN, IMAGENET_STD, step_to_state


# ── Attention capture ──────────────────────────────────────────────────────

class AttentionCapture:
    """Hooks into the decoder's cross-attention to capture weights."""

    def __init__(self, model):
        self.weights = None
        self._handle = None
        # The single decoder layer's cross-attention module
        self.mha = model.decoder.transformer.layers[0].multihead_attn

    def attach(self):
        """Monkey-patch the MHA forward to return attention weights."""
        original_forward = self.mha.forward

        def patched_forward(query, key, value, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False  # per-head weights
            out, weights = original_forward(query, key, value, **kwargs)
            self.weights = weights.detach().cpu()  # [B, n_heads, 50, 302]
            return out, weights

        self.mha.forward = patched_forward
        self._original_forward = original_forward

    def detach(self):
        """Restore original forward."""
        if self._original_forward is not None:
            self.mha.forward = self._original_forward


# ── Visualization ──────────────────────────────────────────────────────────

def plot_attention_heatmap(attn_weights, image_pil, step_idx, save_dir):
    """
    Plot cross-attention heatmap: x=queries (time→), y=fuser tokens.
    Three separate panels for image/state/z with shared x-axis.
    Log-scale colormap to reveal low-attention detail.

    attn_weights: [n_heads, 50, 302]
    """
    from matplotlib.colors import LogNorm

    attn_avg = attn_weights.mean(dim=0).numpy()  # [50, 302]

    # Split into image (300), state (1), z (1)
    img_attn = attn_avg[:, :300].T    # [300, 50] — y=image patches, x=queries
    state_attn = attn_avg[:, 300:301].T  # [1, 50]
    z_attn = attn_avg[:, 301:302].T      # [1, 50]

    # Widen state and z for visibility (repeat rows)
    state_wide = np.repeat(state_attn, 20, axis=0)  # [20, 50]
    z_wide = np.repeat(z_attn, 20, axis=0)           # [20, 50]

    # Shared color range across all three panels (log scale)
    vmin = max(attn_avg.min(), 1e-5)  # floor to avoid log(0)
    vmax = attn_avg.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Layout: 3 heatmap rows on the left, camera frame on the right
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[300, 20, 20], width_ratios=[4, 1],
                          hspace=0.12, wspace=0.05)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_state = fig.add_subplot(gs[1, 0], sharex=ax_img)
    ax_z = fig.add_subplot(gs[2, 0], sharex=ax_img)
    ax_cam = fig.add_subplot(gs[:, 1])  # camera spans all rows on the right

    # Use extent to set x-axis to query indices [0, 50]
    extent_img = [0, img_attn.shape[1], 0, img_attn.shape[0]]  # [0, 50, 0, 300]
    extent_wide = [0, state_wide.shape[1], 0, 1]                # [0, 50, 0, 1]

    # Panel 1: Image patches (300 tokens)
    im = ax_img.imshow(img_attn, aspect="auto", cmap="inferno", norm=norm,
                       origin="lower", extent=extent_img)
    ax_img.set_ylabel("Image patches\n(300 tokens)")
    ax_img.set_title(
        f"Cross-Attention Weights — Step {step_idx}\n"
        f"(softmax-normalized: each column sums to 1.0 across all 302 tokens)",
        fontsize=12,
    )
    plt.setp(ax_img.get_xticklabels(), visible=False)

    # Panel 2: State token (widened)
    ax_state.imshow(state_wide, aspect="auto", cmap="inferno", norm=norm,
                    origin="lower", extent=extent_wide)
    ax_state.set_ylabel("State")
    ax_state.set_yticks([])
    plt.setp(ax_state.get_xticklabels(), visible=False)

    # Panel 3: z token (widened)
    ax_z.imshow(z_wide, aspect="auto", cmap="inferno", norm=norm,
                origin="lower", extent=extent_wide)
    ax_z.set_ylabel("z")
    ax_z.set_yticks([])
    ax_z.set_xlabel("Action query (timestep → future)")

    # Right panel: camera frame
    ax_cam.imshow(image_pil)
    ax_cam.set_title("Camera frame", fontsize=10)
    ax_cam.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=[ax_img, ax_state, ax_z], shrink=0.8, pad=0.02)
    cbar.set_label("Attention weight (log scale)\nSoftmax over 302 tokens → each column sums to 1.0")
    path = os.path.join(save_dir, f"attention_heatmap_step{step_idx:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_spatial_attention(attn_weights, image_pil, step_idx, save_dir):
    """
    Overlay attention on the original image for selected action queries.

    Shows which image patches each query attends to.
    attn_weights: [n_heads, 50, 302]
    """
    attn_avg = attn_weights.mean(dim=0).numpy()  # [50, 302]

    # Select representative queries: first, quarter, half, three-quarter, last
    query_indices = [0, 12, 25, 37, 49]
    query_labels = ["now", "+2s", "+4s", "+6s", "+8s"]
    image_np = np.array(image_pil)

    fig, axes = plt.subplots(1, len(query_indices) + 1, figsize=(20, 4))

    # First panel: original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (qi, label) in enumerate(zip(query_indices, query_labels)):
        ax = axes[i + 1]

        # Extract attention over image tokens only (first 300)
        img_attn = attn_avg[qi, :300]  # [300]

        # Reshape to 15x20 spatial grid (matches ResNet18 output)
        spatial_attn = img_attn.reshape(15, 20)

        # Upsample to image size
        spatial_attn_resized = np.array(
            Image.fromarray(spatial_attn).resize(
                (image_np.shape[1], image_np.shape[0]), Image.BILINEAR
            )
        )

        # Normalize to [0, 1] for overlay
        spatial_attn_resized = (spatial_attn_resized - spatial_attn_resized.min())
        denom = spatial_attn_resized.max()
        if denom > 0:
            spatial_attn_resized = spatial_attn_resized / denom

        ax.imshow(image_np)
        ax.imshow(spatial_attn_resized, cmap="jet", alpha=0.5)
        ax.set_title(f"Query {qi} ({label})")
        ax.axis("off")

    plt.suptitle(f"Spatial Attention per Query — Step {step_idx}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"spatial_attention_step{step_idx:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_head_attention(attn_weights, image_pil, step_idx, save_dir, query_idx=0):
    """
    Show what each of the 8 attention heads focuses on for a single query.

    attn_weights: [n_heads, 50, 302]
    """
    n_heads = attn_weights.shape[0]
    image_np = np.array(image_pil)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for h in range(n_heads):
        ax = axes[h // 4, h % 4]

        img_attn = attn_weights[h, query_idx, :300].numpy()
        spatial_attn = img_attn.reshape(15, 20)
        spatial_attn_resized = np.array(
            Image.fromarray(spatial_attn).resize(
                (image_np.shape[1], image_np.shape[0]), Image.BILINEAR
            )
        )
        spatial_attn_resized = (spatial_attn_resized - spatial_attn_resized.min())
        denom = spatial_attn_resized.max()
        if denom > 0:
            spatial_attn_resized = spatial_attn_resized / denom

        ax.imshow(image_np)
        ax.imshow(spatial_attn_resized, cmap="jet", alpha=0.5)
        ax.set_title(f"Head {h}")
        ax.axis("off")

    plt.suptitle(
        f"Per-Head Attention — Step {step_idx}, Query {query_idx} (immediate next action)",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"per_head_step{step_idx:03d}_q{query_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_state_z_attention(attn_weights, step_idx, save_dir):
    """
    Bar chart: how much each query attends to state token vs z token vs image.

    Reveals whether the model uses z at all (expect ~0 given KL collapse).
    """
    attn_avg = attn_weights.mean(dim=0).numpy()  # [50, 302]

    image_attn = attn_avg[:, :300].sum(axis=1)  # total attention on image
    state_attn = attn_avg[:, 300]                # attention on state token
    z_attn = attn_avg[:, 301]                    # attention on z token

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(50)
    width = 0.8

    ax.bar(x, image_attn, width, label="Image tokens (300)", color="#4C72B0")
    ax.bar(x, state_attn, width, bottom=image_attn, label="State token", color="#DD8452")
    ax.bar(x, z_attn, width, bottom=image_attn + state_attn, label="z token", color="#55A868")

    ax.set_xlabel("Action query (timestep)")
    ax.set_ylabel("Total attention weight")
    ax.set_title(f"Attention Distribution: Image vs State vs z — Step {step_idx}")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"token_distribution_step{step_idx:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize ACT cross-attention")
    parser.add_argument("--checkpoint", default="checkpoints/act_final.pt")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 50, 100],
                        help="Episode step indices to visualize")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", default="attention_viz")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model, state_mean, state_std, config = load_model(args.checkpoint, device)

    # Load episode
    ep_dirs = sorted([d for d in os.listdir(args.data_dir) if d.startswith("episode_")])
    ep_dir = os.path.join(args.data_dir, ep_dirs[args.episode])
    with open(os.path.join(ep_dir, "episode.json")) as f:
        steps = json.load(f)["steps"]
    print(f"\nEpisode: {ep_dirs[args.episode]} ({len(steps)} steps)")

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Hook into cross-attention
    capture = AttentionCapture(model)
    capture.attach()

    for step_idx in args.steps:
        if step_idx >= len(steps):
            print(f"  Skipping step {step_idx} (episode has {len(steps)} steps)")
            continue

        print(f"\nStep {step_idx}:")

        # Load image and state
        image_pil = Image.open(os.path.join(ep_dir, steps[step_idx]["frame"])).convert("RGB")
        image_tensor = preprocess_image(image_pil, device)
        state = torch.tensor(step_to_state(steps[step_idx]), device=device)
        state_norm = (state - state_mean) / state_std

        # Forward pass (captures attention weights via hook)
        with torch.no_grad():
            model(image_tensor, state_norm.unsqueeze(0))

        attn = capture.weights[0]  # [n_heads, 50, 302] — remove batch dim

        # Generate all plots
        plot_attention_heatmap(attn, image_pil, step_idx, args.save_dir)
        plot_spatial_attention(attn, image_pil, step_idx, args.save_dir)
        plot_per_head_attention(attn, image_pil, step_idx, args.save_dir)
        plot_state_z_attention(attn, step_idx, args.save_dir)

    capture.detach()
    print(f"\nAll visualizations saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
