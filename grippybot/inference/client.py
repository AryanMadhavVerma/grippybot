"""
Inference client — runs on Mac.
Receives frames + state from Pi, runs ACT model, sends predicted actions back.

Usage on Mac:
    grippybot-client --host raspi.local --port 5555
"""

import argparse
import io
import socket
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from grippybot.model.dataset import IMAGENET_MEAN, IMAGENET_STD
from grippybot.model.ensemble import TemporalEnsemble
from grippybot.evaluation.evaluate import load_model
from grippybot.inference.protocol import send_msg, recv_msg, recv_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="raspi.local")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", default="checkpoints/act_final.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_ensemble", action="store_true")
    parser.add_argument("--max_steps", type=int, default=150, help="Stop after N steps")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model, state_mean, state_std, config = load_model(args.checkpoint, device)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Temporal ensemble
    ensemble = None if args.no_ensemble else TemporalEnsemble(config["chunk_size"], config["state_dim"])

    # Connect to Pi
    print(f"Connecting to Pi at {args.host}:{args.port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print("Connected!")

    step = 0
    try:
        while step < args.max_steps:
            t_start = time.time()

            # Receive header (state + jpeg_size)
            header = recv_msg(sock)
            if header is None:
                print("Connection closed by Pi.")
                break

            # Receive JPEG bytes
            jpeg_bytes = recv_exact(sock, header["jpeg_size"])
            if jpeg_bytes is None:
                break

            # Decode image
            image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            image_tensor = image_transform(image).unsqueeze(0).to(device)  # [1, 3, 480, 640]

            # Build state tensor
            state_list = header["state"] + [header["gripper"]]
            state = torch.tensor(state_list, dtype=torch.float32, device=device)

            # Normalize and predict
            state_norm = (state - state_mean) / state_std
            with torch.no_grad():
                pred_norm, _, _ = model(image_tensor, state_norm.unsqueeze(0))
            pred = (pred_norm[0] * state_std + state_mean).cpu().numpy()  # [50, 5]

            # Temporal ensemble
            if ensemble is not None:
                ensemble.add_chunk(pred)
                action = ensemble.get_action()
            else:
                action = pred[0]

            # Send action back to Pi
            action_msg = {
                "base": float(action[0]),
                "shoulder": float(action[1]),
                "elbow": float(action[2]),
                "wrist": float(action[3]),
                "gripper": float(action[4]),
            }
            send_msg(sock, action_msg)

            elapsed = time.time() - t_start
            if step % 5 == 0:
                print(f"step {step:3d}/{args.max_steps} | latency {elapsed*1000:.0f}ms | "
                      f"action: [{', '.join(f'{v:6.1f}' for v in action)}]")

            step += 1

        print(f"\nDone! Completed {step} steps.")

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
