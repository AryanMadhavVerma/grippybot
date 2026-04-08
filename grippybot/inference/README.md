# Inference

Distributed inference: Pi captures frames and executes actions, Mac runs the model.

This split is needed because the Pi can't run the ACT model in real-time (no GPU), but it has the camera and servos. The Mac has the compute (MPS/CUDA) but no hardware.

## Setup

### On Pi:
```bash
grippybot-server --port 5555
```

### On Mac:
```bash
grippybot-client --host raspi.local --port 5555 --checkpoint checkpoints/act_final.pt
```

## Protocol
TCP socket with length-prefixed JSON messages:
1. Pi sends: JSON header (joint state + JPEG size) + raw JPEG bytes
2. Mac sends back: JSON with predicted joint angles + gripper

## Latency
- ~200-300ms round trip at ~5Hz over local WiFi
- MPS inference adds ~100-150ms (open contribution: benchmark on CUDA)

## Options
- `--max_steps 150` — stop after N steps (policy doesn't know when to stop)
- `--no_ensemble` — disable temporal ensembling
