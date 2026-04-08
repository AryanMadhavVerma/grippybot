# Model

ACT (Action Chunking with Transformers) implementation from scratch in pure PyTorch.

## Architecture (18.7M params with d_model=256)

```
Image [B, 3, 480, 640]
  │
  ▼
VisionBackbone (ResNet18 → 1x1 conv → 300 tokens + 2D sinusoidal pos embeddings)
  │
  ▼
visual_tokens [B, 300, 256]
  │
  ├── + state_token [B, 1, 256]
  ├── + z_token [B, 1, 256]    ← from CVAE (training) or zeros (inference)
  │
  ▼
ObservationFuser (4-layer transformer encoder)
  │
  ▼
fused [B, 302, 256]
  │
  ▼
ActionDecoder (50 learned queries × 1-layer transformer decoder)
  │
  ▼
predicted_actions [B, 50, 5]   ← 50 future timesteps × 5 joints
```

## Files
- `act.py` — Full architecture: VisionBackbone, CVAEEncoder, ObservationFuser, ActionDecoder, ACTPolicy
- `dataset.py` — PyTorch Dataset loading from episode JSON + JPEGs
- `ensemble.py` — Temporal ensemble for smooth inference

## Key Design Choices
- **d_model=256** (not 512): smaller dataset (52 demos), simpler task (5-DOF vs 14-DOF)
- **chunk_size=50**: ~8 seconds of future actions at 6Hz
- **1 decoder layer**: sufficient for single-task pick-and-place
- **No Accelerate**: explicit `model.to(device)` for guaranteed GPU placement
