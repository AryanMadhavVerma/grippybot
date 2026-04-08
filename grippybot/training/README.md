# Training

Pure PyTorch training loop for ACT. No LeRobot, no HuggingFace Accelerate.

## Quick Start
```bash
# On a machine with GPU:
grippybot-train --data_dir data/pick_tissue --device cuda

# Short test run:
grippybot-train --steps 1000 --device cpu
```

## Setup (Vast.ai)
1. Rent a GPU instance (RTX 3090/4080 recommended, ~$0.20/hr)
2. SCP your data: `scp -r data/ root@<ip>:~/grippybot/`
3. Install: `pip install -e .`
4. Train: `grippybot-train --device cuda --steps 100000`
5. SCP checkpoint back: `scp root@<ip>:~/grippybot/checkpoints/act_final.pt checkpoints/`

## Hyperparameters
| Param | Value | Notes |
|-------|-------|-------|
| d_model | 256 | Transformer hidden dim |
| chunk_size | 50 | Future action steps predicted |
| batch_size | 8 | |
| learning_rate | 1e-5 | AdamW |
| KL weight (beta) | 10.0 | CVAE information bottleneck |
| grad_clip | 10.0 | Max gradient norm |
| steps | 100K | ~2.7h on RTX 4080S |

## Expected Loss Curve
- Step 0: ~0.78
- Step 10K: ~0.10
- Step 50K: ~0.06
- Step 100K: ~0.05

Loss = L1 (reconstruction) + 10 * KL (regularization). KL will likely collapse to ~0 for single-task data — that's expected.
