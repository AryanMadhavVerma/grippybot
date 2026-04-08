# Evaluation

Offline validation and real robot inference.

## Offline Evaluation
Replays training episodes and compares model predictions to actual human actions.

```bash
grippybot-eval --mode offline --data_dir data/pick_tissue --episode 5
grippybot-eval --mode offline --no_ensemble   # disable temporal ensembling
```

## Robot Evaluation
Runs the trained policy on the real robot (requires Pi with camera + servos).

```bash
# On Pi:
grippybot-eval --mode robot --checkpoint checkpoints/act_final.pt
```

## Our Results (52 demos, 100K training steps)
- Offline: 0.68 degrees mean error, 99.6% gripper accuracy
- Real robot: 7/30 (23%) successful picks
  - Center: 4/10, Left: 2/10, Right: 1/10

## Temporal Ensembling
Enabled by default. Each step queries the policy for 50 future actions. Overlapping predictions are averaged with exponential decay — recent predictions weighted more. Smooths trajectories and reduces jitter. Disable with `--no_ensemble`.
