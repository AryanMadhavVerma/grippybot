# Contributing

grippybot is an open test bed for learning robotics + behavior cloning. Fork it, print your own arm, train your own policy, and push the boundaries.

## Getting Started

```bash
git clone https://github.com/AryanMadhavVerma/grippybot.git
cd grippybot
pip install -e ".[dev]"
pytest tests/ -v
```

## Open Problems

### Hardware
- **New board support**: Implement ServoDriver/Camera for Arduino, Jetson Nano, etc. See [hardware README](grippybot/hardware/README.md) for the interface contract.
- **USB camera**: OpenCV-based Camera class as alternative to picamera2.
- **Better servos**: Digital servos with position feedback, or stepper motors for higher precision.

### Teleop & Data Collection
- **Gamepad/joystick support**: Smoother than keyboard, faster data collection.
- **Web-based teleop UI**: Remove the SSH + curses requirement.
- **Leader-follower teleop**: Potentiometer-based (read leader arm position → mirror on follower).

### Policy & Training
- **More demos**: Current 23% success rate is limited by 52 demos with insufficient position diversity. 100+ diverse demos should significantly improve performance.
- **Alternative policies**: Diffusion policy, SmolVLA, simple MLP baseline — compare against ACT.
- **Swappable vision backbones**: Replace ResNet18 with MobileNet, EfficientNet, or a ViT.
- **Multi-task training**: Pick different objects with language conditioning.

### Inference & Evaluation
- **GPU inference benchmarking**: MPS adds 100-150ms latency. What's the real-time FPS on CUDA?
- **Automatic episode termination**: The policy doesn't know when to stop — needs a "done" predictor or confidence threshold.
- **Quantitative evaluation**: Automated success detection instead of manual counting.

### New Tasks
Collect demos for any task (stack blocks, sort by color, open bottles) and train with the same pipeline. Document your task + results.

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: model forward pass, dataset loading, temporal ensemble, TCP protocol, loss computation, checkpointing. They run in CI on every PR.

## Code Style
- Keep it practical — no over-abstraction
- Match existing patterns in the codebase
- Add tests for new functionality
