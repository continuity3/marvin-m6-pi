# Real-time Inference for LIBERO Robot

This script performs real-time inference on a physical LIBERO robot using the official OpenPI model.

## Features

- ✅ Real-time inference with RealSense camera
- ✅ ROS2 integration for joint states and action publishing
- ✅ Action and observation recording
- ✅ Input/output normalization statistics logging
- ✅ Follows official OpenPI example structure

## Prerequisites

1. **Policy Server**: The policy server must be running (see below)
2. **RealSense Camera**: A RealSense camera connected
3. **ROS2**: ROS2 environment set up (if not using `--test-mode`)

## Usage

### 1. Start the Policy Server

**Terminal 1:**
```bash
# Using official checkpoint
PYTHONPATH="" uv run scripts/serve_policy.py \
    --port 8000 \
    policy:local-checkpoint \
    --policy.path="checkpoints/pi05_libero_pytorch.pt" \
    --policy.config=pi05_libero

# Or using default checkpoint
uv run scripts/serve_policy.py --env LIBERO
```

### 2. Run Real-time Inference

**Terminal 2:**
```bash
# Source ROS2 (if not in test mode)
source /opt/ros/humble/setup.bash  # or iron

# Run inference
python3 examples/libero/realtime_inference.py \
    --prompt "Pick up the blue square and place it in the blue tray." \
    --record-dir "data/realtime_inference" \
    --inference-rate 2.0 \
    --show-camera

# Test mode (without ROS2)
python3 examples/libero/realtime_inference.py \
    --test-mode \
    --prompt "Pick up the blue square and place it in the blue tray." \
    --record-dir "data/realtime_inference"
```

## Arguments

### Model Server Parameters
- `--host`: Policy server host (default: `localhost`)
- `--port`: Policy server port (default: `8000`)

### Task Parameters
- `--prompt`: Task instruction/prompt (default: `"Pick up the blue square and place it in the blue tray."`)

### ROS2 Parameters
- `--joint-states-topic`: ROS2 topic for joint states (default: `/joint_states`)
- `--action-topic`: ROS2 topic for publishing actions (default: `/libero/actions`)
- `--publish-actions`: Whether to publish actions to ROS2 (default: `True`)

### Recording Parameters
- `--record-dir`: Directory to save recorded data (default: `data/realtime_inference`)
- `--save-images`: Save images during recording (default: `True`)
- `--save-normalization`: Save normalization statistics (default: `True`)

### Camera Parameters
- `--use-realsense`: Use RealSense camera (default: `True`)
- `--camera-serial`: RealSense camera serial number (auto-detect if not specified)
- `--show-camera`: Show camera feed in a window (default: `False`)

### Inference Parameters
- `--inference-rate`: Inference frequency in Hz (default: `2.0`)
- `--test-mode`: Test mode without ROS2 (default: `False`)

## Recorded Data

The script saves the following data to `--record-dir`:

1. **`actions.npy`**: Normalized actions (shape: `(num_steps, action_dim)`)
2. **`actions_raw.npy`**: Raw (unnormalized) actions (shape: `(num_steps, action_dim)`)
3. **`states.npy`**: Joint states (shape: `(num_steps, state_dim)`)
4. **`prompts.txt`**: Task prompts for each step
5. **`normalization_stats.npy`**: Normalization statistics (mean, std, min, max) for actions and states
6. **`image_*.npy`**: Individual images (if `--save-images` is enabled)

## Analysis

After recording, you can analyze the data:

```bash
# Analyze action velocity distribution
python3 monitor_action_velocity.py \
    --actions-file data/realtime_inference/actions_raw.npy \
    --output data/realtime_inference/action_velocity_histogram.png

# Compare with training data
python3 monitor_action_velocity.py \
    --actions-file data/realtime_inference/actions_raw.npy \
    --comparison-data path/to/training_velocities.npy \
    --output data/realtime_inference/action_velocity_comparison.png
```

## Example Workflow

```bash
# Terminal 1: Start server
PYTHONPATH="" uv run scripts/serve_policy.py \
    --port 8000 \
    policy:local-checkpoint \
    --policy.path="checkpoints/pi05_libero_pytorch.pt" \
    --policy.config=pi05_libero

# Terminal 2: Run inference
source /opt/ros/humble/setup.bash
python3 examples/libero/realtime_inference.py \
    --prompt "Pick up the blue square and place it in the blue tray." \
    --record-dir "data/realtime_inference_run1" \
    --inference-rate 2.0 \
    --show-camera

# After inference, analyze results
python3 monitor_action_velocity.py \
    --actions-file data/realtime_inference_run1/actions_raw.npy \
    --output data/realtime_inference_run1/action_velocity_histogram.png
```

## Notes

- The script follows the same structure as `examples/libero/main.py` (the official simulation example)
- Actions are automatically unnormalized by the policy server before being returned
- States should match the format expected by the policy (8-dim for LIBERO: 7 joints + 1 gripper)
- Camera images are automatically preprocessed to match training format (224x224, uint8)






















