# Marvin M6-Pi: HDF5 to LeRobot Data Conversion

This repository provides tools for converting robot demonstration data from HDF5 format to LeRobot format, enabling training of vision-language-action (VLA) models using the [OpenPi](https://github.com/Physical-Intelligence/openpi) framework.

## Overview

The conversion pipeline processes HDF5 files containing robot demonstration data (joint states, end-effector poses, images, and gripper states) and converts them into the LeRobot dataset format, which is compatible with OpenPi's training infrastructure.

## HDF5 Data Format Requirements

Your HDF5 files should contain the following structure:

### Required Topics

1. **`topics/_info_eef_right`** (Required)
   - `position`: `(T, 3)` - End-effector position in 3D space
   - `orientation`: `(T, 4)` - End-effector orientation as quaternion `[x, y, z, w]`

2. **`topics/_joint_states`** (Required)
   - `position`: `(T, 14)` - Joint positions (first 7 dims = left arm, last 7 dims = right arm)
   - `velocity`: `(T, 14)` - Joint velocities (optional, not used in conversion)

3. **`topics/_gripper_feedback_R`** (Optional but recommended)
   - `data`: `(T, 5)` or `(T,)` - Right gripper feedback data (first column is used as gripper state)

4. **Image Data** (One of the following formats):
   - **Format 1**: `topics/_camera_image`
     - `data`: `(T, H, W, 3)` - Pre-decoded RGB images (uint8)
   - **Format 2**: `topics/_camera_camera_color_image_raw`
     - `data`: `(T, 921600)` - Flattened image data (JPEG encoded or raw)
     - `data_length`: `(T,)` - Actual length of each image

5. **Optional: Validity Flags**
   - `valid/_joint_states`: `(T,)` - Boolean array marking valid joint state timesteps
   - `valid/_camera_image` or `valid/_camera_camera_color_image_raw`: `(T,)` - Boolean array marking valid image timesteps

### Example HDF5 Structure

```
data.h5
├── time: (T,) - Timestamps
└── topics/
    ├── _info_eef_right/
    │   ├── position: (T, 3)
    │   └── orientation: (T, 4)
    ├── _joint_states/
    │   ├── position: (T, 14)
    │   └── velocity: (T, 14)
    ├── _gripper_feedback_R/
    │   └── data: (T, 5) or (T,)
    └── _camera_image/ or _camera_camera_color_image_raw/
        ├── data: (T, H, W, 3) or (T, 921600)
        └── data_length: (T,) [if using raw format]
```

## Output Format (LeRobot)

The conversion script produces a LeRobot dataset with the following features:

- **`image`**: `(256, 256, 3)` uint8 - Resized RGB camera image
- **`wrist_image`**: `(256, 256, 3)` uint8 - Wrist camera image (uses main camera if not available)
- **`state`**: `(8,)` float32 - Robot state vector:
  - `[0:3]`: End-effector position (3D)
  - `[3:6]`: End-effector orientation as axis-angle (3D)
  - `[6]`: Gripper state value
  - `[7]`: Negative gripper state value
- **`actions`**: `(7,)` float32 - Action vector:
  - `[0:3]`: End-effector position delta (3D displacement)
  - `[3:6]`: End-effector orientation delta as axis-angle (3D rotation)
  - `[6]`: Gripper action (0 = open, 1 = close)
- **`task`**: string - Task description

## Installation

1. Clone this repository and ensure you have the OpenPi dependencies installed:

```bash
git clone --recurse-submodules https://github.com/continuity3/marvin-m6-pi.git
cd marvin-m6-pi

# Install dependencies using uv (see OpenPi README for details)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

2. Ensure you have the required Python packages:
   - `h5py`
   - `numpy`
   - `PIL` (Pillow)
   - `lerobot`
   - `scipy`
   - `tyro`

## Usage

### Basic Usage

Convert a single HDF5 file:

```bash
uv run examples/libero/convert_hdf5_to_lerobot.py \
    --hdf5_path /path/to/your/data.h5 \
    --output_repo_name your_hf_username/dataset_name
```

Convert all HDF5 files in a directory (recursively):

```bash
uv run examples/libero/convert_hdf5_to_lerobot.py \
    --hdf5_path /path/to/your/data_directory \
    --output_repo_name your_hf_username/dataset_name
```

### Advanced Options

```bash
uv run examples/libero/convert_hdf5_to_lerobot.py \
    --hdf5_path /path/to/data.h5 \
    --output_repo_name your_hf_username/dataset_name \
    --task_description "Your task description" \
    --fps 30.0 \
    --ignore_valid \
    --push_to_hub
```

**Parameters:**
- `--hdf5_path`: Path to HDF5 file or directory containing HDF5 files
- `--output_repo_name`: Output dataset name (format: `username/dataset_name`)
- `--task_description`: Task description string (default: "Robot manipulation task")
- `--fps`: Output dataset frame rate (default: 30.0 fps)
- `--ignore_valid`: Ignore validity flags and use all data
- `--push_to_hub`: Push converted dataset to Hugging Face Hub

### Example

```bash
# Convert a single demonstration file
uv run examples/libero/convert_hdf5_to_lerobot.py \
    --hdf5_path ./data/demo_001.h5 \
    --output_repo_name myusername/my_robot_dataset \
    --task_description "Pick and place task" \
    --fps 30.0

# Convert all files in a directory
uv run examples/libero/convert_hdf5_to_lerobot.py \
    --hdf5_path ./data/demonstrations/ \
    --output_repo_name myusername/my_robot_dataset \
    --task_description "Pick and place task"
```

## Data Processing Details

### State Computation

The state vector is computed as:
- **EEF Position**: Directly from `_info_eef_right/position`
- **EEF Orientation**: Converted from quaternion `[x, y, z, w]` to axis-angle representation
- **Gripper State**: Extracted from the first column of `_gripper_feedback_R/data` (or zeros if not available)

### Action Computation

Actions are computed geometrically:
- **EEF Action**: Relative displacement and rotation from timestep `t` to `t+1`
  - Position delta: `ee_pos[t+1] - ee_pos[t]`
  - Orientation delta: Relative rotation in axis-angle form
- **Gripper Action**: Binary action based on next timestep's gripper value
  - `0.0` if next gripper value ≤ 0.4 (open)
  - `1.0` if next gripper value > 0.4 (close)
  - Last timestep always has action `0.0` (no change)

### Image Processing

- Images are automatically decoded (JPEG or raw format)
- Resized to `(256, 256, 3)` using bicubic interpolation
- Converted to uint8 format
- If wrist camera images are not available, main camera images are used for both `image` and `wrist_image`

### NaN Handling

The script automatically handles NaN values:
- First NaN: Uses the next non-NaN value
- Last NaN: Uses the previous non-NaN value
- Middle NaN: Uses the previous non-NaN value (forward fill)

### Validity Filtering

If validity flags are present and `--ignore_valid` is not set:
- Only timesteps where both joint states and images are valid are used
- If only one validity flag exists, that flag is used

## Output Location

The converted dataset is saved to:
```
~/.cache/huggingface/lerobot/datasets/{output_repo_name}/
```

You can also access it programmatically:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("your_hf_username/dataset_name")
```

## Troubleshooting

### Common Issues

1. **Missing required topics**
   - Error: `KeyError: "找不到 _info_eef_right topic"`
   - Solution: Ensure your HDF5 file contains the required topics listed above

2. **Image decoding failures**
   - The script will use zero images as placeholders if decoding fails
   - Check that your image data is in the correct format (JPEG or raw RGB)

3. **NaN values in data**
   - The script automatically handles NaN values, but excessive NaNs may indicate data quality issues
   - Check your source data for sensor failures or recording issues

4. **Data length mismatches**
   - Ensure all topics have the same number of timesteps `T`
   - The script will use the minimum length across all topics

### Verifying Your HDF5 File

You can inspect your HDF5 file structure using:

```python
import h5py

with h5py.File("your_data.h5", "r") as f:
    print("Keys:", list(f.keys()))
    if "topics" in f:
        print("Topics:", list(f["topics"].keys()))
        for topic in f["topics"].keys():
            print(f"\n{topic}:")
            for key in f[f"topics/{topic}"].keys():
                data = f[f"topics/{topic}/{key}"]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
```

## Next Steps

After converting your data to LeRobot format:

1. **Train a model**: Use the converted dataset with OpenPi's training pipeline
2. **Fine-tune**: Fine-tune a base model (e.g., π₀.₅) on your dataset
3. **Evaluate**: Test your trained model on your robot platform

See the [OpenPi documentation](https://github.com/Physical-Intelligence/openpi) for training and inference instructions.

## License

This repository follows the same license as the OpenPi project. Please refer to the OpenPi repository for license details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- [OpenPi Repository](https://github.com/Physical-Intelligence/openpi)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [LIBERO Dataset](https://libero-project.github.io/datasets)

