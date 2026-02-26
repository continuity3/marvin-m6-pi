# Pick Blue Bottle 微调数据的 State 和 Action 提取说明

## 数据流程概览

```
ROS2 Bag 文件 (rosbag2)
    ↓
转换为 HDF5 格式
    ↓
通过 convert_pick_blue_bottle_hdf5_to_lerobot.py 转换为 LeRobot 格式
    ↓
用于训练/微调
```

## 1. 原始数据来源（ROS2 Bag）

原始数据存储在 `rosbag2_*.db3` 文件中，包含以下 ROS2 topics：

- **`/joint_states`** (sensor_msgs/msg/JointState)
  - 包含 14 个关节的位置（position）和速度（velocity）
  - 这 14 个关节可能包括双臂机器人的左右臂关节

- **`/camera/camera/color/image_raw`** (sensor_msgs/msg/Image)
  - 相机图像数据

## 2. HDF5 文件结构

rosbag2 文件被转换为 HDF5 格式后，结构如下：

```
HDF5 文件
├── time: (T,) 时间戳数组
├── topics/
│   ├── _joint_states/
│   │   ├── position: (T, 14) 关节位置数组
│   │   └── velocity: (T, 14) 关节速度数组
│   └── _camera_camera_color_image_raw/
│       ├── data: (T, 921600) 图像数据（扁平化）
│       └── data_length: (T,) 每个图像的实际长度
└── valid/ (可选)
    ├── _joint_states: (T,) 有效性标记
    └── _camera_camera_color_image_raw: (T,) 有效性标记
```

## 3. State 的提取（8维）

在 `convert_pick_blue_bottle_hdf5_to_lerobot.py` 的 `load_pick_blue_bottle_hdf5()` 函数中：

**代码位置**：`examples/libero/convert_pick_blue_bottle_hdf5_to_lerobot.py` 第 205-213 行

```python
# 提取左臂关节（前7个关节）用于 LIBERO
left_positions = positions[:, :7]  # (T, 7)

# 组合状态（关节位置 + 夹爪，LIBERO 需要8维）
states = np.concatenate([left_positions, np.zeros((len(left_positions), 1))], axis=1)  # (T, 8)
```

**State 组成**：
- **前 7 维**：左臂的 7 个关节位置（从 HDF5 的 `topics/_joint_states/position` 中提取前 7 列）
- **第 8 维**：夹爪状态（当前实现中设为 0，因为原始数据中没有夹爪信息）

**维度**：`(T, 8)`，其中 T 是时间步数

## 4. Action 的提取（7维）

在同一个函数中：

**代码位置**：`examples/libero/convert_pick_blue_bottle_hdf5_to_lerobot.py` 第 205-210 行

```python
# 提取左臂关节（前7个关节）用于 LIBERO
left_positions = positions[:, :7]  # (T, 7)
left_velocities = velocities[:, :7]  # (T, 7)

# 计算动作（使用速度）
actions = left_velocities  # (T, 7)
```

**Action 组成**：
- **7 维**：左臂的 7 个关节速度（从 HDF5 的 `topics/_joint_states/velocity` 中提取前 7 列）

**维度**：`(T, 7)`，其中 T 是时间步数

**注意**：Action 直接使用关节速度，而不是位置增量（delta position）。这是 LIBERO 数据格式的常见做法。

## 5. 数据转换的完整流程

### 步骤 1：从 HDF5 读取原始数据
```python
joint_states = f["topics/_joint_states"]
positions = joint_states["position"][:]  # (T, 14)
velocities = joint_states["velocity"][:]  # (T, 14)
```

### 步骤 2：提取左臂数据
```python
left_positions = positions[:, :7]   # (T, 7)
left_velocities = velocities[:, :7] # (T, 7)
```

### 步骤 3：构建 State（8维）
```python
states = np.concatenate([left_positions, np.zeros((len(left_positions), 1))], axis=1)  # (T, 8)
```

### 步骤 4：构建 Action（7维）
```python
actions = left_velocities  # (T, 7)
```

### 步骤 5：转换为 LeRobot 格式
```python
steps.append({
    "image": image,                    # 调整后的图像 (256, 256, 3)
    "wrist_image": wrist_image,        # 手腕相机图像（这里使用主相机）
    "state": states[i].astype(np.float32),      # (8,) 状态向量
    "action": actions[i].astype(np.float32),    # (7,) 动作向量
    "task": task_description,
})
```

## 6. 关键代码文件

- **转换脚本**：`examples/libero/convert_pick_blue_bottle_hdf5_to_lerobot.py`
  - `load_pick_blue_bottle_hdf5()`: 从 HDF5 文件加载数据并提取 state 和 action

## 7. 数据特点

1. **State 维度**：8 维（7 个关节位置 + 1 个夹爪状态）
2. **Action 维度**：7 维（7 个关节速度）
3. **夹爪处理**：当前实现中，state 的夹爪维度被设为 0，因为原始 rosbag2 数据中没有夹爪信息
4. **动作类型**：使用关节速度（velocity）作为动作，而不是位置增量

## 8. 与标准 LIBERO 数据的对比

标准 LIBERO 数据（来自官方数据集）：
- **State**：8 维 = [ee_pos (3), ee_ori (3), gripper_states (1), joint_states (1)]
- **Action**：7 维 = [ee_pos_delta (3), ee_ori_delta (3), gripper_delta (1)]

你的 pick_blue_bottle 数据：
- **State**：8 维 = [joint_positions (7), gripper (1，设为0)]
- **Action**：7 维 = [joint_velocities (7)]

**主要区别**：
- 标准 LIBERO 使用末端执行器（end-effector）空间（位置和姿态）
- 你的数据使用关节空间（joint space）
- 标准 LIBERO 的 action 是位置增量，你的数据是速度

## 9. 注意事项

1. **夹爪状态**：当前实现中，state 的第 8 维（夹爪）被设为 0。如果需要真实的夹爪状态，需要从 rosbag2 的其他 topic（如 `/gripper/feedback_L`）中提取。

2. **关节选择**：代码假设前 7 个关节是左臂关节。如果你的机器人配置不同，可能需要调整索引。

3. **数据有效性**：转换脚本支持使用 `valid` 标记来过滤无效数据点。

4. **图像处理**：图像从 rosbag2 的原始格式解码后，会被调整到 (256, 256, 3) 大小。

















