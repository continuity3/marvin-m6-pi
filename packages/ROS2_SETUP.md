# ROS2 模式运行指南

## 前提条件

1. **ROS2 已安装**（Humble 或 Foxy）
2. **策略服务器正在运行**
3. **机器人 ROS2 节点正在运行**

## 快速启动（推荐）

### 方法 1：使用启动脚本（最简单）

```bash
cd /home/wyz/openpi
bash packages/start_pick_blue_bottle_subscriber.sh ros2
```

这个脚本会：
- 自动检测 ROS2 环境
- 自动 source ROS2 setup.bash
- 尝试使用兼容的 Python 环境
- 如果依赖缺失，会给出提示

### 方法 2：手动启动

#### 步骤 1：启动策略服务器（终端 1）

```bash
cd /home/wyz/openpi
uv run scripts/serve_policy_pick_blue_bottle.py --port 8000
```

#### 步骤 2：启动 ROS2 客户端（终端 2）

```bash
cd /home/wyz/openpi

# Source ROS2 环境
source /opt/ros/humble/setup.bash  # 或 foxy，根据你的 ROS2 版本

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"

# 运行脚本（使用 packages/.venv 如果可用）
if [ -f "packages/.venv/bin/python3" ]; then
    packages/.venv/bin/python3 packages/pose_subscriber_pick_blue_bottle.py \
        --host localhost \
        --port 8000 \
        --use-realsense \
        --publish-actions
else
    # 或者使用 uv run（但需要确保 ROS2 可用）
    uv run packages/pose_subscriber_pick_blue_bottle.py \
        --host localhost \
        --port 8000 \
        --use-realsense \
        --publish-actions
fi
```

## ROS2 话题配置

脚本默认使用以下话题：

- **订阅话题**：
  - `/pose` - 位姿信息（PoseStamped）
  - `/joint_states` - 关节状态（JointState，需要至少 14 个关节）
  - `/gripper/feedback_R` - 右夹爪反馈（Float64）

- **发布话题**：
  - `/libero/actions` - 动作指令（Float64MultiArray，7 维：6 个关节 + 1 个夹爪）

### 自定义话题

```bash
packages/pose_subscriber_pick_blue_bottle.py \
    --topic /your/pose/topic \
    --action-topic /your/action/topic \
    --joint-states-topic /your/joint/states/topic \
    --gripper-topic /your/gripper/topic
```

## 动作格式

发布到 `/libero/actions` 的动作格式：
- **类型**：`std_msgs/Float64MultiArray`
- **维度**：7 维
  - 前 6 维：关节位置/速度（根据你的机器人配置）
  - 第 7 维（索引 6）：夹爪值
    - `0.0` = 打开
    - `1.0` = 闭合
    - 脚本会自动根据反归一化后的值进行阈值判断

## 状态输入

脚本从 ROS2 话题读取：
- **关节状态**：从 `/joint_states` 的 `position[7:14]`（右臂的 7 个关节）
- **夹爪状态**：从 `/gripper/feedback_R`
- **组合成 8 维状态**：7 个关节位置 + 1 个夹爪值

## 检查 ROS2 连接

在运行脚本前，确保：

1. **ROS2 环境正确**：
```bash
source /opt/ros/humble/setup.bash  # 或你的 ROS2 版本
ros2 topic list  # 应该能看到话题列表
```

2. **机器人节点正在运行**：
```bash
ros2 node list  # 应该能看到你的机器人节点
ros2 topic echo /joint_states  # 应该能看到关节状态数据
```

3. **策略服务器正在运行**：
```bash
# 在另一个终端
cd /home/wyz/openpi
uv run scripts/serve_policy_pick_blue_bottle.py
```

## 调试

### 查看发布的动作

```bash
ros2 topic echo /libero/actions
```

### 查看订阅的状态

```bash
ros2 topic echo /joint_states
ros2 topic echo /gripper/feedback_R
```

### 检查节点是否运行

```bash
ros2 node list | grep pose_subscriber
```

## 常见问题

### 问题：rclpy not available

**原因**：ROS2 环境未正确 source

**解决**：
```bash
source /opt/ros/humble/setup.bash  # 根据你的 ROS2 版本
```

### 问题：JointState has X positions, expected at least 14

**原因**：关节状态话题的数据维度不足

**解决**：确保 `/joint_states` 话题包含至少 14 个关节位置（脚本使用索引 7-13 作为右臂关节）

### 问题：No actions published

**检查**：
1. 策略服务器是否正在运行
2. 是否使用了 `--publish-actions` 参数（默认已启用）
3. 查看脚本日志，确认是否收到动作

### 问题：依赖缺失但想用 ROS2 模式

**解决**：在 ROS2 的 Python 环境中安装依赖：
```bash
# 激活 ROS2 Python 环境（如果有）
source /opt/ros/humble/setup.bash

# 安装依赖到当前环境
cd /home/wyz/openpi
pip3 install -e .  # 或使用你 ROS2 环境的 pip
```

## 完整示例

```bash
# 终端 1：策略服务器
cd /home/wyz/openpi
uv run scripts/serve_policy_pick_blue_bottle.py --port 8000

# 终端 2：ROS2 客户端
cd /home/wyz/openpi
source /opt/ros/humble/setup.bash
bash packages/start_pick_blue_bottle_subscriber.sh ros2
```












