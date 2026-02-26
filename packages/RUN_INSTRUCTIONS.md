# 如何运行 pose_subscriber_pick_blue_bottle.py

## 方法 1：使用 uv run（推荐，最简单）

这是最简单的方法，`uv run` 会自动处理所有依赖：

```bash
cd /home/wyz/openpi

# 测试模式（不需要 ROS2）
uv run packages/pose_subscriber_pick_blue_bottle.py --test-mode

# 或者使用启动脚本
bash packages/start_pick_blue_bottle_subscriber.sh test
```

## 方法 2：使用启动脚本

```bash
cd /home/wyz/openpi

# 测试模式
bash packages/start_pick_blue_bottle_subscriber.sh test

# ROS2 模式（需要 ROS2 环境）
bash packages/start_pick_blue_bottle_subscriber.sh ros2
```

## 方法 3：手动安装依赖后运行

如果你想在当前的虚拟环境中运行：

```bash
cd /home/wyz/openpi

# 安装所有依赖
uv pip install -e .

# 然后运行
python3 packages/pose_subscriber_pick_blue_bottle.py --test-mode
```

## 完整运行流程（两个终端）

### 终端 1：启动策略服务器

```bash
cd /home/wyz/openpi
uv run scripts/serve_policy_pick_blue_bottle.py --port 8000
```

### 终端 2：启动客户端

```bash
cd /home/wyz/openpi

# 测试模式
uv run packages/pose_subscriber_pick_blue_bottle.py --test-mode --host localhost --port 8000

# 或者使用启动脚本
bash packages/start_pick_blue_bottle_subscriber.sh test
```

## 常见问题

### 问题：ModuleNotFoundError: No module named 'flax'

**解决方案**：使用 `uv run` 运行脚本，它会自动处理依赖。

```bash
uv run packages/pose_subscriber_pick_blue_bottle.py --test-mode
```

### 问题：连接服务器失败

**解决方案**：确保策略服务器正在运行：

```bash
# 在另一个终端运行
uv run scripts/serve_policy_pick_blue_bottle.py --port 8000
```

## 参数说明

- `--test-mode`: 测试模式，不需要 ROS2
- `--host`: 策略服务器地址（默认：localhost）
- `--port`: 策略服务器端口（默认：8000）
- `--use-realsense`: 使用 RealSense 相机
- `--show-camera`: 显示相机画面
- `--record`: 记录数据到指定目录












