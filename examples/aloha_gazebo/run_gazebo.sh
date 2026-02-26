#!/bin/bash
# 快速启动 Gazebo 仿真的脚本

set -e

echo "=========================================="
echo "OpenPI ALOHA Gazebo 仿真启动脚本"
echo "=========================================="

# 检查 ROS2 是否已安装
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠️  警告: 未检测到 ROS2 环境"
    echo "请先 source ROS2 环境:"
    echo "  source /opt/ros/humble/setup.bash  # 或其他 ROS2 版本"
    exit 1
fi

echo "✅ ROS2 环境: $ROS_DISTRO"

# 检查策略服务器是否运行
echo ""
echo "检查策略服务器连接..."
if ! timeout 2 bash -c "echo > /dev/tcp/localhost/8000" 2>/dev/null; then
    echo "⚠️  警告: 策略服务器未运行在 localhost:8000"
    echo ""
    echo "请在另一个终端运行策略服务器:"
    echo "  cd /home/wyz/openpi"
    echo "  uv run scripts/serve_policy.py --env ALOHA_SIM"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ 策略服务器已运行"
fi

# 检查 Gazebo 话题
echo ""
echo "检查 Gazebo 话题..."
if ! ros2 topic list 2>/dev/null | grep -q "image_raw\|joint"; then
    echo "⚠️  警告: 未检测到 Gazebo 话题"
    echo "请确保 Gazebo 已启动并发布了相机和关节话题"
    echo ""
    echo "可以使用以下命令查看话题:"
    echo "  ros2 topic list"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ 检测到 Gazebo 话题"
fi

# 运行策略客户端
echo ""
echo "启动策略客户端..."
cd "$(dirname "$0")/../.."
python3 examples/aloha_gazebo/main.py \
    --host localhost \
    --port 8000 \
    --action-horizon 10






































