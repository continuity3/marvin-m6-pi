#!/usr/bin/env python3
"""
修复后的版本：将策略输出的 action 发送给真实 ALOHA 机器人

这个脚本展示了如何：
1. 连接到 WebSocket 策略服务器
2. 从真实机器人获取观测（通过 ROS1）
3. 调用策略获取 action
4. 将 action 发送给真实机器人执行

注意：这个版本使用 ROS1（rospy），与 ALOHA 兼容
"""
import argparse
import sys
import time
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入 openpi_client
from openpi_client import websocket_client_policy
from openpi_client import action_chunk_broker
from openpi_client import image_tools

# 导入 ALOHA 真实环境
sys.path.append("/home/wyz/openpi/examples/aloha_real")
from examples.aloha_real import env as _env

# 检查 ROS1 是否可用
try:
    import rospy
    ROS1_AVAILABLE = True
except ImportError:
    ROS1_AVAILABLE = False
    print("⚠️  ROS1 (rospy) 不可用。请确保 ROS1 Noetic 已安装并 source 了环境。")
    print("   或者使用 Docker 运行。")


def main():
    parser = argparse.ArgumentParser(description='将策略 action 发送给真实 ALOHA 机器人')
    parser.add_argument('--host', default='localhost', help='策略服务器地址')
    parser.add_argument('--port', type=int, default=8000, help='策略服务器端口')
    parser.add_argument('--task-prompt', default='take the toast out of the toaster', 
                       help='任务描述')
    parser.add_argument('--action-horizon', type=int, default=25, help='动作块大小')
    parser.add_argument('--max-steps', type=int, default=1000, help='最大步数')
    parser.add_argument('--control-hz', type=float, default=10.0, help='控制频率 (Hz)')
    args = parser.parse_args()

    if not ROS1_AVAILABLE:
        print("❌ 无法运行：需要 ROS1 环境")
        sys.exit(1)

    # ============================================================
    # 步骤 1: 连接到 WebSocket 策略服务器
    # ============================================================
    print("📡 正在连接到策略服务器...")
    try:
        ws_policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        
        # 获取服务器元数据
        metadata = ws_policy.get_server_metadata()
        print(f"✅ 已连接到策略服务器")
        print(f"   - 服务器元数据: {metadata}")
        
        # 使用 ActionChunkBroker 包装策略
        policy = action_chunk_broker.ActionChunkBroker(
            policy=ws_policy,
            action_horizon=args.action_horizon,
        )
        print(f"✅ 策略包装完成 (action_horizon={args.action_horizon})")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("   请确保策略服务器已启动:")
        print(f"   uv run scripts/serve_policy.py --env ALOHA --default-prompt='{args.task_prompt}'")
        sys.exit(1)

    # ============================================================
    # 步骤 2: 初始化真实机器人环境
    # ============================================================
    print("🤖 正在初始化真实机器人环境...")
    try:
        # 从服务器元数据获取重置位置（如果有）
        reset_position = metadata.get("reset_pose")
        
        # 创建 ALOHA 真实环境
        robot_env = _env.AlohaRealEnvironment(
            reset_position=reset_position,
            render_height=224,
            render_width=224,
        )
        
        print("✅ 机器人环境初始化完成")
        
    except Exception as e:
        print(f"❌ 机器人环境初始化失败: {e}")
        print("   请确保:")
        print("   1. ROS 节点已启动: roslaunch aloha ros_nodes.launch")
        print("   2. 机器人硬件已连接")
        print("   3. 相机已正确配置")
        sys.exit(1)

    # ============================================================
    # 步骤 3: 重置机器人到初始位置
    # ============================================================
    print("🔄 正在重置机器人...")
    try:
        robot_env.reset()
        print("✅ 机器人已重置到初始位置")
        
        # 等待机器人稳定
        time.sleep(2.0)
        print("✅ 机器人已稳定")
        
    except Exception as e:
        print(f"❌ 重置失败: {e}")
        sys.exit(1)

    # ============================================================
    # 步骤 4: 主控制循环
    # ============================================================
    print(f"\n🚀 开始控制循环 (任务: '{args.task_prompt}')")
    print("⚠️  按 Ctrl+C 停止机器人\n")

    step_count = 0
    target_step_time = 1.0 / args.control_hz

    try:
        while step_count < args.max_steps:
            step_start_time = time.time()
            
            # ===== 4.1 获取当前观测 =====
            obs = robot_env.get_observation()
            
            # 检查必需的相机图像
            images = obs["images"]
            if "cam_high" not in images:
                print(f"⚠️  Step {step_count}: 缺少相机图像，跳过")
                time.sleep(0.1)
                continue
            
            # ===== 4.2 构建策略输入 =====
            # ALOHA 策略期望的格式：state + images + prompt
            policy_input = {
                "state": obs["state"].astype(np.float32),  # 14维状态
                "images": images,  # 字典，包含 cam_high, cam_left_wrist, cam_right_wrist
                "prompt": args.task_prompt,
            }
            
            # ===== 4.3 调用策略获取动作 =====
            try:
                inference_start = time.perf_counter()
                policy_output = policy.infer(policy_input)
                inference_time = (time.perf_counter() - inference_start) * 1000
                
                # 获取动作（已经是单个动作，不是动作块）
                action = policy_output["actions"]  # shape: (14,)
                
                if step_count % 10 == 0:
                    print(f"Step {step_count}: 推理时间 {inference_time:.2f} ms, "
                          f"动作范围 [{action.min():.3f}, {action.max():.3f}]")
                
            except Exception as e:
                print(f"❌ Step {step_count}: 策略推理失败: {e}")
                break
            
            # ===== 4.4 检查动作维度 =====
            if len(action) != 14:
                print(f"⚠️  Step {step_count}: 动作维度错误，期望 14，得到 {len(action)}")
                break
            
            # ===== 4.5 将动作发送给真实机器人 =====
            try:
                robot_env.apply_action({"actions": action.tolist()})
                
            except Exception as e:
                print(f"❌ Step {step_count}: 执行动作失败: {e}")
                break
            
            step_count += 1
            
            # ===== 4.6 控制频率 =====
            step_time = time.time() - step_start_time
            if step_time < target_step_time:
                time.sleep(target_step_time - step_time)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n✅ 控制循环结束，总步数: {step_count}")
        print("⚠️  机器人已停止，请手动检查机器人状态")


if __name__ == '__main__':
    main()






































