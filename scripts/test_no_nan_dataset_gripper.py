#!/usr/bin/env python3
"""
测试脚本：从去掉NaN后的LeRobot数据集中提取state的gripper值并绘制折线图

使用方法:
    python scripts/test_no_nan_dataset_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x_no_nan
"""

import argparse
import sys

import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("⚠️  lerobot not available. Install with: pip install lerobot")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def plot_state_gripper_from_dataset(repo_id: str, threshold: float = 0.8, output_path: str = None):
    """
    从LeRobot数据集中提取state的gripper值并绘制折线图
    
    Args:
        repo_id: LeRobot数据集repo_id
        threshold: gripper值的阈值（用于标记）
        output_path: 输出图片路径
    """
    if not LEROBOT_AVAILABLE:
        print("❌ lerobot不可用，无法加载数据集")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib不可用，无法绘图")
        return
    
    print(f"📂 加载数据集: {repo_id}")
    
    try:
        dataset = LeRobotDataset(repo_id)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    # 获取episode数量
    num_episodes = len(dataset)
    print(f"📊 数据集包含 {num_episodes} 个episodes")
    
    # 收集所有episode的gripper值
    all_gripper_values = []
    all_indices = []
    
    for ep_idx in range(num_episodes):
        episode_gripper = []
        
        try:
            # 获取episode信息
            if hasattr(dataset, 'episode_data_index'):
                episode_index = dataset.episode_data_index
                if isinstance(episode_index, dict):
                    if 'from' in episode_index and 'to' in episode_index:
                        from_indices = episode_index['from']
                        to_indices = episode_index['to']
                        
                        # 转换为numpy数组（如果是tensor）
                        if TORCH_AVAILABLE and isinstance(from_indices, torch.Tensor):
                            from_indices = from_indices.cpu().numpy()
                        if TORCH_AVAILABLE and isinstance(to_indices, torch.Tensor):
                            to_indices = to_indices.cpu().numpy()
                        
                        if ep_idx < len(from_indices):
                            start_idx = int(from_indices[ep_idx])
                            end_idx = int(to_indices[ep_idx])
                            
                            # 遍历episode的所有帧
                            for frame_idx in range(start_idx, end_idx):
                                try:
                                    frame = dataset[frame_idx]
                                    if isinstance(frame, dict):
                                        # 提取state的最后一维（gripper值）
                                        for key in ["state", "observation.state", "observation/state"]:
                                            if key in frame:
                                                state = frame[key]
                                                # 转换为numpy数组
                                                if TORCH_AVAILABLE and isinstance(state, torch.Tensor):
                                                    state = state.cpu().numpy()
                                                else:
                                                    state = np.array(state)
                                                
                                                # state应该是8维，最后一维是gripper值
                                                if state.ndim == 1 and len(state) >= 8:
                                                    gripper_value = float(state[7])  # 索引7是最后一维
                                                    episode_gripper.append(gripper_value)
                                                break
                                except Exception:
                                    continue
        except Exception as e:
            print(f"  ⚠️  访问episode {ep_idx} 数据时出错: {e}")
        
        if episode_gripper:
            all_gripper_values.extend(episode_gripper)
            # 记录全局索引
            start_global_idx = len(all_indices)
            all_indices.extend(range(start_global_idx, start_global_idx + len(episode_gripper)))
            print(f"  ✅ Episode {ep_idx}: {len(episode_gripper)} 个gripper值")
    
    if len(all_gripper_values) == 0:
        print("❌ 没有提取到任何gripper值")
        return
    
    all_gripper_values = np.array(all_gripper_values)
    
    print()
    print("=" * 80)
    print("📊 统计信息:")
    print("=" * 80)
    print(f"   总样本数: {len(all_gripper_values)}")
    print(f"   最小值: {np.min(all_gripper_values):.6f}")
    print(f"   最大值: {np.max(all_gripper_values):.6f}")
    print(f"   平均值: {np.mean(all_gripper_values):.6f}")
    print(f"   中位数: {np.median(all_gripper_values):.6f}")
    print(f"   标准差: {np.std(all_gripper_values):.6f}")
    print(f"   大于{threshold}的数量: {np.sum(all_gripper_values > threshold)} ({np.sum(all_gripper_values > threshold)/len(all_gripper_values)*100:.2f}%)")
    print("=" * 80)
    
    # 绘制折线图
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # 绘制折线图
    ax.plot(all_indices, all_gripper_values, 'b-', linewidth=1.5, label='State Gripper Value (No NaN)', alpha=0.7)
    
    # 标记大于阈值的点
    above_threshold_mask = all_gripper_values > threshold
    if np.any(above_threshold_mask):
        ax.scatter(
            np.array(all_indices)[above_threshold_mask],
            all_gripper_values[above_threshold_mask],
            c='red',
            s=30,
            marker='o',
            label=f'> {threshold} (closed)',
            zorder=5
        )
    
    # 添加阈值线
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold={threshold}')
    
    # 设置标签和标题
    ax.set_xlabel('Time Step (All Episodes, NaN Removed)', fontsize=12)
    ax.set_ylabel('State Gripper Value (0=open, 1=closed)', fontsize=12)
    ax.set_title(
        f'State Gripper Values from Dataset (No NaN)\n'
        f'Total samples: {len(all_gripper_values)}, >{threshold}: {np.sum(above_threshold_mask)} ({np.sum(above_threshold_mask)/len(all_gripper_values)*100:.2f}%)',
        fontsize=12,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # 设置y轴范围
    y_min = min(0, np.min(all_gripper_values) * 0.1)
    y_max = max(1.0, np.max(all_gripper_values) * 1.1)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Image saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="从去掉NaN后的LeRobot数据集中提取state的gripper值并绘制折线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绘制gripper值折线图
  python scripts/test_no_nan_dataset_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x_no_nan

  # 保存图片
  python scripts/test_no_nan_dataset_gripper.py --repo-id your_hf_username/pick_blue_bottle_libero_downsample4x_no_nan --output no_nan_gripper_plot.png
        """
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='LeRobot数据集repo_id'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='gripper值的阈值（默认：0.8）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='图片输出路径（如果指定，则保存图片；否则显示）'
    )
    
    args = parser.parse_args()
    
    plot_state_gripper_from_dataset(
        repo_id=args.repo_id,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()


















