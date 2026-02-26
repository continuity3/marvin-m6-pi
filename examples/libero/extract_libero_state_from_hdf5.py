#!/usr/bin/env python3
"""
从Libero HDF5文件中提取state和actions的示例脚本

根据你提供的HDF5结构：
demo_0/
  actions: shape=(272, 7), dtype=float64
  obs/
    ee_pos: shape=(272, 3), dtype=float64
    ee_ori: shape=(272, 3), dtype=float64
    gripper_states: shape=(272, 2), dtype=float64
    ...
"""

import h5py
import numpy as np
from pathlib import Path
from openpi.shared.normalize import RunningStats, save, load
from openpi.transforms import Normalize, Unnormalize
import tyro


def extract_state_from_hdf5(hdf5_file, demo_key="demo_0"):
    """
    从HDF5文件中提取8维state
    
    State组成:
    - state[0:3]: ee_pos (末端执行器位置 x, y, z)
    - state[3:6]: ee_ori (末端执行器方向，轴角表示 rx, ry, rz)
    - state[6:8]: gripper_states (夹爪状态，2维)
    
    Args:
        hdf5_file: 打开的HDF5文件对象
        demo_key: demo的键名，如"demo_0"
    
    Returns:
        state: (T, 8) numpy数组，dtype=float32
    """
    demo_group = hdf5_file[demo_key]
    obs_group = demo_group["obs"]
    
    # 提取各个组件
    ee_pos = obs_group["ee_pos"][:]  # (T, 3)
    ee_ori = obs_group["ee_ori"][:]  # (T, 3) - 轴角表示
    gripper_states = obs_group["gripper_states"][:]  # (T, 2)
    
    # 组合成8维state
    state = np.concatenate([ee_pos, ee_ori, gripper_states], axis=1)  # (T, 8)
    
    return state.astype(np.float32)


def extract_actions_from_hdf5(hdf5_file, demo_key="demo_0"):
    """
    从HDF5文件中提取actions
    
    Args:
        hdf5_file: 打开的HDF5文件对象
        demo_key: demo的键名
    
    Returns:
        actions: (T, 7) numpy数组，dtype=float32
    """
    demo_group = hdf5_file[demo_key]
    actions = demo_group["actions"][:]  # (T, 7)
    return actions.astype(np.float32)


def compute_norm_stats_from_hdf5_files(hdf5_files):
    """
    从多个HDF5文件计算归一化统计信息
    
    Args:
        hdf5_files: HDF5文件路径列表
    
    Returns:
        norm_stats: 包含state和actions的归一化统计信息
    """
    from tqdm import tqdm
    
    # 初始化统计信息收集器
    state_stats = RunningStats()
    action_stats = RunningStats()
    
    print(f"处理 {len(hdf5_files)} 个HDF5文件...")
    
    # 遍历所有HDF5文件
    for hdf5_path in tqdm(hdf5_files, desc="计算统计信息"):
        with h5py.File(hdf5_path, "r") as f:
            # 遍历所有demo
            for demo_key in f.keys():
                if not demo_key.startswith("demo_"):
                    continue
                
                try:
                    # 提取state
                    state = extract_state_from_hdf5(f, demo_key)
                    
                    # 提取actions
                    actions = extract_actions_from_hdf5(f, demo_key)
                    
                    # 更新统计信息
                    state_stats.update(state)
                    action_stats.update(actions)
                except Exception as e:
                    print(f"⚠️  处理 {hdf5_path}/{demo_key} 时出错: {e}")
                    continue
    
    # 获取最终统计信息
    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    
    return norm_stats


def print_norm_stats(norm_stats):
    """打印归一化统计信息"""
    print("\n" + "=" * 80)
    print("归一化统计信息")
    print("=" * 80)
    
    for key in ["state", "actions"]:
        stats = norm_stats[key]
        print(f"\n{key}:")
        print(f"  Shape: {stats.mean.shape}")
        print(f"  Mean:  {stats.mean}")
        print(f"  Std:   {stats.std}")
        if stats.q01 is not None:
            print(f"  Q01:   {stats.q01}")
            print(f"  Q99:   {stats.q99}")


def main(
    hdf5_dir: str,
    output_dir: str | None = None,
    compute_stats: bool = True,
    print_stats: bool = True,
):
    """
    主函数
    
    Args:
        hdf5_dir: HDF5文件所在目录
        output_dir: 输出目录（保存norm_stats.json）
        compute_stats: 是否计算归一化统计信息
        print_stats: 是否打印统计信息
    """
    hdf5_dir = Path(hdf5_dir)
    
    # 查找所有HDF5文件
    hdf5_files = sorted(list(hdf5_dir.glob("*.hdf5")) + list(hdf5_dir.glob("*.h5")))
    
    if not hdf5_files:
        raise ValueError(f"在 {hdf5_dir} 中找不到HDF5文件")
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 示例：提取第一个文件的第一个demo
    print("\n" + "=" * 80)
    print("示例：提取第一个demo的数据")
    print("=" * 80)
    
    with h5py.File(hdf5_files[0], "r") as f:
        demo_keys = [k for k in f.keys() if k.startswith("demo_")]
        if demo_keys:
            demo_key = demo_keys[0]
            print(f"\n处理: {hdf5_files[0].name}/{demo_key}")
            
            state = extract_state_from_hdf5(f, demo_key)
            actions = extract_actions_from_hdf5(f, demo_key)
            
            print(f"\nState shape: {state.shape}")
            print(f"State dtype: {state.dtype}")
            print(f"State components:")
            print(f"  - ee_pos (0:3):      {state[0, 0:3]}")
            print(f"  - ee_ori (3:6):      {state[0, 3:6]}")
            print(f"  - gripper (6:8):     {state[0, 6:8]}")
            print(f"\nActions shape: {actions.shape}")
            print(f"Actions dtype: {actions.dtype}")
            print(f"Actions (first step): {actions[0]}")
    
    # 计算归一化统计信息
    if compute_stats:
        print("\n" + "=" * 80)
        print("计算归一化统计信息")
        print("=" * 80)
        
        norm_stats = compute_norm_stats_from_hdf5_files(hdf5_files)
        
        if print_stats:
            print_norm_stats(norm_stats)
        
        # 保存统计信息
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save(output_dir, norm_stats)
            print(f"\n✅ 统计信息已保存到: {output_dir / 'norm_stats.json'}")
        
        # 示例：使用归一化
        print("\n" + "=" * 80)
        print("归一化示例")
        print("=" * 80)
        
        normalize = Normalize(norm_stats, use_quantiles=False)
        unnormalize = Unnormalize(norm_stats, use_quantiles=False)
        
        # 归一化示例state
        sample_state = state[0:1]  # (1, 8)
        normalized_state = normalize({"state": sample_state})["state"]
        
        print(f"\n原始state: {sample_state[0]}")
        print(f"归一化state: {normalized_state[0]}")
        
        # 反归一化
        restored_state = unnormalize({"state": normalized_state})["state"]
        print(f"反归一化state: {restored_state[0]}")
        print(f"误差: {np.abs(sample_state[0] - restored_state[0]).max()}")


if __name__ == "__main__":
    tyro.cli(main)

















