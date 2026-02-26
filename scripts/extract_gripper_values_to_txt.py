#!/usr/bin/env python3
"""
从HDF5文件中提取_control_gripperValueR话题的所有数据值并保存到txt文件

使用方法:
    python scripts/extract_gripper_values_to_txt.py <hdf5_file_path> [output_file.txt]
    python scripts/extract_gripper_values_to_txt.py --directory <directory> --max-files 5
"""

import argparse
import sys
import os
import glob

import h5py
import numpy as np


def extract_gripper_values_to_txt(hdf5_path: str, output_path: str = None):
    """
    从HDF5文件中提取_control_gripperValueR话题的所有数据值
    
    Args:
        hdf5_path: HDF5文件路径
        output_path: 输出txt文件路径（如果为None，则输出到标准输出）
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查是否存在gripper话题
            if "_control_gripperValueR" not in f.get("topics", {}):
                print(f"❌ 未找到 _control_gripperValueR 话题")
                return False
            
            gripper_topic = f["topics/_control_gripperValueR"]
            if "data" not in gripper_topic:
                print(f"❌ 夹爪话题中没有 'data' 键")
                return False
            
            # 读取所有gripper数据
            gripper_data = gripper_topic["data"][:]  # (T,)
            
            # 准备输出内容
            output_lines = []
            output_lines.append("=" * 80)
            output_lines.append(f"HDF5文件: {hdf5_path}")
            output_lines.append(f"话题: _control_gripperValueR")
            output_lines.append(f"数据总数: {len(gripper_data)}")
            output_lines.append("=" * 80)
            output_lines.append("")
            output_lines.append("数据统计:")
            output_lines.append("-" * 80)
            
            # 过滤NaN值进行统计
            valid_data = gripper_data[~np.isnan(gripper_data)]
            nan_count = np.sum(np.isnan(gripper_data))
            
            if len(valid_data) > 0:
                output_lines.append(f"有效值数量: {len(valid_data)}")
                output_lines.append(f"NaN值数量: {nan_count}")
                output_lines.append(f"最小值: {np.min(valid_data):.6f}")
                output_lines.append(f"最大值: {np.max(valid_data):.6f}")
                output_lines.append(f"平均值: {np.mean(valid_data):.6f}")
                output_lines.append(f"中位数: {np.median(valid_data):.6f}")
                output_lines.append(f"标准差: {np.std(valid_data):.6f}")
            else:
                output_lines.append("⚠️  所有值都是NaN")
            
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("所有数据值:")
            output_lines.append("-" * 80)
            output_lines.append("索引\t值")
            output_lines.append("-" * 80)
            
            # 写入所有数据值
            for i, value in enumerate(gripper_data):
                if np.isnan(value):
                    output_lines.append(f"{i}\tnan")
                else:
                    output_lines.append(f"{i}\t{value:.6f}")
            
            output_lines.append("")
            output_lines.append("=" * 80)
            
            output_content = "\n".join(output_lines)
            
            # 输出到文件或标准输出
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_content)
                    print(f"✅ Gripper值已保存到: {output_path}")
                    print(f"   共 {len(gripper_data)} 个值")
                    if len(valid_data) > 0:
                        print(f"   有效值: {len(valid_data)}, NaN值: {nan_count}")
                    return True
                except Exception as e:
                    print(f"❌ 保存文件失败: {e}")
                    return False
            else:
                print(output_content)
                return True
                
    except Exception as e:
        print(f"❌ 读取HDF5文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_from_directory(directory: str, max_files: int = None, output_dir: str = None):
    """
    从目录中批量提取gripper值
    
    Args:
        directory: HDF5文件所在目录
        max_files: 最多处理的文件数量
        output_dir: 输出目录（如果为None，则在每个文件同目录下创建txt文件）
    """
    pattern = os.path.join(directory, "**", "*.h5")
    h5_files = sorted(glob.glob(pattern, recursive=True))
    
    if max_files:
        h5_files = h5_files[:max_files]
    
    if len(h5_files) == 0:
        print(f"❌ 在目录 {directory} 中未找到HDF5文件")
        return
    
    print(f"📂 批量处理目录: {directory}")
    print(f"📊 找到 {len(h5_files)} 个HDF5文件\n")
    
    success_count = 0
    for h5_file in h5_files:
        filename = os.path.basename(h5_file)
        print(f"📄 处理文件: {filename}")
        
        if output_dir:
            # 在指定输出目录创建txt文件
            os.makedirs(output_dir, exist_ok=True)
            txt_filename = os.path.splitext(filename)[0] + "_gripper_values.txt"
            output_path = os.path.join(output_dir, txt_filename)
        else:
            # 在同目录下创建txt文件
            txt_filename = os.path.splitext(h5_file)[0] + "_gripper_values.txt"
            output_path = txt_filename
        
        if extract_gripper_values_to_txt(h5_file, output_path):
            success_count += 1
        print()
    
    print(f"✅ 处理完成: {success_count}/{len(h5_files)} 个文件成功")


def main():
    parser = argparse.ArgumentParser(
        description="从HDF5文件中提取_control_gripperValueR话题的所有数据值并保存到txt文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取单个文件的gripper值
  python scripts/extract_gripper_values_to_txt.py pick_blue_bottle/rosbag2_2026_01_09-21_25_15/rosbag2_2026_01_09-21_25_15_0.h5 gripper_values.txt

  # 批量提取目录中所有文件的gripper值
  python scripts/extract_gripper_values_to_txt.py --directory pick_blue_bottle --max-files 5 --output-dir gripper_values_output
        """
    )
    
    parser.add_argument(
        'hdf5_file',
        type=str,
        nargs='?',
        help='HDF5文件路径（单个文件模式）'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help='输出txt文件路径（单个文件模式，可选）'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='HDF5文件所在目录（批量模式）'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='批量模式下最多处理的文件数量（默认：处理所有文件）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='批量模式下的输出目录（如果指定，所有txt文件保存在此目录；否则保存在每个HDF5文件同目录）'
    )
    
    args = parser.parse_args()
    
    if args.directory:
        # 批量模式
        extract_from_directory(args.directory, args.max_files, args.output_dir)
    elif args.hdf5_file:
        # 单个文件模式
        extract_gripper_values_to_txt(args.hdf5_file, args.output_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()


















