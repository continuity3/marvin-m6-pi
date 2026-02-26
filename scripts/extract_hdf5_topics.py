#!/usr/bin/env python3
"""
从HDF5文件中提取所有话题并保存到txt文件

使用方法:
    python scripts/extract_hdf5_topics.py <hdf5_file_path> [output_file.txt]
"""

import argparse
import sys

import h5py


def extract_topics_from_hdf5(hdf5_path: str, output_path: str = None):
    """
    从HDF5文件中提取所有话题
    
    Args:
        hdf5_path: HDF5文件路径
        output_path: 输出txt文件路径（如果为None，则输出到标准输出）
    """
    topics_list = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 递归遍历所有键，找出所有话题
            def collect_topics(name, obj):
                # 话题通常在 topics/ 目录下
                if name.startswith('topics/'):
                    # 移除 'topics/' 前缀
                    topic_name = name[7:]  # len('topics/') = 7
                    # 只保留话题名称（不包括子键）
                    if '/' not in topic_name:
                        topics_list.append(topic_name)
                    elif topic_name.endswith('/data') or topic_name.endswith('/names'):
                        # 这是话题的子键，提取话题名称
                        topic_base = topic_name.rsplit('/', 1)[0]
                        if topic_base not in topics_list:
                            topics_list.append(topic_base)
            
            f.visititems(collect_topics)
            
            # 去重并排序
            topics_list = sorted(set(topics_list))
            
    except Exception as e:
        print(f"❌ 读取HDF5文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 准备输出内容
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"HDF5文件: {hdf5_path}")
    output_lines.append(f"话题总数: {len(topics_list)}")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("话题列表:")
    output_lines.append("-" * 80)
    
    for i, topic in enumerate(topics_list, 1):
        output_lines.append(f"{i:3d}. {topic}")
    
    output_lines.append("")
    output_lines.append("=" * 80)
    
    output_content = "\n".join(output_lines)
    
    # 输出到文件或标准输出
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"✅ 话题列表已保存到: {output_path}")
            print(f"   共 {len(topics_list)} 个话题")
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            sys.exit(1)
    else:
        print(output_content)


def main():
    parser = argparse.ArgumentParser(
        description="从HDF5文件中提取所有话题并保存到txt文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取话题并保存到文件
  python scripts/extract_hdf5_topics.py pick_blue_bottle/rosbag2_2026_01_09-21_25_15/rosbag2_2026_01_09-21_25_15_0.h5 topics.txt

  # 输出到标准输出
  python scripts/extract_hdf5_topics.py pick_blue_bottle/rosbag2_2026_01_09-21_25_15/rosbag2_2026_01_09-21_25_15_0.h5
        """
    )
    
    parser.add_argument(
        'hdf5_file',
        type=str,
        help='HDF5文件路径'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help='输出txt文件路径（可选，如果不指定则输出到标准输出）'
    )
    
    args = parser.parse_args()
    
    extract_topics_from_hdf5(args.hdf5_file, args.output_file)


if __name__ == '__main__':
    main()


















