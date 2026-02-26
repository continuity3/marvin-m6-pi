#!/usr/bin/env python3
"""ROS2原生API HDF5转换器

基于 rosbag2_py / rosidl_runtime_py / rclpy.serialization，
仅支持 ROS2 bag（sqlite3 / mcap），提供与 convert_any_rosbag_to_hdf5.py 相同的
核心能力：
- 自动检测 sensor_msgs/Image 与 JointState 话题
- 统一时间轴/原始模式二选一
- HDFView 兼容模式（默认启用）
- JPEG 压缩 / 固定长度缓冲（节省空间）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import cv2

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "convert_ros2bag_to_hdf5_native.py 需要 ROS2 Python 环境 (rosbag2_py, rclpy)。"
    ) from exc

# ---------------------------------------------------------------------------
# ROS2 帮助函数
# ---------------------------------------------------------------------------
MESSAGE_TYPE_CACHE: Dict[str, Any] = {}


def _progress(message: str) -> None:
    print(f"[进度] {message}")


def _get_storage_id(bag_path: Path) -> str:
    if bag_path.is_file():
        if bag_path.suffix == ".mcap":
            return "mcap"
        if bag_path.suffix == ".db3":
            return "sqlite3"
    if any(bag_path.rglob("*.mcap")):
        return "mcap"
    return "sqlite3"


def _ensure_ros2_bag(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ROS2 bag 不存在: {path}")
    if path.is_dir() and not (path / "metadata.yaml").exists() and not list(path.glob("*.db3")):
        raise ValueError(f"目录 {path} 不是有效的 ROS2 bag（缺少 metadata.yaml 或 *.db3）")
    if path.is_file() and path.suffix not in {".mcap", ".db3"}:
        raise ValueError("仅支持含 metadata.yaml 的目录、.db3 或 .mcap 文件")


def _open_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    storage_id = _get_storage_id(bag_path)
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def _read_topic_types(bag_path: Path) -> Dict[str, str]:
    reader = _open_reader(bag_path)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    return topic_types


def _msg_class(msg_type: str):
    if msg_type not in MESSAGE_TYPE_CACHE:
        MESSAGE_TYPE_CACHE[msg_type] = get_message(msg_type)
    return MESSAGE_TYPE_CACHE[msg_type]


def _deserialize(raw: bytes, msg_type: str):
    msg_cls = _msg_class(msg_type)
    return deserialize_message(raw, msg_cls)


def _iterate_ros2_messages(
    bag_path: Path,
    topic_types: Dict[str, str],
    topic_filter: Optional[Iterable[str]] = None,
):
    topics = set(topic_filter) if topic_filter else None
    reader = _open_reader(bag_path)
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topics and topic not in topics:
            continue
        msg_type = topic_types.get(topic)
        if not msg_type:
            continue
        try:
            msg = _deserialize(data, msg_type)
        except Exception as exc:
            print(f"警告: 无法解析 {topic} ({msg_type}): {exc}")
            continue
        yield topic, timestamp, msg, msg_type


# ---------------------------------------------------------------------------
# 话题检测
# ---------------------------------------------------------------------------
IMAGE_TYPE_HINTS = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/Image",
    # "sensor_msgs/msg/CompressedImage",
    # "sensor_msgs/CompressedImage",
    
}
JOINT_TYPE_HINTS = {"sensor_msgs/msg/JointState", "sensor_msgs/JointState"}
SPECIAL_JOINT_TOPICS = {"/control/gripperValueR", "/control/gripperValueL"}  # 仅保留标量
POSE_TYPE_HINTS = {
    "geometry_msgs/msg/PoseStamped", "geometry_msgs/PoseStamped",
    "geometry_msgs/msg/Pose", "geometry_msgs/Pose",
}
FOOT_SWITCH_TYPE_HINTS = {"foot_switch/msg/FootSwitch"}  # 新增
FLOAT32_MULTIARRAY_TYPE_HINTS = {"std_msgs/msg/Float32MultiArray", "std_msgs/Float32MultiArray"}  # 新增

# Topic名称映射：将ROS topic名称映射到HDF5中的名称
# realsense435的image topics映射为image_wrist
TOPIC_NAME_MAPPING = {
    "/image_raw": "image_wrist",
    "/image_raw/compressed": "image_wrist",
    "/image_raw/compressedDepth": "image_wrist",
    "/image_raw/theora": "image_wrist",
}

def _get_hdf5_topic_name(topic: str) -> str:
    """获取topic在HDF5中的名称（应用映射规则）"""
    # 如果topic在映射字典中，返回映射后的名称
    if topic in TOPIC_NAME_MAPPING:
        return TOPIC_NAME_MAPPING[topic]
    # 如果topic以映射字典中的某个key开头（用于处理子topic），也进行映射
    for ros_topic, hdf5_name in TOPIC_NAME_MAPPING.items():
        if topic.startswith(ros_topic + "/") or topic == ros_topic:
            return hdf5_name
    return topic

def detect_topics(bag_path: Path) -> Dict[str, List[str]]:
    topic_types = _read_topic_types(bag_path)
    images, joints, poses, foot_switches, float32_arrays = [], [], [], [], []  # 新增 float32_arrays
    for topic, msg_type in topic_types.items():
        if any(hint in msg_type for hint in IMAGE_TYPE_HINTS):
            images.append(topic)
        elif any(hint in msg_type for hint in JOINT_TYPE_HINTS) or topic in SPECIAL_JOINT_TOPICS:
            joints.append(topic)
        elif any(hint in msg_type for hint in POSE_TYPE_HINTS):
            poses.append(topic)
        elif any(hint in msg_type for hint in FOOT_SWITCH_TYPE_HINTS):  # 新增
            foot_switches.append(topic)
        elif any(hint in msg_type for hint in FLOAT32_MULTIARRAY_TYPE_HINTS):  # 新增
            float32_arrays.append(topic)
    return {
        "image": sorted(images),
        "joint_state": sorted(joints),
        "pose": sorted(poses),
        "foot_switch": sorted(foot_switches),  # 新增
        "float32_multiarray": sorted(float32_arrays),  # 新增
        "types": topic_types,
    }


# ---------------------------------------------------------------------------
# 图像转换
# ---------------------------------------------------------------------------

def _image_msg_to_array(msg, msg_type: str) -> np.ndarray:
    if "CompressedImage" in msg_type:
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码 CompressedImage")
        return img

    height = msg.height
    width = msg.width
    encoding = getattr(msg, "encoding", "")
    data = bytes(msg.data)

    def ensure_bgr(array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        return array

    if encoding == "rgb8":
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    if encoding == "mono8":
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return ensure_bgr(array)
    if encoding == "mono16":
        array = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        array = (array / 256).astype(np.uint8)
        return ensure_bgr(array)
    if encoding in {"16UC1", "32FC1", "depth"}:
        dtype = np.uint16 if encoding == "16UC1" else np.float32
        array = np.frombuffer(data, dtype=dtype).reshape(height, width)
        vmax = np.max(array)
        if vmax > 0:
            array = (array / vmax * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        return ensure_bgr(array)
    if encoding == "rgba8":
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        return array[:, :, :3]
    if encoding == "yuv422_yuy2" or encoding == "yuyv":
        # YUY2/YUV422 格式：每个像素2字节，每2个像素共享U和V
        # 数据格式：Y0 U0 Y1 V0 Y2 U2 Y3 V2 ... (每4字节=2像素)
        array = np.frombuffer(data, dtype=np.uint8)
        # YUY2 每4字节代表2个像素，所以总字节数 = height * width * 2
        if len(array) != height * width * 2:
            raise ValueError(f"YUY2数据长度不匹配: 期望 {height * width * 2}, 实际 {len(array)}")
        
        # 手动解析YUY2格式并转换为BGR
        # YUY2格式：每4字节 = [Y0, U, Y1, V]，代表2个像素
        yuyv_flat = array.reshape(-1, 4)  # shape: (height*width/2, 4)
        
        # 提取Y, U, V分量
        y0 = yuyv_flat[:, 0]  # 偶数像素的Y
        u = yuyv_flat[:, 1]    # U分量（每2像素共享）
        y1 = yuyv_flat[:, 2]  # 奇数像素的Y
        v = yuyv_flat[:, 3]    # V分量（每2像素共享）
        
        # 重塑为图像尺寸
        y0_img = y0.reshape(height, width // 2)
        u_img = u.reshape(height, width // 2)
        y1_img = y1.reshape(height, width // 2)
        v_img = v.reshape(height, width // 2)
        
        # 扩展U和V到每个像素（因为YUY2是422格式，U和V是共享的）
        u_expanded = np.repeat(u_img, 2, axis=1)
        v_expanded = np.repeat(v_img, 2, axis=1)
        
        # 合并Y通道（交错Y0和Y1）
        y_combined = np.zeros((height, width), dtype=np.uint8)
        y_combined[:, ::2] = y0_img
        y_combined[:, 1::2] = y1_img
        
        # 构建YUV图像 (height, width, 3) 并转换为BGR
        yuv = np.stack([y_combined, u_expanded, v_expanded], axis=2)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr

    # fallback
    array = np.frombuffer(data, dtype=np.uint8)
    try:
        array = array.reshape(height, width, -1)
        if array.shape[2] == 3:
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    raise ValueError(f"不支持的图像编码: {encoding}")


# ---------------------------------------------------------------------------
# 时间轴扫描/填充
# ---------------------------------------------------------------------------

def _scan_bag_for_timeline(
    bag_path: Path,
    topic_types: Dict[str, str],
    image_topics: List[str],
    joint_topics: List[str],
    pose_topics: List[str],
    foot_switch_topics: List[str],  # 新增参数
    float32_array_topics: List[str],  # 新增参数
) -> Dict[str, Any]:
    _progress("阶段1/3：扫描时间范围与话题信息")
    start_scan = time.time()
    start_time = None
    end_time = None
    joint_info: Dict[str, Dict[str, Any]] = {}
    image_info: Dict[str, Dict[str, Any]] = {}
    pose_info: Dict[str, Dict[str, Any]] = {}
    foot_switch_info: Dict[str, Dict[str, Any]] = {}  # 新增
    float32_array_info: Dict[str, Dict[str, Any]] = {}  # 新增
    msg_count = 0

    for topic, timestamp, msg, msg_type in _iterate_ros2_messages(
        bag_path, topic_types, topic_filter=set(image_topics) | set(joint_topics) | set(pose_topics) | set(foot_switch_topics) | set(float32_array_topics)  # 添加 float32_array
    ):
        msg_count += 1
        if msg_count % 5000 == 0:
            _progress(f"阶段1/3：已扫描 {msg_count} 条消息")
        sec = timestamp / 1e9
        if start_time is None or sec < start_time:
            start_time = sec
        if end_time is None or sec > end_time:
            end_time = sec

        if topic in joint_topics and topic not in joint_info:
            # 处理特殊话题 (Float32)
            if topic in SPECIAL_JOINT_TOPICS:
                joint_info[topic] = {"dof": 1, "names": [topic.split('/')[-1]], "is_float32": True}
            # 处理标准 JointState
            else:
                dof = len(getattr(msg, "position", []))
                names = list(getattr(msg, "name", [])) or [f"joint_{i}" for i in range(dof)]
                joint_info[topic] = {"dof": dof, "names": names, "is_float32": False}
        elif topic in pose_topics and topic not in pose_info:
            pose_info[topic] = {
                "names": ["x","y","z","qx","qy","qz","qw"],
                "dof": 7,
                "msg_type": msg_type,
            }
        elif topic in image_topics and topic not in image_info:
            arr = _image_msg_to_array(msg, msg_type)
            h, w = arr.shape[:2]
            encoding = getattr(msg, "encoding", "compressed") if "CompressedImage" not in msg_type else getattr(msg, "format", "jpeg")
            image_info[topic] = {"width": w, "height": h, "encoding": encoding, "msg_type": msg_type}
        elif topic in foot_switch_topics and topic not in foot_switch_info:  # 新增
            foot_switch_info[topic] = {
                "names": ["key_point"],
                "dof": 1,
                "msg_type": msg_type,
            }
        elif topic in float32_array_topics and topic not in float32_array_info:  # 新增
            data = list(getattr(msg, "data", []))
            dof = len(data) if data else 0
            # 使用 layout 信息或默认名称
            names = []
            if hasattr(msg, "layout") and hasattr(msg.layout, "dim") and msg.layout.dim:
                names = [dim.label if dim.label else f"dim_{i}" for i, dim in enumerate(msg.layout.dim)]
            if not names:
                names = [f"element_{i}" for i in range(dof)] if dof > 0 else ["data"]
            float32_array_info[topic] = {
                "names": names,
                "dof": dof,
                "msg_type": msg_type,
                "is_variable_length": False,  # 初始假设固定长度
            }
        elif topic in float32_array_topics and topic in float32_array_info:  # 检查长度是否一致
            data = list(getattr(msg, "data", []))
            current_dof = len(data) if data else 0
            if current_dof != float32_array_info[topic]["dof"]:
                # 如果发现长度不一致，标记为可变长度
                float32_array_info[topic]["is_variable_length"] = True

        if set(joint_info.keys()) == set(joint_topics) and set(image_info.keys()) == set(image_topics) and set(foot_switch_info.keys()) == set(foot_switch_topics) and set(float32_array_info.keys()) == set(float32_array_topics):
            continue

    if start_time is not None and end_time is not None:
        time_range = f"{start_time:.3f}s ~ {end_time:.3f}s"
    else:
        time_range = "无有效数据"
    _progress(
        f"阶段1/3 完成：耗时 {time.time() - start_scan:.1f}s，时间范围 {time_range}"
    )
    return {
        "start_time": start_time,
        "end_time": end_time,
        "joint_info": joint_info,
        "image_info": image_info,
        "pose_info": pose_info,
        "foot_switch_info": foot_switch_info,  # 新增
        "float32_array_info": float32_array_info,  # 新增
    }


def _generate_timeline(start: float, end: float, target_fps: float) -> np.ndarray:
    period = 1.0 / target_fps
    timeline_sec = np.arange(start, end + period, period)
    return (timeline_sec * 1e9).astype(np.int64)


def _prepare_datasets(
    hdf5_path: Path,
    timeline_ns: np.ndarray,
    scan_result: Dict[str, Any],
    image_topics: List[str],
    joint_topics: List[str],
    pose_topics: List[str],
    foot_switch_topics: List[str],  # 新增参数
    float32_array_topics: List[str],  # 新增参数
    target_fps: float,
    hdfview_compatible: bool,
    image_compression: bool,
    max_image_size: Optional[Tuple[int, int]],
) -> Tuple[h5py.File, Dict[str, Dict[str, h5py.Dataset]]]:
    _progress("阶段2/3：创建 HDF5 数据集结构")
    N = len(timeline_ns)
    compression_algo = "gzip" if hdfview_compatible else "lzf"
    f = h5py.File(hdf5_path, "w")
    f.create_dataset("time", data=timeline_ns, dtype="int64")
    topics_grp = f.create_group("topics")
    valid_grp = f.create_group("valid")
    meta_grp = f.create_group("meta")
    meta_grp.attrs.update(
        {
            "target_fps": target_fps,
            "start_time": scan_result["start_time"],
            "end_time": scan_result["end_time"],
            "timeline_length": N,
            "created_by": "convert_ros2bag_to_hdf5_native.py",
            "scheme": "B",
            "hdfview_compatible": hdfview_compatible,
        }
    )

    datasets: Dict[str, Dict[str, h5py.Dataset]] = {}

    for topic in joint_topics:
        if topic not in scan_result["joint_info"]:
            continue
        info = scan_result["joint_info"][topic]
        topic_safe = topic.replace("/", "_")
        grp = topics_grp.create_group(topic_safe)
        fillvalue = -999999.0 if hdfview_compatible else np.nan
        
        # Float32 话题只有一个值
        if info.get("is_float32", False):
            datasets[topic] = {
                "data": grp.create_dataset(
                    "data",
                    shape=(N,),  # 一维数组
                    dtype="float32",
                    fillvalue=fillvalue,
                    compression=compression_algo,
                    chunks=True,
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe,
                    shape=(N,),
                    dtype="bool",
                    fillvalue=False,
                    compression=compression_algo,
                    chunks=True,
                ),
            }
            grp.attrs["type"] = "Float32"
        else:
            # 标准 JointState
            datasets[topic] = {
                "position": grp.create_dataset(
                    "position",
                    shape=(N, info["dof"]),
                    dtype="float64",
                    fillvalue=fillvalue,
                    compression=compression_algo,
                    chunks=True,
                ),
                "velocity": grp.create_dataset(
                    "velocity",
                    shape=(N, info["dof"]),
                    dtype="float64",
                    fillvalue=fillvalue,
                    compression=compression_algo,
                    chunks=True,
                ),
                "effort": grp.create_dataset(
                    "effort",
                    shape=(N, info["dof"]),
                    dtype="float64",
                    fillvalue=fillvalue,
                    compression=compression_algo,
                    chunks=True,
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe,
                    shape=(N,),
                    dtype="bool",
                    fillvalue=False,
                    compression=compression_algo,
                    chunks=True,
                ),
            }
            grp.attrs["type"] = "JointState"
        
        grp.create_dataset("names", data=np.array(info["names"], dtype="S"))
        grp.attrs["topic"] = topic
        grp.attrs["dof"] = info["dof"]

    for topic in image_topics:
        if topic not in scan_result["image_info"]:
            continue
        info = scan_result["image_info"][topic]
        # 使用映射后的名称创建HDF5 group
        hdf5_topic_name = _get_hdf5_topic_name(topic)
        topic_safe = hdf5_topic_name.replace("/", "_")
        grp = topics_grp.create_group(topic_safe)
        w, h = info["width"], info["height"]
        if max_image_size and (w > max_image_size[0] or h > max_image_size[1]):
            scale = min(max_image_size[0] / w, max_image_size[1] / h)
            w, h = int(w * scale), int(h * scale)
        if hdfview_compatible:
            datasets[topic] = {
                "data": grp.create_dataset(
                    "data",
                    shape=(N, h, w, 3),
                    dtype="uint8",
                    fillvalue=0,
                    chunks=(1, h, w, 3),
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe,
                    shape=(N,),
                    dtype="bool",
                    fillvalue=False,
                    chunks=True,
                ),
            }
            grp.attrs["compressed"] = False
            grp.attrs["hdfview_compatible"] = True
        elif image_compression:
            max_compressed = h * w * 3
            datasets[topic] = {
                "data": grp.create_dataset(
                    "data",
                    shape=(N, max_compressed),
                    dtype="uint8",
                    fillvalue=0,
                    chunks=True,
                    compression=compression_algo,
                ),
                "data_length": grp.create_dataset(
                    "data_length",
                    shape=(N,),
                    dtype="int32",
                    fillvalue=0,
                    chunks=True,
                    compression=compression_algo,
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe,
                    shape=(N,),
                    dtype="bool",
                    fillvalue=False,
                    chunks=True,
                    compression=compression_algo,
                ),
            }
            grp.attrs.update(
                {
                    "compressed": True,
                    "fixed_length_compression": True,
                    "max_compressed_size": max_compressed,
                }
            )
        else:
            datasets[topic] = {
                "data": grp.create_dataset(
                    "data",
                    shape=(N, h, w, 3),
                    dtype="uint8",
                    fillvalue=0,
                    chunks=True,
                    compression=compression_algo,
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe,
                    shape=(N,),
                    dtype="bool",
                    fillvalue=False,
                    chunks=True,
                    compression=compression_algo,
                ),
            }
            grp.attrs["compressed"] = False
        grp.attrs["topic"] = topic
        grp.attrs["encoding"] = info["encoding"]
        grp.attrs["original_width"] = info["width"]
        grp.attrs["original_height"] = info["height"]

    # pose datasets
    for topic in pose_topics:
        info = scan_result["pose_info"].get(topic)
        if not info:
            continue
        topic_safe = topic.replace("/", "_")
        grp = topics_grp.create_group(topic_safe)
        datasets[topic] = {
            "position": grp.create_dataset(
                "position", shape=(N, 3), dtype="float64",
                compression=compression_algo, chunks=True, fillvalue=np.nan
            ),
            "orientation": grp.create_dataset(
                "orientation", shape=(N, 4), dtype="float64",
                compression=compression_algo, chunks=True, fillvalue=np.nan
            ),
            "valid": valid_grp.create_dataset(
                topic_safe, shape=(N,), dtype="bool",
                compression=compression_algo, chunks=True, fillvalue=False
            ),
        }
        grp.create_dataset("names", data=np.array(info["names"], dtype="S"))
        grp.attrs["type"] = "Pose"
        grp.attrs["topic"] = topic
        grp.attrs["dof"] = 7

    # 新增：foot_switch datasets
    for topic in foot_switch_topics:
        info = scan_result["foot_switch_info"].get(topic)
        if not info:
            continue
        topic_safe = topic.replace("/", "_")
        grp = topics_grp.create_group(topic_safe)
        datasets[topic] = {
            "key_point": grp.create_dataset(
                "key_point", shape=(N,), dtype="bool",
                compression=compression_algo, chunks=True, fillvalue=False
            ),
            "valid": valid_grp.create_dataset(
                topic_safe, shape=(N,), dtype="bool",
                compression=compression_algo, chunks=True, fillvalue=False
            ),
        }
        grp.create_dataset("names", data=np.array(info["names"], dtype="S"))
        grp.attrs["type"] = "FootSwitch"
        grp.attrs["topic"] = topic
        grp.attrs["dof"] = 1

    # 新增：float32_array datasets
    for topic in float32_array_topics:
        info = scan_result["float32_array_info"].get(topic)
        if not info:
            continue
        topic_safe = topic.replace("/", "_")
        grp = topics_grp.create_group(topic_safe)
        fillvalue = -999999.0 if hdfview_compatible else np.nan
        dof = info["dof"]
        is_variable_length = info.get("is_variable_length", False)
        
        if is_variable_length or dof == 0:
            # 使用可变长度数组
            datasets[topic] = {
                "data": grp.create_dataset(
                    "data", shape=(N,), dtype=h5py.vlen_dtype(np.float32),
                    compression=compression_algo, chunks=True
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe, shape=(N,), dtype="bool",
                    compression=compression_algo, chunks=True, fillvalue=False
                ),
            }
            grp.attrs["variable_length"] = True
        else:
            # 固定长度数组
            datasets[topic] = {
                "data": grp.create_dataset(
                    "data", shape=(N, dof), dtype="float32",
                    compression=compression_algo, chunks=True, fillvalue=fillvalue
                ),
                "valid": valid_grp.create_dataset(
                    topic_safe, shape=(N,), dtype="bool",
                    compression=compression_algo, chunks=True, fillvalue=False
                ),
            }
            grp.attrs["variable_length"] = False
        grp.create_dataset("names", data=np.array(info["names"], dtype="S"))
        grp.attrs["type"] = "Float32MultiArray"
        grp.attrs["topic"] = topic
        grp.attrs["dof"] = dof

    joint_created = sum(1 for topic in joint_topics if topic in scan_result["joint_info"])
    image_created = sum(1 for topic in image_topics if topic in scan_result["image_info"])
    pose_created = sum(1 for topic in pose_topics if topic in scan_result["pose_info"])   # 新增
    foot_switch_created = sum(1 for topic in foot_switch_topics if topic in scan_result["foot_switch_info"])
    float32_array_created = sum(1 for topic in float32_array_topics if topic in scan_result["float32_array_info"])
    _progress(
        f"阶段2/3 完成：关节话题 {joint_created} 个，图像话题 {image_created} 个，位姿话题 {pose_created} 个，踏板话题 {foot_switch_created} 个，浮点数组话题 {float32_array_created} 个"
    )
    return f, datasets


def _fill_data_with_timeline(
    bag_path: Path,
    topic_types: Dict[str, str],
    datasets: Dict[str, Dict[str, h5py.Dataset]],
    timeline_ns: np.ndarray,
    period_ns: int,
    image_topics: List[str],
    joint_topics: List[str],
    pose_topics: List[str],
    foot_switch_topics: List[str],  # 新增参数
    float32_array_topics: List[str],  # 新增参数
    image_compression: bool,
    max_image_size: Optional[Tuple[int, int]],
    hdfview_compatible: bool,
) -> None:
    _progress("阶段3/3：匹配并填充统一时间轴数据（流式写入优化）")
    tolerance = period_ns // 2
    
    # 第一次遍历：只记录最佳匹配的索引和时间戳（不缓存消息）
    best_matches: Dict[str, Dict[int, Tuple[int, int]]] = {  # 改为 (diff, timestamp)
        topic: {} for topic in image_topics + joint_topics + pose_topics + foot_switch_topics + float32_array_topics
    }
    processed = 0

    for topic, timestamp, msg, msg_type in _iterate_ros2_messages(
        bag_path, topic_types, topic_filter=set(image_topics) | set(joint_topics) | set(pose_topics) | set(foot_switch_topics) | set(float32_array_topics)
    ):
        processed += 1
        if processed % 5000 == 0:
            _progress(f"阶段3/3（第1轮）：已遍历 {processed} 条消息")
        
        diffs = np.abs(timeline_ns - timestamp)
        idx = int(np.argmin(diffs))
        if diffs[idx] > tolerance:
            continue
        
        prev = best_matches[topic].get(idx)
        if prev is None or diffs[idx] < prev[0]:
            best_matches[topic][idx] = (diffs[idx], timestamp)  # 只保存时间戳

    total_matches = sum(len(m) for m in best_matches.values())
    _progress(f"阶段3/3（第1轮）：找到 {total_matches} 个时间轴匹配")

    # 第二次遍历：流式写入数据
    _progress("阶段3/3（第2轮）：写入数据到 HDF5")
    written = 0
    processed = 0

    for topic, timestamp, msg, msg_type in _iterate_ros2_messages(
        bag_path, topic_types, topic_filter=set(image_topics) | set(joint_topics) | set(pose_topics) | set(foot_switch_topics) | set(float32_array_topics)
    ):
        processed += 1
        if processed % 2000 == 0:
            _progress(f"阶段3/3（第2轮）：已处理 {processed}/{total_matches + 10000} 条消息，已写入 {written} 条")

        if topic not in datasets:
            continue

        # 找到该消息对应的时间轴索引
        diffs = np.abs(timeline_ns - timestamp)
        idx = int(np.argmin(diffs))
        
        # 检查是否是最佳匹配
        match_info = best_matches[topic].get(idx)
        if match_info is None or match_info[1] != timestamp:
            continue

        # 立即写入数据
        try:
            if topic in image_topics:
                image = _image_msg_to_array(msg, msg_type)
                if max_image_size and (
                    image.shape[1] > max_image_size[0] or image.shape[0] > max_image_size[1]
                ):
                    image = cv2.resize(image, max_image_size)
                
                if hdfview_compatible or not image_compression:
                    datasets[topic]["data"][idx] = image
                else:
                    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    encoded_bytes = np.frombuffer(encoded.tobytes(), dtype=np.uint8)
                    datasets[topic]["data"][idx, :len(encoded_bytes)] = encoded_bytes
                    datasets[topic]["data_length"][idx] = len(encoded_bytes)
                datasets[topic]["valid"][idx] = True
                written += 1

            elif topic in joint_topics:
                if topic in SPECIAL_JOINT_TOPICS:
                    if hasattr(msg, "data"):
                        datasets[topic]["data"][idx] = msg.data
                        datasets[topic]["valid"][idx] = True
                        written += 1
                else:
                    if hasattr(msg, "position"):
                        datasets[topic]["position"][idx] = msg.position
                    if hasattr(msg, "velocity") and msg.velocity:
                        datasets[topic]["velocity"][idx] = msg.velocity
                    if hasattr(msg, "effort") and msg.effort:
                        datasets[topic]["effort"][idx] = msg.effort
                    datasets[topic]["valid"][idx] = True
                    written += 1

            elif topic in pose_topics:
                if "PoseStamped" in msg_type:
                    p, o = msg.pose.position, msg.pose.orientation
                else:
                    p, o = msg.position, msg.orientation
                datasets[topic]["position"][idx] = (p.x, p.y, p.z)
                datasets[topic]["orientation"][idx] = (o.x, o.y, o.z, o.w)
                datasets[topic]["valid"][idx] = True
                written += 1

            elif topic in foot_switch_topics:
                if hasattr(msg, "key_point"):
                    datasets[topic]["key_point"][idx] = msg.key_point
                    datasets[topic]["valid"][idx] = True
                    written += 1

            elif topic in float32_array_topics:
                if hasattr(msg, "data"):
                    data = list(msg.data)
                    if len(data) > 0:
                        data_array = np.array(data, dtype=np.float32)
                        # 检查数据集是否为可变长度
                        if datasets[topic]["data"].dtype == h5py.vlen_dtype(np.float32):
                            # 可变长度数组
                            datasets[topic]["data"][idx] = data_array
                        else:
                            # 固定长度数组
                            if len(data_array) == datasets[topic]["data"].shape[1]:
                                datasets[topic]["data"][idx] = data_array
                            else:
                                # 长度不匹配，跳过或使用截断/填充
                                print(f"警告: {topic}[{idx}] 数据长度不匹配: 期望 {datasets[topic]['data'].shape[1]}, 实际 {len(data_array)}")
                                continue
                        datasets[topic]["valid"][idx] = True
                        written += 1

        except Exception as exc:
            print(f"警告：写入 {topic}[{idx}] 失败: {exc}")

    _progress(f"阶段3/3 完成：共写入 {written} 个样本")


# ---------------------------------------------------------------------------
# 原始模式 (按原始时间戳)
# ---------------------------------------------------------------------------

def _collect_original_mode(
    bag_path: Path,
    topic_types: Dict[str, str],
    image_topics: List[str],
    joint_topics: List[str],
    pose_topics: List[str],
    foot_switch_topics: List[str],  # 新增参数
    float32_array_topics: List[str],  # 新增参数
    image_compression: bool,
    max_image_size: Optional[Tuple[int, int]],
) -> Tuple[Dict[str, List[Any]], Dict[str, List[float]], Dict[str, List[Any]], Dict[str, List[float]], Dict[str, List[Any]], Dict[str, List[float]], Dict[str, List[bool]], Dict[str, List[float]], Dict[str, List[Any]], Dict[str, List[float]]]:
    _progress("原始模式：开始收集原始时间戳数据")
    joint_data: Dict[str, List[List[float]]] = {}
    joint_timestamps: Dict[str, List[float]] = {}
    image_data: Dict[str, List[np.ndarray]] = {}
    image_timestamps: Dict[str, List[float]] = {}
    pose_data: Dict[str, List[List[float]]] = {}
    pose_timestamps: Dict[str, List[float]] = {}
    foot_switch_data: Dict[str, List[bool]] = {}
    foot_switch_timestamps: Dict[str, List[float]] = {}
    float32_array_data: Dict[str, List[List[float]]] = {}
    float32_array_timestamps: Dict[str, List[float]] = {}
    processed = 0

    for topic, timestamp, msg, msg_type in _iterate_ros2_messages(
        bag_path, topic_types, topic_filter=set(image_topics) | set(joint_topics) | set(pose_topics) | set(foot_switch_topics) | set(float32_array_topics)  # 加入 float32_array
    ):
        processed += 1
        if processed % 5000 == 0:
            _progress(f"原始模式：已收集 {processed} 条消息")
        sec = timestamp / 1e9
        if topic in joint_topics:
            # Float32 话题
            if topic in SPECIAL_JOINT_TOPICS:
                if hasattr(msg, "data"):
                    joint_data.setdefault(topic, []).append([msg.data])
                    joint_timestamps.setdefault(topic, []).append(sec)
            # 标准 JointState
            else:
                positions = list(getattr(msg, "position", []))
                if not positions:
                    continue
                joint_data.setdefault(topic, []).append(positions)
                joint_timestamps.setdefault(topic, []).append(sec)
        elif topic in image_topics:
            try:
                image = _image_msg_to_array(msg, msg_type)
            except Exception as exc:
                print(f"警告: 跳过图像 {topic}: {exc}")
                continue
            if max_image_size and (
                image.shape[1] > max_image_size[0] or image.shape[0] > max_image_size[1]
            ):
                image = cv2.resize(image, max_image_size)
            image_data.setdefault(topic, []).append(image)
            image_timestamps.setdefault(topic, []).append(sec)
        elif topic in pose_topics:
            if "PoseStamped" in msg_type:
                p, o = msg.pose.position, msg.pose.orientation
            else:
                p, o = msg.position, msg.orientation
            pose_data.setdefault(topic, []).append([p.x, p.y, p.z, o.x, o.y, o.z, o.w])
            pose_timestamps.setdefault(topic, []).append(sec)
        elif topic in foot_switch_topics:  # 新增
            if hasattr(msg, "key_point"):
                foot_switch_data.setdefault(topic, []).append(msg.key_point)
                foot_switch_timestamps.setdefault(topic, []).append(sec)
        elif topic in float32_array_topics:  # 新增
            if hasattr(msg, "data"):
                data = list(msg.data)
                if len(data) > 0:
                    float32_array_data.setdefault(topic, []).append(data)
                    float32_array_timestamps.setdefault(topic, []).append(sec)

    _progress(
        f"原始模式：收集完成，共处理 {processed} 条消息"
    )
    return joint_data, joint_timestamps, image_data, image_timestamps, pose_data, pose_timestamps, foot_switch_data, foot_switch_timestamps, float32_array_data, float32_array_timestamps  # 改为元组返回


def _write_original_mode(
    hdf5_path: Path,
    joint_data: Dict[str, List[Any]],
    joint_ts: Dict[str, List[float]],
    image_data: Dict[str, List[np.ndarray]],
    image_ts: Dict[str, List[float]],
    pose_data: Dict[str, List[Any]],
    pose_ts: Dict[str, List[float]],
    foot_switch_data: Dict[str, List[bool]],  # 新增
    foot_switch_ts: Dict[str, List[float]],  # 新增
    float32_array_data: Dict[str, List[Any]],  # 新增
    float32_array_ts: Dict[str, List[float]],  # 新增
    image_compression: bool,
) -> None:
    with h5py.File(hdf5_path, "w") as f:
        if joint_data:
            grp = f.create_group("joint_states")
            for topic, data in joint_data.items():
                topic_safe = topic.replace("/", "_")
                tgrp = grp.create_group(topic_safe)
                tgrp.create_dataset("data", data=np.array(data))
                tgrp.create_dataset("timestamps", data=np.array(joint_ts[topic]))
                tgrp.attrs["topic"] = topic
        if image_data:
            grp = f.create_group("images")
            for topic, images in image_data.items():
                # 使用映射后的名称创建HDF5 group
                hdf5_topic_name = _get_hdf5_topic_name(topic)
                topic_safe = hdf5_topic_name.replace("/", "_")
                tgrp = grp.create_group(topic_safe)
                ts = np.array(image_ts[topic])
                tgrp.create_dataset("timestamps", data=ts)
                if image_compression:
                    vlen = h5py.vlen_dtype(np.uint8)
                    ds = tgrp.create_dataset("data", shape=(len(images),), dtype=vlen)
                    for idx, img in enumerate(images):
                        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        ds[idx] = np.frombuffer(encoded.tobytes(), dtype=np.uint8)
                    tgrp.attrs["compressed"] = True
                else:
                    stack = np.stack(images, axis=0)
                    tgrp.create_dataset("data", data=stack, compression="gzip")
                    tgrp.attrs["compressed"] = False
                tgrp.attrs["topic"] = topic
        if pose_data:  # 修改：直接使用 pose_data
            grp = f.create_group("poses")
            for topic, rows in pose_data.items():  # 修改：直接迭代
                tgrp = grp.create_group(topic.replace("/", "_"))
                arr = np.asarray(rows, dtype="float64")
                tgrp.create_dataset("data", data=arr)  # shape: [M,7]
                tgrp.create_dataset("timestamps", data=np.asarray(pose_ts[topic], dtype="float64"))  # 修改
                tgrp.attrs["topic"] = topic
                tgrp.attrs["type"] = "Pose"
        if foot_switch_data:  # 新增
            grp = f.create_group("foot_switches")
            for topic, data in foot_switch_data.items():
                tgrp = grp.create_group(topic.replace("/", "_"))
                tgrp.create_dataset("data", data=np.array(data, dtype=bool))
                tgrp.create_dataset("timestamps", data=np.array(foot_switch_ts[topic]))
                tgrp.attrs["topic"] = topic
                tgrp.attrs["type"] = "FootSwitch"
        if float32_array_data:  # 新增
            grp = f.create_group("float32_arrays")
            for topic, rows in float32_array_data.items():
                tgrp = grp.create_group(topic.replace("/", "_"))
                # 使用可变长度数组存储不同长度的数据
                vlen = h5py.vlen_dtype(np.float32)
                ds = tgrp.create_dataset("data", shape=(len(rows),), dtype=vlen)
                for idx, row in enumerate(rows):
                    ds[idx] = np.array(row, dtype=np.float32)
                tgrp.create_dataset("timestamps", data=np.array(float32_array_ts[topic], dtype="float64"))
                tgrp.attrs["topic"] = topic
                tgrp.attrs["type"] = "Float32MultiArray"
    print(f"数据已保存到: {hdf5_path}")
    _progress("原始模式:写入完成")


# ---------------------------------------------------------------------------
# 主转换逻辑
# ---------------------------------------------------------------------------

def convert_ros2_bag_to_hdf5(
    bag_path: str,
    hdf5_path: str,
    target_fps: Optional[float] = 500.0,
    image_topics: Optional[List[str]] = None,
    joint_topics: Optional[List[str]] = None,
    image_compression: bool = True,
    max_image_size: Optional[Tuple[int, int]] = None,
    hdfview_compatible: bool = True,
    overwrite: bool = True,
) -> None:
    bag_path_obj = Path(bag_path)
    hdf5_path_obj = Path(hdf5_path)
    _ensure_ros2_bag(bag_path_obj)
    if hdf5_path_obj.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {hdf5_path}")
    hdf5_path_obj.parent.mkdir(parents=True, exist_ok=True)
    _progress(f"开始转换：{bag_path_obj} -> {hdf5_path_obj}")

    detected = detect_topics(bag_path_obj)
    topic_types = detected.pop("types")
    if image_topics:
        images = image_topics
    else:
        images = detected["image"]
    if joint_topics:
        joints = joint_topics
    else:
        joints = detected["joint_state"]
    poses = detected["pose"]                      # 新增
    foot_switches = detected["foot_switch"]  # 新增
    float32_arrays = detected["float32_multiarray"]  # 新增
    print(f"检测到的图像话题: {images}")
    print(f"检测到的关节状态话题: {joints}")
    print(f"检测到的位姿话题: {poses}")         # 可选：提示位姿
    print(f"检测到的踏板话题: {foot_switches}")  # 新增
    print(f"检测到的浮点数组话题: {float32_arrays}")  # 新增

    if target_fps is not None:
        _progress("已选择统一时间轴模式")
        scan = _scan_bag_for_timeline(bag_path_obj, topic_types, images, joints, poses, foot_switches, float32_arrays)  # 传入 float32_arrays
        if scan["start_time"] is None or scan["end_time"] is None:
            raise ValueError("未找到有效的数据")
        timeline_ns = _generate_timeline(scan["start_time"], scan["end_time"], target_fps)
        period_ns = int(1e9 / target_fps)
        f, datasets = _prepare_datasets(
            hdf5_path_obj,
            timeline_ns,
            scan,
            images,
            joints,
            poses,                 # 传入 pose_topics
            foot_switches,  # 传入 foot_switch_topics
            float32_arrays,  # 传入 float32_array_topics
            target_fps,
            hdfview_compatible,
            image_compression,
            max_image_size,
        )
        try:
            _fill_data_with_timeline(
                bag_path_obj,
                topic_types,
                datasets,
                timeline_ns,
                period_ns,
                images,
                joints,
                poses,             # 传入 pose_topics
                foot_switches,  # 传入 foot_switch_topics
                float32_arrays,  # 传入 float32_array_topics
                image_compression,
                max_image_size,
                hdfview_compatible,
            )
        finally:
            f.close()
        print(f"统一时间轴模式完成，输出: {hdf5_path}")
    else:
        _progress("已选择原始时间戳模式")
        joint_data, joint_ts, image_data, image_ts, pose_data, pose_ts, foot_switch_data, foot_switch_ts, float32_array_data, float32_array_ts = _collect_original_mode(
            bag_path_obj,
            topic_types,
            images,
            joints,
            poses,                # 传入 pose_topics
            foot_switches,  # 传入 foot_switch_topics
            float32_arrays,  # 传入 float32_array_topics
            image_compression,
            max_image_size,
        )
        if not joint_data and not image_data and not pose_data and not foot_switch_data and not float32_array_data:
            raise ValueError("未找到任何图像、关节、位姿或浮点数组数据")
        _write_original_mode(
            hdf5_path_obj,
            joint_data,
            joint_ts,
            image_data,
            image_ts,
            pose_data,
            pose_ts,
            foot_switch_data,
            foot_switch_ts,
            float32_array_data,
            float32_array_ts,
            image_compression,
        )
        print(f"原始模式完成，输出: {hdf5_path}")
    _progress("转换流程结束")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS2 原生 bag → HDF5 转换器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认统一时间轴模式 (500Hz)
  %(prog)s <ros2_bag_dir> output.h5

  # 使用原始时间戳模式
  %(prog)s <ros2_bag_dir> output.h5 --original-mode

  # 指定话题、禁用压缩
  %(prog)s <ros2_bag_dir> output.h5 --image-topics /cam --no-compression

  # 仅列出话题
  %(prog)s <ros2_bag_dir> output.h5 --list-topics

  # 批处理模式
  %(prog)s dummy dummy --batch-root /path/to/root --target-fps 30 --no-compression
        """,
    )
    parser.add_argument("bag_path", nargs="?", default=None, help="ROS2 bag 目录或 .db3/.mcap 文件（批处理模式下忽略）")
    parser.add_argument("hdf5_path", nargs="?", default=None, help="输出 HDF5 文件（批处理模式下忽略）")
    parser.add_argument("--target-fps", type=float, default=30.0, help="统一时间轴目标帧率")
    parser.add_argument("--original-mode", action="store_true", help="使用原始时间戳模式")
    parser.add_argument("--image-topics", nargs="*", help="指定图像话题")
    parser.add_argument("--joint-topics", nargs="*", help="指定关节状态话题")
    parser.add_argument("--no-compression", action="store_true", help="禁用 JPEG 压缩")
    parser.add_argument("--max-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="限制图像最大尺寸")
    parser.add_argument("--list-topics", action="store_true", help="仅列出话题，不转换")
    parser.add_argument("--no-hdfview-compatible", dest="hdfview_compatible", action="store_false", help="禁用 HDFView 模式")
    parser.add_argument("--batch-root", type=str, help="批处理根目录，递归查找所有 .db3 并就地生成同名 .h5")
    parser.set_defaults(hdfview_compatible=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # 批处理模式：忽略 bag_path/hdf5_path
    if args.batch_root:
        root = Path(args.batch_root)
        if not root.exists():
            print(f"批处理根目录不存在: {root}")
            return 1
        db3_files = list(root.rglob("*.db3"))
        if not db3_files:
            print(f"在 {root} 下未找到任何 .db3 文件")
            return 0
        print(f"批处理模式：共找到 {len(db3_files)} 个 .db3")
        for db3 in db3_files:
            out_h5 = db3.with_suffix(".h5")
            print(f"\n=== 转换 {db3} -> {out_h5} ===")
            try:
                convert_ros2_bag_to_hdf5(
                    bag_path=str(db3),
                    hdf5_path=str(out_h5),
                    target_fps=None if args.original_mode else args.target_fps,
                    image_topics=args.image_topics,
                    joint_topics=args.joint_topics,
                    image_compression=not args.no_compression,
                    max_image_size=tuple(args.max_size) if args.max_size else None,
                    hdfview_compatible=args.hdfview_compatible,
                )
            except Exception as exc:
                print(f"转换失败: {db3} -> {exc}")
        return 0

    # 单文件/目录模式
    if args.bag_path is None or args.hdf5_path is None:
        print("错误：单文件模式需要指定 bag_path 和 hdf5_path，或使用 --batch-root 进行批处理")
        parser.print_help()
        return 1

    bag_path = Path(args.bag_path)
    try:
        _ensure_ros2_bag(bag_path)
    except Exception as exc:
        print(f"输入校验失败: {exc}")
        return 1

    detected = detect_topics(bag_path)
    if args.list_topics:
        images = detected["image"]
        joints = detected["joint_state"]
        print(f"检测到的图像话题 ({len(images)}):")
        for topic in images:
            print(f"  - {topic}")
        print(f"\n检测到的关节状态话题 ({len(joints)}):")
        for topic in joints:
            print(f"  - {topic}")
        return 0

    target_fps = None if args.original_mode else args.target_fps
    max_size = tuple(args.max_size) if args.max_size else None

    try:
        convert_ros2_bag_to_hdf5(
            bag_path=str(bag_path),
            hdf5_path=args.hdf5_path,
            target_fps=target_fps,
            image_topics=args.image_topics,
            joint_topics=args.joint_topics,
            image_compression=not args.no_compression,
            max_image_size=max_size,
            hdfview_compatible=args.hdfview_compatible,
        )
    except Exception as exc:
        print(f"转换失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
