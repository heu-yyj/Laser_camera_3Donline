# -*- coding: utf-8 -*-
"""
读取图像坐标TXT文件夹，按激光+动捕逻辑转换到世界坐标系，并保存为PLY文件
支持批量处理，根据文件名时间戳匹配位姿文件，并生成单帧与总点云。
此版本使用标准库csv模块处理位姿文件，无需pandas。
添加了 ZeroMQ 发送功能，将每帧处理后的点云和位姿信息发送出去。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
from pathlib import Path
import csv # 使用标准库csv模块
import time
import zmq # 导入 ZeroMQ

# ========================= 从原始代码复制的配置区 (保持与实时脚本一致) =========================
# 相机内参矩阵（像素单位）
fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

PIXEL_SIZE = 0.00345              # mm/pixel
f_mm       = fx * PIXEL_SIZE    # 焦距（mm）
BASELINE_S = 220.0              # 激光三角测距基线长度（mm）
A_RAD      = np.deg2rad(19.6)  # 激光发射角（弧度）

# AUV marker灯 长815mm 高80 另一个高40-45  2.8 3.2
# --- 关键修改：使用与实时脚本相同的 angles_deg ---
angles_deg = [199.6, 0, 90.0] # 修改为包含 3.2 度的 Y 轴旋转
# ---
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])
R_y = R.from_euler('y', angles_rad[1])
R_x = R.from_euler('x', angles_rad[2])

# intrinsic zyx = R_z * R_y * R_x
R_A = (R_z * R_y * R_x)
#R_cam2auv = R_A.as_matrix().T  # 转置得到相机到AUV的旋转矩阵

T_cam_in_auv = np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

# --- ZeroMQ 配置 ---
ZMQ_SERVER_IP = "192.168.1.101"  # ZMQ服务器IP10.101.30.58
ZMQ_PORT = 5557  # ZMQ端口
# --------------------

# ===================== 激光处理函数 =====================
def compute_depths(x_coords):
    """根据图像x坐标计算深度"""
    d = (x_coords - cx) * PIXEL_SIZE    # mm
    tanB = d / f_mm
    return BASELINE_S / (np.tan(A_RAD) + tanB)

def image_to_camera(points_img, depths):
    """将图像坐标和深度转换为相机坐标系"""
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt['x'], pt['y'], 1.0])
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm   # mm
    return pts

def camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat):
    """将相机坐标系下的点转换到世界坐标系"""
    # 1. 相机坐标系 -> AUV坐标系
    auv_pts_mm = (R_A.as_matrix() @ cam_pts_mm.T).T + T_cam_in_auv  # 转到AUV坐标系下

    # 2. AUV坐标系 -> 世界坐标系
    r = R.from_quat(auv_quat)  # [x, y, z, w]
    R_w = r.as_matrix()
    world_pts_mm = (R_w @ auv_pts_mm.T).T + auv_pos_mm
    
    # 3. 计算距离（可选，用于调试）
    distances = np.linalg.norm(cam_pts_mm, axis=1)
    return world_pts_mm, distances

# ===================== PLY 文件函数 ======================
def save_points_as_ply(world_pts_mm, filename, dir_name="D:/工作/LaserData"):
    """将世界坐标点保存为PLY文件"""
    total_points = len(world_pts_mm)
    if total_points == 0:
        print(f"[PLY] 没有数据点可以保存。跳过 {filename}")
        return

    output_dir = os.path.join(dir_name, "converted_pointclouds")
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, filename)
    
    with open(full_path, "w", encoding='utf-8') as ply_file:
        ply_file.write("ply\nformat ascii 1.0\n")
        ply_file.write(f"comment Generated @ {datetime.now().isoformat()}\n")
        ply_file.write("comment Unit: millimeter\n")  # 明确标注单位
        ply_file.write(f"element vertex {total_points}\n") # 写入实际点数
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write("end_header\n")
        
        # 写入点数据
        for x, y, z in world_pts_mm:
            ply_file.write(f"{x:.4f} {y:.4f} {z:.4f}\n")

    print(f"[PLY] 文件已保存至：{full_path}（{total_points}点，单位：毫米）")

def read_coordinates_from_txt(file_path):
    """
    从文本文件中读取图像坐标。

    Args:
        file_path (str): 文件路径。

    Returns:
        list of dict: 包含 {'x': float, 'y': float} 的字典列表，用于模拟原始代码的输入格式。
    """
    coordinates = []
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line: # 跳过空行
                    continue
                parts = line.split()
                if len(parts) >= 2:  # 确保有至少两个数值
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coordinates.append({'x': x, 'y': y})
                    except ValueError:
                        print(f"警告：第 {line_num} 行包含非数字数据 '{line}', 已跳过。")
                else:
                    print(f"警告：第 {line_num} 行格式不正确 '{line}', 已跳过。")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return None
    return coordinates

def load_pose_data_csv(pose_file_path):
    """
    使用标准库csv模块加载位姿文件数据 (.csv 或 .tsv)。

    Args:
        pose_file_path (str): 位姿文件路径。

    Returns:
        list of dict: 包含位姿信息的列表，每个元素是一个字典。
                      返回 None 如果加载失败。
    """
    pose_list = []
    try:
        with open(pose_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # 自动检测分隔符，常见的是逗号或制表符
            first_line = csvfile.readline()
            csvfile.seek(0) # 重置文件指针到开头
            if '\t' in first_line:
                delimiter = '\t'
                print("检测到位姿文件使用制表符分隔。")
            elif ',' in first_line:
                delimiter = ','
                print("检测到位姿文件使用逗号分隔。")
            else:
                print("警告：未能自动检测到位姿文件的分隔符，默认使用逗号。")
                delimiter = ','
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            # 检查必要的列是否存在
            required_headers = {'Timestamp', 'XToGlobal1', 'YToGlobal1', 'ZToGlobal1', 
                                'QxToGlobal1', 'QyToGlobal1', 'QzToGlobal1', 'QwToGlobal1'}
            if not required_headers.issubset(set(reader.fieldnames)):
                 missing = required_headers - set(reader.fieldnames)
                 raise KeyError(f"位姿文件缺少必要的列: {missing}")

            for row in reader:
                try:
                    # 将字符串转换为数字
                    entry = {
                        'Timestamp': int(row['Timestamp']), # 读取毫秒级时间戳
                        'XToGlobal1': float(row['XToGlobal1']),
                        'YToGlobal1': float(row['YToGlobal1']),
                        'ZToGlobal1': float(row['ZToGlobal1']),
                        'QxToGlobal1': float(row['QxToGlobal1']),
                        'QyToGlobal1': float(row['QyToGlobal1']),
                        'QzToGlobal1': float(row['QzToGlobal1']),
                        'QwToGlobal1': float(row['QwToGlobal1'])
                    }
                    pose_list.append(entry)
                except (ValueError, KeyError) as e:
                    print(f"警告：位姿文件中有一行数据格式错误，已跳过: {row}, 错误: {e}")
                    continue
            
            # 按时间戳排序
            pose_list.sort(key=lambda x: x['Timestamp'])
            print(f"位姿文件加载成功，共 {len(pose_list)} 条有效记录。")
            return pose_list
            
    except FileNotFoundError:
        print(f"错误：找不到位姿文件 '{pose_file_path}'")
        return None
    except KeyError as ke:
        print(f"加载位姿文件时出错: {ke}")
        return None
    except Exception as e:
        print(f"加载位姿文件时出错: {e}")
        return None

def find_closest_pose_csv(timestamp, pose_list, threshold_ms=10):
    """
    在位姿列表中找到最接近给定时间戳的位姿记录，并检查是否超过阈值。

    Args:
        timestamp (int): 目标时间戳。
        pose_list (list of dict): 排序好的位姿列表。
        threshold_ms (int): 时间戳差异的最大允许值，默认为10ms。

    Returns:
        tuple: (position_array, quaternion_array) 或 (None, None) 如果未找到或超出阈值。
    """
    if not pose_list:
        return None, None

    # 由于列表已排序，可以使用二分查找逻辑
    low, high = 0, len(pose_list) - 1
    closest_idx = -1

    while low <= high:
        mid = (low + high) // 2
        mid_ts = pose_list[mid]['Timestamp']

        if mid_ts == timestamp:
            closest_idx = mid
            break
        elif mid_ts < timestamp:
            low = mid + 1
        else:
            high = mid - 1

    # 此时 high 是小于等于 timestamp 的最大索引，low 是大于 timestamp 的最小索引
    # 需要比较 high 和 low 对应的时间戳
    candidates = []
    if high >= 0:
        candidates.append(high)
    if low < len(pose_list):
        candidates.append(low)

    if not candidates:
        return None, None

    # 选择时间戳差距最小的
    closest_idx = min(candidates, key=lambda i: abs(pose_list[i]['Timestamp'] - timestamp))
    entry = pose_list[closest_idx]
    
    # 检查时间戳差异是否超过阈值
    time_diff_ms = abs(entry['Timestamp'] - timestamp)
    print(f"  - 文件时间戳 {timestamp} -> 匹配位姿时间戳 {entry['Timestamp']} (差值: {time_diff_ms} ms)")
    
    if time_diff_ms > threshold_ms:
        print(f"  - 超过设定的时间戳差异阈值({threshold_ms} ms)，跳过当前帧。")
        return None, None
    
    pos = np.array([entry['XToGlobal1'], entry['YToGlobal1'], entry['ZToGlobal1']])
    quat = np.array([entry['QxToGlobal1'], entry['QyToGlobal1'], entry['QzToGlobal1'], entry['QwToGlobal1']])
    
    return pos, quat

def extract_timestamp_from_filename(filepath):
    """
    从文件名中提取时间戳。

    Args:
        filepath (str): 文件完整路径。

    Returns:
        int: 提取到的时间戳，如果失败则返回 None。
    """
    stem = Path(filepath).stem
    if stem.startswith("image_points_"):
        ts_part = stem[len("image_points_"):]
        try:
            return int(ts_part)
        except ValueError:
            print(f"  - 无法从文件名 '{stem}' 中解析时间戳。")
            return None
    else:
        print(f"  - 文件名 '{stem}' 不符合 'image_points_*.txt' 格式。")
        return None

# --- ZeroMQ 相关函数 ---
def build_pose_matrix(pos_mm, quat):
    """返回 4x4 位姿矩阵（嵌套列表），单位：毫米"""
    r = R.from_quat(quat)
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos_mm  # mm
    # 转为 list of lists，保留 float 类型（JSON 可序列化）
    return T.tolist()  # [[...], [...], [...], [...]]

def initialize_zmq():
    """初始化 ZeroMQ 客户端"""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    # sock.set_hwm(0) # 可选：取消高水位标记
    # sock.set(zmq.CONFLATE, 1) # 可选：只保留最新消息
    sock.connect(f"tcp://{ZMQ_SERVER_IP}:{ZMQ_PORT}")
    print(f"[ZMQ] 已连接 {ZMQ_SERVER_IP}:{ZMQ_PORT}")
    return ctx, sock

def send_zmq_message(sock, world_pts_mm, auv_pos_mm, auv_quat, frame_idx):
    """发送 ZMQ 消息"""
    pose_matrix = build_pose_matrix(auv_pos_mm, auv_quat)  # 4x4 嵌套列表
    points_flat_mm = [round(coord, 4) for coord in world_pts_mm.reshape(-1)] # 毫米单位，扁平列表

    payload = {
        "header": {
            "timestamp": time.time(),
            "frame_id": "lidar",
            "frame_idx": frame_idx,
            "pose": pose_matrix  # 格式: [[...], [...], [...], [...]]
        },
        "points": points_flat_mm  # 毫米单位，扁平列表
    }
    try:
        sock.send_json(payload, flags=zmq.NOBLOCK)
        print(f"[ZMQ] Published Frame {frame_idx:05d} | {len(world_pts_mm)} pts")
    except zmq.Again:
        print(f"[ZMQ Warning] 发送失败，消息队列可能已满 (Frame {frame_idx})")

def finalize_zmq(ctx):
    """关闭 ZeroMQ 客户端"""
    ctx.destroy() # 这会关闭所有相关的 sockets
    print("[ZMQ] 连接已关闭。")

# ... (其他代码不变) ...

def batch_process_images_and_poses(image_folder_path, pose_file_path, output_base_dir="D:/工作/LaserData", zmq_enabled=True):
    """
    批量处理图像点文件夹和位姿文件，生成单帧及总点云PLY文件。
    可选择启用 ZMQ 发送。
    """
    image_folder = Path(image_folder_path)
    if not image_folder.exists():
        print(f"错误：图像文件夹不存在: {image_folder_path}")
        return

    # 1. 加载位姿数据
    pose_data = load_pose_data_csv(pose_file_path)
    if pose_data is None:
        print("位姿文件加载失败，终止处理。")
        return

    # 2. 初始化 ZMQ (如果启用)
    zmq_context = None
    zmq_socket = None
    if zmq_enabled:
        zmq_context, zmq_socket = initialize_zmq()
    else:
        print("[ZMQ] ZMQ 发送已禁用。")

    # 3. 获取所有图像点文件
    txt_files = sorted(list(image_folder.glob("image_points_*.txt")))
    if not txt_files:
        print(f"在文件夹 '{image_folder_path}' 中未找到任何 'image_points_*.txt' 文件。")
        return

    print(f"找到 {len(txt_files)} 个图像点文件，开始批量处理...")

    all_world_points = [] # 存储所有帧的世界坐标点
    frame_idx = 0 # 初始化帧索引

    # 4. 遍历处理每个文件
    for txt_file in txt_files:
        print(f"\n处理文件: {txt_file.name}")

        # 提取时间戳
        file_timestamp_ns = extract_timestamp_from_filename(str(txt_file)) # 提取出的是纳秒级时间戳
        if file_timestamp_ns is None:
            print(f"  - 跳过文件: {txt_file.name} (无法提取时间戳)")
            continue

        # --- 关键修改：将纳秒时间戳转换为毫秒时间戳 ---
        # 假设输入时间戳是纳秒(ns)，目标是毫秒(ms)，则除以 1,000,000
        # 注意：整数除法 // 会向下取整
        file_timestamp_ms = file_timestamp_ns // 1_000_000  
        print(f"  - 原始文件时间戳 (ns): {file_timestamp_ns}, 转换后 (ms): {file_timestamp_ms}")
        # ---

        # 使用转换后的时间戳查找位姿
        auv_pos, auv_quat = find_closest_pose_csv(file_timestamp_ms, pose_data) # 传入毫秒级时间戳
        if auv_pos is None or auv_quat is None:
            print(f"  - 跳过文件: {txt_file.name} (未找到对应位姿 for converted timestamp {file_timestamp_ms})")
            continue

        # 读取图像坐标
        image_points = read_coordinates_from_txt(str(txt_file))
        if image_points is None or not image_points:
            print(f"  - 跳过文件: {txt_file.name} (读取或解析坐标失败或文件为空)")
            continue

        print(f"  - 成功读取 {len(image_points)} 个图像坐标点。")

        # 计算深度
        xs = np.array([p["x"] for p in image_points], dtype=np.float64)
        depths = compute_depths(xs)

        # 图像坐标 -> 相机坐标
        cam_pts_mm = image_to_camera(image_points, depths)

        # 相机坐标 -> 世界坐标
        world_pts_mm, _ = camera_to_world_with_distance(cam_pts_mm, auv_pos, auv_quat)

        # 保存单帧PLY
        frame_timestamp_str = str(file_timestamp_ms) # 使用转换后的毫秒时间戳作为文件名的一部分
        single_frame_filename = f"single_frame_{frame_timestamp_str}.ply"
        save_points_as_ply(world_pts_mm, single_frame_filename, output_base_dir)

        # --- 发送 ZMQ 消息 (如果启用) ---
        if zmq_enabled and zmq_socket:
            send_zmq_message(zmq_socket, world_pts_mm, auv_pos, auv_quat, frame_idx)
        # ---

        # 添加到总列表
        all_world_points.append(world_pts_mm)
        print(f"  - 处理完成，添加 {len(world_pts_mm)} 个点到总点云。")
        frame_idx += 1 # 更新帧索引

    # 5. 合并所有点并保存总PLY文件
    if all_world_points:
        combined_points = np.vstack(all_world_points)
        total_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_filename = f"total_pointcloud_{total_timestamp_str}.ply"
        save_points_as_ply(combined_points, total_filename, output_base_dir)
        print(f"\n--- 批量处理完成 ---")
        print(f"总共处理了 {len(all_world_points)} 个有效帧。")
        print(f"总点云包含 {len(combined_points)} 个点。")
        print(f"总点云文件: {total_filename}")
        print("--------------------")
    else:
        print("\n--- 批量处理完成，但没有有效的点云数据可供保存。---")

    # 6. 关闭 ZMQ (如果启用)
    if zmq_enabled and zmq_context:
        finalize_zmq(zmq_context)


# ... (其余代码不变) ...


if __name__ == "__main__":
    # --- 用户需要修改的部分 ---
    input_image_folder = "D:\\激光试验数据\\20260115水池测试\\LaserRawImagePoints-20260115\\"  # 图像点文件所在的文件夹路径
    input_pose_file = "D:\\激光试验数据\\20260115水池测试\\14\\AUV_pose_4.csv"  # 位姿文件路径，已修正为 .csv
    # --- END 用户需要修改的部分 ---

    output_directory = "D:/工作/LaserData/yewei" # 输出PLY文件的根目录
    enable_zmq = True # 设置为 False 可以禁用 ZMQ 发送

    print(f"开始批量处理...")
    print(f"图像文件夹: {input_image_folder}")
    print(f"位姿文件: {input_pose_file}")
    batch_process_images_and_poses(input_image_folder, input_pose_file, output_directory, zmq_enabled=enable_zmq)
    print("批量处理脚本执行完毕。")