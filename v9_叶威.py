# v9 版本 
# -*- coding: utf-8 -*- 

import socket
import json
import threading
import time
import numpy as np
from collections import deque
from bisect import bisect_left
import zmq
from nokov.nokovsdk import PySDKClient
from scipy.spatial.transform import Rotation as R, Slerp
import os
from datetime import datetime
import sys
import atexit
import msvcrt  # Windows系统专用的库，用于检测按键

# ========================= 配置区 =========================
UDP_IP = "0.0.0.0"
UDP_PORT = 8888
NOKOV_SERVER_IP = "10.104.21.38"

# 添加需要接收点云数据的IP及端口
ZMQ_SERVERS = [
    # {"ip": "10.104.21.145", "port": 5557}, 
    #{"ip": "10.101.30.58", "port": 5557},  
    {"ip": "192.168.5.111", "port": 5556},  
    {"ip": "192.168.5.111", "port": 5557},  
    # {"ip": "10.102.21.110", "port": 5557},  
    {"ip": "192.168.5.110", "port": 5557}, 
    #{"ip": "10.101.30.135", "port": 5557},  


]

# 相机内参矩阵（像素单位）
fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

PIXEL_SIZE = 0.00345
f_mm = fx * PIXEL_SIZE  # 焦距（mm）
BASELINE_S = 220.0      # 激光三角测距基线长度（mm）
A_RAD = np.deg2rad(19.6) # 激光发射角（弧度）

# AUV marker灯 长815mm 高80 另一个高40-45  2.8 3.2
angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])
R_y = R.from_euler('y', angles_rad[1])
R_x = R.from_euler('x', angles_rad[2])
R_auv2cam = (R_z * R_y * R_x)
T_cam_in_auv = np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

# --- 修改配置 ---
# 因为两者频率不同，需要缓存足够长的时间窗口以容纳双方数据
POSE_CACHE_SEC = 2.0  # 缓存位姿2秒
LASER_CACHE_SEC = 10.0 # 缓存激光10秒
SYNC_THRESHOLD_MS = 30 # 同步阈值，超过该值的最近邻位姿不参与重建
SYNC_THRESHOLD_NS = int(SYNC_THRESHOLD_MS * 1_000_000)

# 如果 Nokov 中存在多个刚体，请在这里填写 AUV 对应的 rigid body ID。
# 留空(None)时会继续使用当前看到的所有刚体，但程序会给出告警。
TARGET_RIGID_BODY_ID = None

# 全局状态
POSE_CACHE_SEC = 2.0
LASER_CACHE_SEC = 10.0
SYNC_THRESHOLD_MS = 30
SYNC_THRESHOLD_NS = int(SYNC_THRESHOLD_MS * 1_000_000)
TARGET_RIGID_BODY_ID = None

running = True
frame_idx = 0
frame_idx_lock = threading.Lock()
pose_lock = threading.Lock()
laser_lock = threading.Lock()
matcher_lock = threading.Lock()
timestamp_debug_lock = threading.Lock()

# --- 修改：使用两个独立的缓存 ---
# pose_cache: [(timestamp_ms, pos_mm, quat), ...]
pose_cache = deque()
# laser_cache: [(timestamp_ns, laser_points_list), ...]
laser_cache = deque()

first_image_received = False
first_pose_received = False
multiple_rigid_body_warning_emitted = False
latest_pose_timestamp_debug = None
latest_laser_timestamp_debug = None
last_timestamp_debug_print_monotonic = 0.0

# Timestamp debug: set to False after you capture a few lines of logs.
TIMESTAMP_DEBUG_ENABLED = True
TIMESTAMP_DEBUG_INTERVAL_SEC = 1.0

# --- 新增：控制数据保存的标志 ---
save_data_enabled = False

# --- 修改：保存目录 ---
SAVE_DIR_ROOT = r"D:\工作\LaserRawImagePoints"
# --- 新增：全局变量，用于存储本次运行的主保存目录 ---
RAW_DATA_SAVE_DIR = None
# --- 新增：用于保护全局目录初始化的锁 ---
save_dir_init_lock = threading.Lock()
save_dir_initialized = False

# --- 新增：PLY文件相关全局变量 ---
current_ply_path = None
ply_file = None
total_points = 0
ply_lock = threading.Lock()

# --- 修改 ZeroMQ 初始化 ---
context = zmq.Context()
zmq_sockets = []

for server in ZMQ_SERVERS:
    socket_instance = context.socket(zmq.PUSH)
    socket_instance.set_hwm(0)
    socket_instance.set(zmq.CONFLATE, 1)
    connect_addr = f"tcp://{server['ip']}:{server['port']}"
    socket_instance.connect(connect_addr)
    zmq_sockets.append(socket_instance)
    print(f"[ZMQ] 已连接到服务端 {connect_addr}")

# ===================== 保存单帧原始图像点数据 =====================
def save_single_frame_image_points(points, laser_timestamp_ns, frame_counter):
    """
    为每一帧成功匹配的数据保存点数据到指定的统一目录
    此函数现在会检查 save_data_enabled 标志。
    """
    global RAW_DATA_SAVE_DIR, save_dir_initialized
    if not save_data_enabled:
        # print(f"[Debug] 数据保存未启用，跳过保存时间戳: {laser_timestamp_ns}") # 可选的调试信息
        return

    with save_dir_init_lock:
        if not save_dir_initialized:
            # 如果目录未初始化，则在第一次调用时创建
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            RAW_DATA_SAVE_DIR = os.path.join(SAVE_DIR_ROOT, f"LaserRawImagePoints_Session_{timestamp_str}")
            print(f"[Init] 初始化原始图像点数据保存目录: '{RAW_DATA_SAVE_DIR}'")
            os.makedirs(RAW_DATA_SAVE_DIR, exist_ok=True)
            print(f"[Init] 已创建原始图像点数据保存目录: '{RAW_DATA_SAVE_DIR}'")
            save_dir_initialized = True

    # 使用全局定义的目录
    if RAW_DATA_SAVE_DIR is None:
        print("[ERROR] 保存目录未初始化！")
        return

    # 确保目录存在后再尝试创建文件
    if not os.path.exists(RAW_DATA_SAVE_DIR):
        print(f"[ERROR] 保存目录 '{RAW_DATA_SAVE_DIR}' 不存在！")
        return

    # 文件名包含帧计数器，方便排序
    file_path = os.path.join(RAW_DATA_SAVE_DIR, f"frame_{frame_counter:05d}_ts_{laser_timestamp_ns}.txt")
    
    # 将点数据写入文件，每个点一行，格式为 "x, y"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for point in points:
                f.write(f"{point['x']}, {point['y']}\n")
        # print(f"[Save] 原始图像点数据已保存至: {file_path}") # 移除此行
    except Exception as e:
        print(f"[Save] 保存图像点数据失败: {e} -> {file_path}")


# ===================== PLY 文件函数 ======================
def create_new_ply_file():
    """创建一个新的 PLY 文件"""
    global current_ply_path, ply_file, total_points
    if not save_data_enabled:
        # 如果未启用保存，则PLY文件句柄为空
        current_ply_path = None
        ply_file = None
        total_points = 0
        print("[PLY] 数据保存未启用，PLY文件将不会被创建。")
        return
    
    total_points = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = "D:/工作/LaserData"
    os.makedirs(output_dir, exist_ok=True)
    
    current_ply_path = os.path.join(output_dir, f"laser_pointcloud_{timestamp_str}.ply")
    ply_file = open(current_ply_path, "w", encoding='utf-8')
    
    ply_file.write("ply\nformat ascii 1.0\n")
    ply_file.write(f"comment Generated @ {datetime.now().isoformat()}\n")
    ply_file.write("comment Unit: millimeter\n")  # 明确标注单位
    ply_file.write("element vertex 0\n")
    ply_file.write("property float x\nproperty float y\nproperty float z\n")
    ply_file.write("end_header\n")
    ply_file.flush()
    print(f"[PLY] 保存至：{current_ply_path}（单位：毫米）")

def append_points_to_ply(world_pts_mm):
    """向PLY文件追加点"""
    global total_points
    # 如果未启用保存，则直接返回
    if not save_data_enabled or ply_file is None:
        return
        
    # 直接写入毫米单位（不转换）
    lines = [f"{x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in world_pts_mm]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points += len(world_pts_mm)

def close_ply_file():
    """关闭PLY文件并更新顶点数量"""
    global ply_file
    # 如果未启用保存，则无需操作
    if not save_data_enabled or ply_file is None or ply_file.closed:
        return
        
    ply_file.close()
    try:
        with open(current_ply_path, "r+") as f:
            content = f.read()
            header_end = content.index("end_header\n") + len("end_header\n")
            header, body = content[:header_end], content[header_end:]
            new_header = header.replace("element vertex 0", f"element vertex {total_points}")
            f.seek(0)
            f.write(new_header + body)
            f.truncate()
        print(f"\n[PLY] 已完成：{current_ply_path}（{total_points}点，单位：毫米）")
    except Exception as e:
        print(f"\n[PLY] 文件已保存（更新顶点数失败: {e})")

# 注册程序退出时的回调函数
atexit.register(close_ply_file)

# ===================== 交互式启动函数 ======================
def wait_for_start_key():
    """
    等待用户按回车键以启用数据保存功能
    """
    global save_data_enabled
    print("\n[提示] 程序已启动，正在接收数据...")
    print("[提示] 请按 'Enter' 键开始保存数据，或按 'Q' 键退出程序。")
    
    while running and not save_data_enabled:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()
            if key == '\r':  # Enter键
                print("\n[启动] 用户按下回车键，开始保存数据...")
                save_data_enabled = True
                create_new_ply_file()  # 在这里创建PLY文件
                print("[提示] 数据保存已启用。")
            elif key == 'q':  # Q键
                print("\n[退出] 用户按下 'Q' 键，正在关闭程序...")
                os._exit(0)  # 直接退出，避免阻塞
            else:
                print(f"\n[提示] 按下了 '{key}' 键，无效。请按 'Enter' 开始或 'Q' 退出。")
        time.sleep(0.1) # 避免CPU占用过高

# ===================== 激光点处理相关函数 (修正) =====================
# 以下三个函数已修正为与v5版本一致的正确算法
def compute_depths(x_coords): 
    """根据图像x坐标计算深度"""
    d = (x_coords - cx) * PIXEL_SIZE    # mm
    tanB = d / f_mm
    return BASELINE_S / (np.tan(A_RAD) + tanB)

def image_to_camera(points_img, depths):
    """将图像坐标和深度转换为相机坐标系下的点"""
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt['x'], pt['y'], 1.0])
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm   # mm
    return pts

def camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat):
    """将相机坐标系下的点转换到世界坐标系，并计算距离"""
    # 1. 将点从相机坐标系转换到AUV坐标系
    auv_pts_mm = (R_auv2cam.as_matrix() @ cam_pts_mm.T).T + T_cam_in_auv  # 转到AUV坐标系下
    
    # 2. 将点从AUV坐标系转换到世界坐标系
    r = R.from_quat(auv_quat)  # [x, y, z, w]
    R_w = r.as_matrix()
    world_pts_mm = (R_w @ auv_pts_mm.T).T + auv_pos_mm 
    
    # 3. 计算距离（可选）
    distances = np.linalg.norm(cam_pts_mm, axis=1)
    return world_pts_mm, distances

def build_pose_matrix(pos_mm, quat):
    """返回 4x4 位姿矩阵（嵌套列表），单位：毫米 (与v5版本完全一致)"""
    r = R.from_quat(quat) # [x, y, z, w] -> Rotation object
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos_mm  # mm
    # 转为 list of lists，保留 float 类型（JSON 可序列化）
    return T.tolist()  # [[...], [...], [...], [...]]


def allocate_output_frame_idx():
    """线程安全地分配输出帧号。"""
    global frame_idx
    with frame_idx_lock:
        current_frame_idx = frame_idx
        frame_idx += 1
    return current_frame_idx


def maybe_print_timestamp_debug(sensor_name, frame_num, sample_ts_ns, arrival_ts_ns):
    """
    Periodically print raw sensor timestamps and local receive timestamps.
    Use these lines to verify whether Nokov and laser timestamps share the same time base.
    """
    global latest_pose_timestamp_debug, latest_laser_timestamp_debug, last_timestamp_debug_print_monotonic

    if not TIMESTAMP_DEBUG_ENABLED:
        return

    now_mono = time.monotonic()
    with timestamp_debug_lock:
        current = {
            "frame_num": frame_num,
            "sample_ts_ns": int(sample_ts_ns),
            "arrival_ts_ns": int(arrival_ts_ns)
        }

        if sensor_name == "pose":
            latest_pose_timestamp_debug = current
        elif sensor_name == "laser":
            latest_laser_timestamp_debug = current
        else:
            return

        if (now_mono - last_timestamp_debug_print_monotonic) < TIMESTAMP_DEBUG_INTERVAL_SEC:
            return

        pose_info = latest_pose_timestamp_debug
        laser_info = latest_laser_timestamp_debug
        last_timestamp_debug_print_monotonic = now_mono

    lines = ["[TSDBG] ----------------"]

    if pose_info is not None:
        pose_offset_ms = (pose_info["arrival_ts_ns"] - pose_info["sample_ts_ns"]) / 1_000_000
        lines.append(
            f"[TSDBG] pose  frame={pose_info['frame_num']} "
            f"sample_ns={pose_info['sample_ts_ns']} arrival_ns={pose_info['arrival_ts_ns']} "
            f"arrival_minus_sample_ms={pose_offset_ms:+.3f}"
        )

    if laser_info is not None:
        laser_offset_ms = (laser_info["arrival_ts_ns"] - laser_info["sample_ts_ns"]) / 1_000_000
        lines.append(
            f"[TSDBG] laser frame={laser_info['frame_num']} "
            f"sample_ns={laser_info['sample_ts_ns']} arrival_ns={laser_info['arrival_ts_ns']} "
            f"arrival_minus_sample_ms={laser_offset_ms:+.3f}"
        )

    if pose_info is not None and laser_info is not None:
        sensor_delta_ms = (laser_info["sample_ts_ns"] - pose_info["sample_ts_ns"]) / 1_000_000
        arrival_delta_ms = (laser_info["arrival_ts_ns"] - pose_info["arrival_ts_ns"]) / 1_000_000
        lines.append(
            f"[TSDBG] laser_sample_minus_pose_sample_ms={sensor_delta_ms:+.3f} "
            f"laser_arrival_minus_pose_arrival_ms={arrival_delta_ms:+.3f}"
        )

    print("\n".join(lines))


def interpolate_pose_at_timestamp(target_ts_ns, pose_before, pose_after):
    """在两帧位姿之间插值。"""
    ts0_ns = int(pose_before[0] * 1_000_000)
    ts1_ns = int(pose_after[0] * 1_000_000)
    if ts1_ns <= ts0_ns:
        return None

    alpha = (target_ts_ns - ts0_ns) / (ts1_ns - ts0_ns)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    pos_interp = (1.0 - alpha) * pose_before[1] + alpha * pose_after[1]
    rotations = R.from_quat(np.vstack([pose_before[2], pose_after[2]]))
    quat_interp = Slerp([ts0_ns, ts1_ns], rotations)([target_ts_ns])[0].as_quat()
    return pos_interp, quat_interp


def estimate_pose_for_laser(laser_ts_ns, poses_snapshot):
    """
    为一帧激光估计位姿。
    优先使用前后位姿插值；如果激光已经落后于最新位姿足够久，再退化为最近邻。
    """
    if not poses_snapshot:
        return None

    pose_ts_ns = [int(item[0] * 1_000_000) for item in poses_snapshot]
    newest_pose_ts_ns = pose_ts_ns[-1]
    idx = bisect_left(pose_ts_ns, laser_ts_ns)

    if idx < len(poses_snapshot) and pose_ts_ns[idx] == laser_ts_ns:
        pose = poses_snapshot[idx]
        return {
            "pose_ts_ms": pose[0],
            "auv_pos_mm": pose[1].copy(),
            "auv_quat": pose[2].copy(),
            "sync_delay_ms": 0.0,
            "pose_frame_num": pose[3],
            "match_mode": "exact"
        }

    if 0 < idx < len(poses_snapshot):
        pose_before = poses_snapshot[idx - 1]
        pose_after = poses_snapshot[idx]
        dt_before_ms = (laser_ts_ns - pose_ts_ns[idx - 1]) / 1_000_000
        dt_after_ms = (pose_ts_ns[idx] - laser_ts_ns) / 1_000_000

        if dt_before_ms <= SYNC_THRESHOLD_MS and dt_after_ms <= SYNC_THRESHOLD_MS:
            interpolated_pose = interpolate_pose_at_timestamp(laser_ts_ns, pose_before, pose_after)
            if interpolated_pose is not None:
                auv_pos_mm, auv_quat = interpolated_pose
                return {
                    "pose_ts_ms": laser_ts_ns / 1_000_000,
                    "auv_pos_mm": auv_pos_mm,
                    "auv_quat": auv_quat,
                    "sync_delay_ms": max(dt_before_ms, dt_after_ms),
                    "pose_frame_num": f"{pose_before[3]}->{pose_after[3]}",
                    "match_mode": "interp"
                }

    nearest_idx = min(range(len(poses_snapshot)), key=lambda i: abs(pose_ts_ns[i] - laser_ts_ns))
    nearest_dt_ms = abs(pose_ts_ns[nearest_idx] - laser_ts_ns) / 1_000_000
    has_enough_future_pose = (newest_pose_ts_ns - laser_ts_ns) >= SYNC_THRESHOLD_NS

    if nearest_dt_ms <= SYNC_THRESHOLD_MS and (nearest_dt_ms == 0.0 or has_enough_future_pose):
        pose = poses_snapshot[nearest_idx]
        return {
            "pose_ts_ms": pose[0],
            "auv_pos_mm": pose[1].copy(),
            "auv_quat": pose[2].copy(),
            "sync_delay_ms": nearest_dt_ms,
            "pose_frame_num": pose[3],
            "match_mode": "nearest"
        }

    return None


def try_process_pending_lasers():
    """
    处理待重建激光帧。
    每帧激光最多只处理一次，位姿只按时间过期，不在匹配成功后被消费。
    """
    ready_frames = []
    dropped_frames = []

    with matcher_lock:
        with pose_lock:
            poses_snapshot = list(pose_cache)

        if not poses_snapshot:
            return

        oldest_pose_ts_ns = int(poses_snapshot[0][0] * 1_000_000)

        with laser_lock:
            remaining_lasers = deque()
            for laser_ts_ns, laser_points, laser_frame_num in laser_cache:
                if laser_ts_ns < (oldest_pose_ts_ns - SYNC_THRESHOLD_NS):
                    dropped_frames.append((laser_frame_num, laser_ts_ns))
                    continue

                pose_estimate = estimate_pose_for_laser(laser_ts_ns, poses_snapshot)
                if pose_estimate is None:
                    remaining_lasers.append((laser_ts_ns, laser_points, laser_frame_num))
                    continue

                ready_frames.append((
                    laser_ts_ns,
                    laser_points,
                    pose_estimate["pose_ts_ms"],
                    pose_estimate["auv_pos_mm"],
                    pose_estimate["auv_quat"],
                    pose_estimate["sync_delay_ms"],
                    laser_frame_num,
                    pose_estimate["pose_frame_num"],
                    pose_estimate["match_mode"]
                ))

            laser_cache.clear()
            laser_cache.extend(remaining_lasers)

    for laser_frame_num, laser_ts_ns in dropped_frames:
        print(f"[Sync] Drop laser frame {laser_frame_num}: timestamp {laser_ts_ns} ns 早于当前 pose 缓存窗口")

    for ready_frame in ready_frames:
        process_matched_frame(*ready_frame)


# ===================== 处理匹配帧的函数 =====================
def process_matched_frame(laser_ts_ns, laser_points, pose_ts_ms, auv_pos_mm, auv_quat, sync_delay_ms, laser_frame_num, pose_frame_num, match_mode):
    """处理一对匹配好的激光和位姿数据"""
    return _process_matched_frame_impl(
        laser_ts_ns,
        laser_points,
        pose_ts_ms,
        auv_pos_mm,
        auv_quat,
        sync_delay_ms,
        laser_frame_num,
        pose_frame_num,
        match_mode
    )
    global frame_idx
    
    # (修改) 在处理 Nokov 位姿之前，先保存原始图像点数据
    # 这是唯一调用保存函数的地方，确保只在有位姿时才保存
    save_single_frame_image_points(laser_points, laser_ts_ns, output_frame_idx)

    xs = np.array([p["x"] for p in laser_points], dtype=np.float64)
    depths = compute_depths(xs)
    cam_pts_mm = image_to_camera(laser_points, depths)
    world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat)

    # 构建 ZMQ 消息
    pose_matrix = build_pose_matrix(auv_pos_mm, auv_quat) # 4x4 嵌套列表
    points_flat_mm = [round(coord, 4) for coord in world_pts_mm.reshape(-1)]
    
    payload = {
        "header": {
            "timestamp": time.time(),
            "frame_id": "lidar",
            "frame_idx": output_frame_idx,
            "pose": pose_matrix # 格式: [[...], [...], [...], [...]]
        },
        "points": points_flat_mm # 毫米单位，扁平列表
    }

    # ZMQ 发送逻辑 (保持不变)
    send_results = []
    for i, socket_instance in enumerate(zmq_sockets):
        try:
            socket_instance.send_json(payload, flags=zmq.NOBLOCK)
            send_results.append(f"成功发送到服务端 {i+1}")
        except zmq.Again:
            send_results.append(f"警告: 发送到服务端 {i+1} 失败 (可能队列满)")
        except Exception as e:
            send_results.append(f"错误: 发送到服务端 {i+1} 异常: {e}")
    
    success_count = sum(1 for res in send_results if res.startswith("成功"))
    total_count = len(zmq_sockets)
    print(f"[Published] Frame {frame_idx:05d} | {len(laser_points)} pts | "
          f"sync_delay: {sync_delay_ms:+.1f}ms | 成功匹配 发送结果: {success_count}/{total_count} 个服务端成功")
    
    # --- 新增：将三维点追加到PLY文件 ---
    append_points_to_ply(world_pts_mm)

    frame_idx += 1


# ===================== 配对逻辑 (核心) =====================
def _process_matched_frame_impl(laser_ts_ns, laser_points, pose_ts_ms, auv_pos_mm, auv_quat, sync_delay_ms, laser_frame_num, pose_frame_num, match_mode):
    """实际的激光重建与发布逻辑。"""
    output_frame_idx = allocate_output_frame_idx()

    save_single_frame_image_points(laser_points, laser_ts_ns, output_frame_idx)

    xs = np.array([p["x"] for p in laser_points], dtype=np.float64)
    depths = compute_depths(xs)
    cam_pts_mm = image_to_camera(laser_points, depths)
    world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat)

    pose_matrix = build_pose_matrix(auv_pos_mm, auv_quat)
    points_flat_mm = [round(coord, 4) for coord in world_pts_mm.reshape(-1)]

    payload = {
        "header": {
            "timestamp": time.time(),
            "frame_id": "lidar",
            "frame_idx": output_frame_idx,
            "pose": pose_matrix
        },
        "points": points_flat_mm
    }

    send_results = []
    for i, socket_instance in enumerate(zmq_sockets):
        try:
            socket_instance.send_json(payload, flags=zmq.NOBLOCK)
            send_results.append(f"success {i+1}")
        except zmq.Again:
            send_results.append(f"warn {i+1}")
        except Exception as e:
            send_results.append(f"error {i+1}: {e}")

    success_count = sum(1 for res in send_results if res.startswith("success"))
    total_count = len(zmq_sockets)
    print(f"[Published] Frame {output_frame_idx:05d} | laser#{laser_frame_num} | pose#{pose_frame_num} | "
          f"{match_mode} | {len(laser_points)} pts | sync_delay: {sync_delay_ms:+.1f}ms | "
          f"send {success_count}/{total_count}")

    append_points_to_ply(world_pts_mm)


def match_and_process_data(new_data_type, new_ts, new_data_payload, new_frame_num):
    """
    通用配对函数
    :param new_data_type: 'pose' or 'laser'
    :param new_ts: 时间戳 (ms for pose, ns for laser)
    :param new_data_payload: 对应数据
                             - For 'pose': (pos_mm, quat)
                             - For 'laser': (laser_points_list)
    :param new_frame_num: 新数据的帧号
    """
    try_process_pending_lasers()
    return

    if new_data_type == 'pose':
        # New pose arrived, look for laser data
        auv_pos_mm, auv_quat = new_data_payload
        
        target_cache = laser_cache
        target_lock = laser_lock
        target_time_converter = lambda x: x # Target (laser) ts is in ns
        source_time_converter = lambda x: x * 1_000_000 # Source (pose) ts is in ms, convert to ns
        data_processor = process_matched_frame
        
    elif new_data_type == 'laser':
        # New laser arrived, look for pose data
        laser_points = new_data_payload
        laser_ts_ns = new_ts # Timestamp comes separately
        
        target_cache = pose_cache
        target_lock = pose_lock
        target_time_converter = lambda x: x * 1_000_000 # Target (pose) ts is in ms, convert to ns
        source_time_converter = lambda x: x # Source (laser) ts is in ns
        data_processor = None # We'll call process_matched_frame directly later
    else:
        return

    # Acquire lock for target cache
    with target_lock:
        # Iterate through target cache to find matches
        matched_items = []
        for item_ts, *item_data in target_cache: # Unpack timestamp and rest of the data
            src_ts_ns = source_time_converter(new_ts)
            tgt_ts_ns = target_time_converter(item_ts)
            
            dt_ns = abs(src_ts_ns - tgt_ts_ns)
            dt_ms = dt_ns / 1_000_000

            if dt_ms <= SYNC_THRESHOLD_MS:
                # Also store the frame numbers for logging
                matched_items.append((item_ts, item_data, dt_ms))

        # Remove matched items from target cache
        for ts, data, _ in matched_items:
            try:
                target_cache.remove((ts, *data))
            except ValueError:
                pass # Might have been removed by another thread concurrently

    # Process matched pairs outside the lock to avoid blocking
    for matched_item_ts, matched_item_data, sync_delay_ms in matched_items:
        if new_data_type == 'pose':
            # We had a new pose, matched with an old laser
            # matched_item_data[0] is the points list, matched_item_ts is laser_ts, new_ts is pose_ts
            # matched_item_data[1] is the laser frame number
            laser_frame_num = matched_item_data[1] # Retrieve stored laser frame num
            pose_frame_num = new_frame_num
            # print(f"第 {laser_frame_num} 帧激光数据 (时间戳: {matched_item_ts}, 包含 {len(matched_item_data[0])} 个点) 与 第 {pose_frame_num} 帧位姿数据 (时间戳: {new_ts}, 同步延迟: {sync_delay_ms:+.1f}ms) 成功匹配。") # 移除此行
            process_matched_frame(matched_item_ts, matched_item_data[0], new_ts, *new_data_payload, sync_delay_ms, laser_frame_num, pose_frame_num)
        else: # new_data_type == 'laser'
            # We had a new laser, matched with an old pose
            # matched_item_data[0] is pos, matched_item_data[1] is quat, matched_item_data[2] is pose frame num
            laser_frame_num = new_frame_num
            pose_frame_num = matched_item_data[2] # Retrieve stored pose frame num
            # print(f"第 {laser_frame_num} 帧激光数据 (时间戳: {new_ts}, 包含 {len(new_data_payload)} 个点) 与 第 {pose_frame_num} 帧位姿数据 (时间戳: {matched_item_ts}, 同步延迟: {sync_delay_ms:+.1f}ms) 成功匹配。") # 移除此行
            process_matched_frame(new_ts, new_data_payload, matched_item_ts, *matched_item_data[:2], sync_delay_ms, laser_frame_num, pose_frame_num)


# ===================== Nokov 线程 =====================
pose_frame_counter = 0
def nokov_thread():
    global client, first_pose_received, pose_frame_counter, multiple_rigid_body_warning_emitted
    client = PySDKClient()
    ret = client.Initialize(bytes(NOKOV_SERVER_IP, encoding="utf-8"))
    if ret != 0:
        print(f"[NOKOV] 连接失败: {ret}")
        return
    print("[NOKOV] 已连接，等待第一帧位姿...")
    while running:
        frame = client.PyGetLastFrameOfMocapData()
        if frame:
            try:
                frame_arrival_ts_ns = time.time_ns()
                data = frame.contents
                ts_ms = data.iTimeStamp

                if data.nRigidBodies > 1 and TARGET_RIGID_BODY_ID is None and not multiple_rigid_body_warning_emitted:
                    print(f"[NOKOV] Warning: detected {data.nRigidBodies} rigid bodies. Configure TARGET_RIGID_BODY_ID to avoid mixing poses.")
                    multiple_rigid_body_warning_emitted = True

                for i in range(data.nRigidBodies):
                    rb = data.RigidBodies[i]
                    if rb.x > 9999990:
                        continue

                    rb_id = getattr(rb, "ID", getattr(rb, "iID", None))
                    if TARGET_RIGID_BODY_ID is not None and rb_id != TARGET_RIGID_BODY_ID:
                        continue
                    
                    pos_mm = np.array([rb.x, rb.y, rb.z])
                    quat = np.array([rb.qx, rb.qy, rb.qz, rb.qw])
                    quat = quat / np.linalg.norm(quat)

                    # --- 核心改动：将新位姿加入缓存并尝试配对 ---
                    with pose_lock:
                        pose_frame_counter += 1
                        # Store (ts_ms, pos_mm, quat, frame_num) in cache
                        pose_cache.append((ts_ms, pos_mm.copy(), quat.copy(), pose_frame_counter))
                        # 清理过期的位姿数据
                        cutoff_ts_ms = ts_ms - int(POSE_CACHE_SEC * 1e3)
                        while pose_cache and pose_cache[0][0] < cutoff_ts_ms:
                            pose_cache.popleft()
                    
                    if not first_pose_received:
                        first_pose_received = True
                        print(f"[NOKOV] 收到第一帧位姿，时间戳: {ts_ms} ms")

                    # 调用配对函数，处理新来的位姿
                    # Pass (ts_ms, (pos_mm, quat), frame_num)
                    maybe_print_timestamp_debug(
                        "pose",
                        pose_frame_counter,
                        int(ts_ms) * 1_000_000,
                        frame_arrival_ts_ns
                    )
                    match_and_process_data('pose', ts_ms, (pos_mm, quat), pose_frame_counter)

            finally:
                client.PyNokovFreeFrame(frame)
        else:
            time.sleep(0.001)

# ===================== UDP 主线程 =====================
laser_frame_counter = 0
def udp_thread():
    global first_image_received, laser_frame_counter
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
    print(f"[UDP] 监听 {UDP_IP}:{UDP_PORT}（非阻塞 + 16MB缓冲）")

    while running:
        latest_msg = None
        latest_arrival_ts_ns = None
        while running:
            try:
                data, _ = sock.recvfrom(65535)
                latest_arrival_ts_ns = time.time_ns()
                latest_msg = json.loads(data.decode('utf-8'))
            except BlockingIOError:
                break
            except Exception as e:
                print("UDP解析异常:", e)

        if not latest_msg:
            time.sleep(0.001)
            continue

        points = latest_msg.get("laser_points", [])
        if not points:
            continue

        ts_ns = latest_msg["timestamp"]
        if not first_image_received:
            first_image_received = True
            print(f"[Laser] 收到第一帧激光点，时间戳: {ts_ns} ns")

        # --- 核心改动：将新激光数据加入缓存并尝试配对 ---
        with laser_lock:
            laser_frame_counter += 1
            # Store (ts_ns, points, frame_num) in cache
            laser_cache.append((ts_ns, points, laser_frame_counter))
            # 清理过期的激光数据
            cutoff_ts_ns = ts_ns - int(LASER_CACHE_SEC * 1e9)
            while laser_cache and laser_cache[0][0] < cutoff_ts_ns:
                laser_cache.popleft()

        # 调用配对函数，处理新来的激光
        # Pass (ts_ns, points, frame_num)
        maybe_print_timestamp_debug(
            "laser",
            laser_frame_counter,
            ts_ns,
            latest_arrival_ts_ns if latest_arrival_ts_ns is not None else time.time_ns()
        )
        match_and_process_data('laser', ts_ns, points, laser_frame_counter)

    print("[UDP Thread] 退出")


# ===================== 主程序 (保持不变) =====================
if __name__ == "__main__":
    # --- 新增：启动交互式等待 ---
    threading.Thread(target=wait_for_start_key, daemon=True).start()
    
    threading.Thread(target=nokov_thread, daemon=True).start()
    threading.Thread(target=udp_thread, daemon=True).start()  
    print("\n=== 激光结构光实时融合系统 + ZMQ发布（双向缓存主动配对）已启动 ===")
    print(f"ZMQ 数据将发送到以下服务端:")
    for i, server in enumerate(ZMQ_SERVERS):
        print(f"  - 服务端 {i+1}: tcp://{server['ip']}:{server['port']}")

    try:
        # 主循环现在需要等待 save_data_enabled 变为 True 才创建PLY文件
        # PLY文件的创建由 wait_for_start_key 函数触发
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
        running = False
        time.sleep(2) 
        for socket_instance in zmq_sockets:
            socket_instance.close()
        context.term()
        
        # --- 修改：在程序结束时同时打印原始图像点和PLY文件的路径 ---
        print("\n--- 程序结束 ---")
        if save_data_enabled and RAW_DATA_SAVE_DIR:
            print(f"本次会话生成的原始图像点数据已保存至: {RAW_DATA_SAVE_DIR}")
        elif not save_data_enabled:
            print(f"本次会话未启用数据保存，未生成原始图像点数据文件。")
        else:
            print(f"本次会话未生成原始图像点数据文件。")
        
        if save_data_enabled and current_ply_path:
            print(f"本次会话生成的三维点云数据已保存至: {current_ply_path}")
        elif not save_data_enabled:
            print(f"本次会话未启用数据保存，未生成三维点云数据文件。")
        else:
            print(f"本次会话未生成三维点云数据文件。")
        
        print("资源已释放，程序退出。")
        sys.exit(0)
