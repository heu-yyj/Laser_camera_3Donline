#v9版本：2025-04-09
# -*- coding: utf-8 -*-
import socket
import json
import threading
import time
import numpy as np
from collections import deque
import zmq
from nokov.nokovsdk import PySDKClient
from scipy.spatial.transform import Rotation as R
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
T_cam_in_auv = np.array([389.0 - 50.0, 39.8, 405.0 + 7.0])  # 相机在AUV坐标系中的位置 (mm)

# --- 配置 ---
SYNC_THRESHOLD_MS = 30  # 同步阈值30毫秒

# 用于计算网络延迟的样本数量
DELAY_SAMPLES_COUNT = 10

# 网络延迟相关的配置
LASER_DELAY_ADJUSTMENT_BASE = 300  # 激光延迟基础值(ms)
LASER_DELAY_REDUNDANCY = 300       # 激光延迟冗余值(ms)
MAX_LASER_DELAY = 1000             # 最大激光延迟限制(ms)

# 如果 Nokov 中存在多个刚体，请在这里填写 AUV 对应的 rigid body ID。
TARGET_RIGID_BODY_ID = 0  # 假设刚体ID为1，你可以根据实际情况修改

# 全局状态
running = True
frame_idx = 0
frame_idx_lock = threading.Lock()
pose_lock = threading.Lock()
laser_lock = threading.Lock()

# --- 使用两个独立的缓存 ---
# pose_cache: [(timestamp_ms, pos_mm, quat, arrival_time_ns), ...]
pose_cache = deque()
# laser_cache: [(timestamp_ns, laser_points_list, arrival_time_ns), ...]
laser_cache = deque()

# --- 网络延迟统计 ---
# 用于记录到达延迟的deque
pose_arrival_delays = deque(maxlen=DELAY_SAMPLES_COUNT)
laser_arrival_delays = deque(maxlen=DELAY_SAMPLES_COUNT)

# 用于记录位姿-激光时间差（用于动态调整激光延迟）
pose_laser_time_diffs = deque(maxlen=DELAY_SAMPLES_COUNT)

# 动态激光延迟参数
dynamic_laser_delay_ms = 0
laser_delay_samples = deque(maxlen=DELAY_SAMPLES_COUNT)

first_image_received = False
first_pose_received = False

# --- 控制数据保存的标志 ---
save_data_enabled = False

# --- 保存目录 ---
SAVE_DIR_ROOT = r"D:\工作\LaserRawImagePoints"
RAW_DATA_SAVE_DIR = None
save_dir_init_lock = threading.Lock()
save_dir_initialized = False

# --- PLY文件相关全局变量 ---
current_ply_path = None
ply_file = None
total_points = 0
ply_lock = threading.Lock()

# --- ZeroMQ 初始化 ---
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
    """
    global RAW_DATA_SAVE_DIR, save_dir_initialized
    if not save_data_enabled:
        return

    with save_dir_init_lock:
        if not save_dir_initialized:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            RAW_DATA_SAVE_DIR = os.path.join(SAVE_DIR_ROOT, f"LaserRawImagePoints_Session_{timestamp_str}")
            os.makedirs(RAW_DATA_SAVE_DIR, exist_ok=True)
            print(f"[Init] 已创建原始图像点数据保存目录: '{RAW_DATA_SAVE_DIR}'")
            save_dir_initialized = True

    if RAW_DATA_SAVE_DIR is None:
        print("[ERROR] 保存目录未初始化！")
        return

    if not os.path.exists(RAW_DATA_SAVE_DIR):
        print(f"[ERROR] 保存目录 '{RAW_DATA_SAVE_DIR}' 不存在！")
        return

    file_path = os.path.join(RAW_DATA_SAVE_DIR, f"frame_{frame_counter:05d}_ts_{laser_timestamp_ns}.txt")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for point in points:
                f.write(f"{point['x']}, {point['y']}\n")
    except Exception as e:
        print(f"[Save] 保存图像点数据失败: {e} -> {file_path}")

# ===================== PLY 文件函数 ======================
def create_new_ply_file():
    """创建一个新的 PLY 文件"""
    global current_ply_path, ply_file, total_points
    if not save_data_enabled:
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
    ply_file.write("comment Unit: millimeter\n")
    ply_file.write("element vertex 0\n")
    ply_file.write("property float x\nproperty float y\nproperty float z\n")
    ply_file.write("end_header\n")
    ply_file.flush()
    print(f"[PLY] 保存至：{current_ply_path}（单位：毫米）")

def append_points_to_ply(world_pts_mm):
    """向PLY文件追加点"""
    global total_points
    if not save_data_enabled or ply_file is None:
        return
        
    lines = [f"{x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in world_pts_mm]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points += len(world_pts_mm)

def close_ply_file():
    """关闭PLY文件并更新顶点数量"""
    global ply_file
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
                create_new_ply_file()
                print("[提示] 数据保存已启用。")
            elif key == 'q':  # Q键
                print("\n[退出] 用户按下 'Q' 键，正在关闭程序...")
                os._exit(0)
            else:
                print(f"\n[提示] 按下了 '{key}' 键，无效。请按 'Enter' 开始或 'Q' 退出。")
        time.sleep(0.1)

# ===================== 激光点处理相关函数 =====================
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
    """将相机坐标系下的点转换到世界坐标系"""
    # 1. 将点从相机坐标系转换到AUV坐标系
    auv_pts_mm = (R_auv2cam.as_matrix() @ cam_pts_mm.T).T + T_cam_in_auv  # 转到AUV坐标系下
    
    # 2. 将点从AUV坐标系转换到世界坐标系
    r = R.from_quat(auv_quat)  # [x, y, z, w]
    R_w = r.as_matrix()
    world_pts_mm = (R_w @ auv_pts_mm.T).T + auv_pos_mm 
    
    return world_pts_mm

def build_pose_matrix(pos_mm, quat):
    """返回 4x4 位姿矩阵（嵌套列表），单位：毫米"""
    r = R.from_quat(quat) # [x, y, z, w] -> Rotation object
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos_mm  # mm
    return T.tolist()

def allocate_output_frame_idx():
    """线程安全地分配输出帧号。"""
    global frame_idx
    with frame_idx_lock:
        current_frame_idx = frame_idx
        frame_idx += 1
    return current_frame_idx

def update_dynamic_laser_delay():
    """根据历史数据更新动态激光延迟参数"""
    global dynamic_laser_delay_ms
    
    if len(pose_laser_time_diffs) >= 5:  # 至少需要5个样本才开始计算
        # 计算平均位姿-激光时间差（负值表示激光先到）
        avg_diff_ms = sum(pose_laser_time_diffs) / len(pose_laser_time_diffs)
        
        if avg_diff_ms < 0:  # 激光先到的情况
            # 需要延迟激光数据处理
            laser_delay_needed = abs(avg_diff_ms) + LASER_DELAY_REDUNDANCY
            dynamic_laser_delay_ms = min(laser_delay_needed, MAX_LASER_DELAY)
            laser_delay_samples.append(dynamic_laser_delay_ms)
            print(f"[Delay] 检测到激光先到达，动态调整激光延迟: {dynamic_laser_delay_ms:.1f}ms (基于平均差值: {avg_diff_ms:.1f}ms)")
        else:  # 位姿先到的情况
            # 不需要额外延迟
            dynamic_laser_delay_ms = 0
            print(f"[Delay] 位姿先到达，无需延迟处理")

def try_process_pending_lasers():
    """15fps 安全版 + 未来高帧率友好版
    边遍历边处理，只移除已成功匹配的帧，不清空整个缓存
    """
    ready_frames = []
    to_remove_indices = []   # 记录需要删除的帧在 deque 中的索引

    with laser_lock:
        if not laser_cache:
            return

        # 遍历当前缓存中的每一帧
        for idx, (laser_ts_ns, laser_points, laser_arrival_time_ns) in enumerate(laser_cache):
            laser_ts_ms = laser_ts_ns / 1_000_000.0

            # 如果还没有位姿数据，直接跳出（后续帧也不可能匹配）
            if not pose_cache:
                break

            # 寻找最近的位姿
            pose_ts_list = [item[0] for item in pose_cache]
            nearest_idx = min(range(len(pose_cache)), key=lambda i: abs(pose_ts_list[i] - laser_ts_ms))
            nearest_dt_ms = abs(pose_ts_list[nearest_idx] - laser_ts_ms)

            if nearest_dt_ms <= SYNC_THRESHOLD_MS:
                # 匹配成功
                pose = pose_cache[nearest_idx]
                time_diff_ms = pose[0] - laser_ts_ms
                pose_laser_time_diffs.append(time_diff_ms)

                ready_frames.append((
                    laser_ts_ns,
                    laser_points,
                    pose[0],      # pose_ts_ms
                    pose[1],      # auv_pos_mm
                    pose[2],      # auv_quat
                    nearest_dt_ms
                ))
                to_remove_indices.append(idx)   # 标记为删除
            else:
                # 判断是否太旧需要丢弃
                cache_window_ms = 2000 + dynamic_laser_delay_ms
                if laser_ts_ms < pose_cache[0][0] - cache_window_ms:
                    to_remove_indices.append(idx)   # 标记为丢弃

        # 反向删除（防止索引错位）
        for idx in sorted(to_remove_indices, reverse=True):
            del laser_cache[idx]

    # 在锁外面处理匹配成功的帧（耗时操作）
    for ready_frame in ready_frames:
        process_matched_frame(*ready_frame)

def process_matched_frame(laser_ts_ns, laser_points, pose_ts_ms, auv_pos_mm, auv_quat, sync_delay_ms):
    """处理一对匹配好的激光和位姿数据"""
    global frame_idx
    
    output_frame_idx = allocate_output_frame_idx()
    
    # 保存原始图像点数据
    save_single_frame_image_points(laser_points, laser_ts_ns, output_frame_idx)

    xs = np.array([p["x"] for p in laser_points], dtype=np.float64)
    depths = compute_depths(xs)
    cam_pts_mm = image_to_camera(laser_points, depths)
    world_pts_mm = camera_to_world_with_distance(cam_pts_mm, auv_pos_mm, auv_quat)

    # 构建 ZMQ 消息
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

    # ZMQ 发送
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
    print(f"[Published] Frame {output_frame_idx:05d} | {len(laser_points)} pts | "
          f"sync_delay: {sync_delay_ms:+.1f}ms | 成功匹配 发送结果: {success_count}/{total_count} 个服务端成功")
    
    # 将三维点追加到PLY文件
    append_points_to_ply(world_pts_mm)

# ===================== Nokov 线程 =====================
pose_frame_counter = 0
def nokov_thread():
    global client, first_pose_received, pose_frame_counter
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

                # 只处理目标刚体
                for i in range(data.nRigidBodies):
                    rb = data.RigidBodies[i]
                    if rb.x > 9999990:
                        continue

                    rb_id = getattr(rb, "ID", getattr(rb, "iID", None))
                    if rb_id != TARGET_RIGID_BODY_ID:
                        continue
                    
                    pos_mm = np.array([rb.x, rb.y, rb.z])
                    quat = np.array([rb.qx, rb.qy, rb.qz, rb.qw])
                    quat = quat / np.linalg.norm(quat)

                    # 计算到达延迟并记录
                    sample_ts_ns = int(ts_ms * 1_000_000)  # 转换为纳秒
                    arrival_delay_ns = frame_arrival_ts_ns - sample_ts_ns
                    arrival_delay_ms = arrival_delay_ns / 1_000_000  # 转换为毫秒
                    pose_arrival_delays.append(arrival_delay_ms)

                    # 将新位姿加入缓存
                    with pose_lock:
                        pose_frame_counter += 1
                        pose_cache.append((ts_ms, pos_mm.copy(), quat.copy(), frame_arrival_ts_ns))
                        
                        # 清理过期的位姿数据（根据动态延迟调整窗口）
                        cache_window_ms = 2000  # 默认2秒
                        if dynamic_laser_delay_ms > 0:
                            cache_window_ms += dynamic_laser_delay_ms
                        
                        cutoff_ts_ms = ts_ms - cache_window_ms
                        while pose_cache and pose_cache[0][0] < cutoff_ts_ms:
                            pose_cache.popleft()
                    
                    if not first_pose_received:
                        first_pose_received = True
                        print(f"[NOKOV] 收到第一帧位姿，时间戳: {ts_ms} ms")

                    # 尝试处理积压的激光数据
                    try_process_pending_lasers()
                    break  # 找到目标刚体后退出循环

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

    last_processed_time = time.time()

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
        
        # 检查是否需要延迟处理（根据动态延迟参数）
        current_time = time.time()
        if dynamic_laser_delay_ms > 0:
            time_since_last_process = (current_time - last_processed_time) * 1000  # 转换为毫秒
            remaining_delay = dynamic_laser_delay_ms - time_since_last_process
            if remaining_delay > 0:
                print(f"[Delay] 按计划延迟处理激光数据 {remaining_delay:.1f}ms")
                time.sleep(remaining_delay / 1000.0)  # 转换为秒

        # 更新最后处理时间
        last_processed_time = time.time()
        
        # 计算到达延迟并记录
        arrival_delay_ns = latest_arrival_ts_ns - ts_ns
        arrival_delay_ms = arrival_delay_ns / 1_000_000  # 转换为毫秒
        laser_arrival_delays.append(arrival_delay_ms)

        if not first_image_received:
            first_image_received = True
            print(f"[Laser] 收到第一帧激光点，时间戳: {ts_ns} ns")

        # 将新激光数据加入缓存
        with laser_lock:
            laser_frame_counter += 1
            laser_cache.append((ts_ns, points, latest_arrival_ts_ns))
            
            # 清理过期的激光数据
            cache_window_ms = 2000  # 默认2秒
            if dynamic_laser_delay_ms > 0:
                cache_window_ms += dynamic_laser_delay_ms
                
            cutoff_ts_ns = ts_ns - cache_window_ms * 1_000_000
            while laser_cache and laser_cache[0][0] < cutoff_ts_ns:
                laser_cache.popleft()

        # 尝试处理新加入的激光数据
        try_process_pending_lasers()

    print("[UDP Thread] 退出")

# ===================== 主程序 =====================
if __name__ == "__main__":
    threading.Thread(target=wait_for_start_key, daemon=True).start()
    threading.Thread(target=nokov_thread, daemon=True).start()
    threading.Thread(target=udp_thread, daemon=True).start()  
    
    print("\n=== 激光结构光实时融合系统 + ZMQ发布（动态网络延迟补偿）已启动 ===")
    print(f"ZMQ 数据将发送到以下服务端:")
    for i, server in enumerate(ZMQ_SERVERS):
        print(f"  - 服务端 {i+1}: tcp://{server['ip']}:{server['port']}")
    
    # 启动动态延迟调整监控线程
    def delay_adjustment_thread():
        while running:
            update_dynamic_laser_delay()
            time.sleep(5.0)  # 每5秒更新一次延迟参数
    
    threading.Thread(target=delay_adjustment_thread, daemon=True).start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
        running = False
        time.sleep(2) 
        for socket_instance in zmq_sockets:
            socket_instance.close()
        context.term()
        
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
