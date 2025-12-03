# -*- coding: utf-8 -*-
"""
终极版：激光 + Nokov + ZeroMQ + Open3D实时可视化（带轨迹+坐标系）
"""

import socket
import json
import threading
import time
import numpy as np
from collections import deque
import zmq
from nokov.nokovsdk import *
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
import signal
import sys
import atexit
import open3d as o3d

# ========================= 配置区 =========================
UDP_IP          = "0.0.0.0"
UDP_PORT        = 8888
NOKOV_SERVER_IP = "192.168.5.110"
ZMQ_SERVER_IP   = "192.168.5.110"
ZMQ_PORT        = 5557

fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

PIXEL_SIZE = 0.00345
f_mm       = fx * PIXEL_SIZE
BASELINE_S = 220.0
A_RAD      = np.deg2rad(19.6)

T_cam2ins = np.array([424.0, 27.4, 247.6])   # 相机相对于惯导中心的平移（mm）

# Marker灯/刚体 相对于惯导中心的平移（mm）
# 比如灯在惯导/INS前方3-4.5cm,取35mm，左边12.4mm，上方12.4mm+100+45mm
T_marker_to_ins = np.array([35, -12.4, -157.4])

T_cam2marker = T_cam2ins - T_marker_to_ins  #  最终外参：相机 → 动捕刚体原点（Marker灯）

 # 旋转部分通常不变（因为你刚体坐标轴和AUV一致） XYZ为AUV前右下
R_cam2marker =  R.from_euler('zyx', [np.deg2rad(199.6), 0, np.deg2rad(90)]).as_matrix() 
R_cam2auv = R_cam2marker
T_cam2auv = T_cam2marker

POSE_CACHE_SEC    = 5.0
SYNC_THRESHOLD_MS = 100
# =========================================================

# 全局状态
running = True
frame_idx = 0
pose_lock = threading.Lock()
pose_cache = deque()
first_image_received = False
first_pose_received = False
current_ply_path = None
ply_file = None
total_points = 0
ply_lock = threading.Lock()

# 实时可视化全局变量（必须定义！）
latest_world_pts = None
latest_auv_pose = None
trajectory_points = []

# ZeroMQ
context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.set_hwm(0)
zmq_socket.set(zmq.CONFLATE, 1)
zmq_socket.connect(f"tcp://{ZMQ_SERVER_IP}:{ZMQ_PORT}")
print(f"[ZMQ] 已连接 {ZMQ_SERVER_IP}:{ZMQ_PORT}")

# ===================== PLY 文件函数（保持不变）=====================
def create_new_ply_file():
    global current_ply_path, ply_file, total_points
    total_points = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_ply_path = f"laser_pointcloud_{timestamp_str}.ply"
    ply_file = open(current_ply_path, "w")
    ply_file.write("ply\nformat ascii 1.0\n")
    ply_file.write(f"comment Generated @ {datetime.now().isoformat()}\n")
    ply_file.write("element vertex 0\n")
    ply_file.write("property float x\nproperty float y\nproperty float z\n")
    ply_file.write("end_header\n")
    ply_file.flush()
    print(f"[PLY] 保存至：{current_ply_path}")

def append_points_to_ply(world_pts):
    global total_points
    lines = [f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in world_pts]
    with ply_lock:
        ply_file.writelines(lines)
        ply_file.flush()
        total_points += len(world_pts)

def close_ply_file():
    global ply_file
    if ply_file and not ply_file.closed:
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
            print(f"\n[PLY] 已完成：{current_ply_path}（{total_points}点）")
        except:
            print(f"\n[PLY] 文件已保存")
atexit.register(close_ply_file)

# ===================== 激光处理函数 =====================
def compute_depths(x_coords):
    d = (x_coords - cx) * PIXEL_SIZE
    tanB = d / f_mm
    return BASELINE_S / (np.tan(A_RAD) + tanB)

def image_to_camera(points_img, depths):
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt['x'], pt['y'], 1.0])
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm
    return pts

def camera_to_world_with_distance(cam_pts_mm, auv_pos, auv_quat):
    auv_pts = (R_cam2auv @ cam_pts_mm.T).T + T_cam2auv
    r = R.from_quat(auv_quat[[1,2,3,0]])
    R_w = r.as_matrix()
    world_pts = (R_w @ auv_pts.T).T + auv_pos
    distances = np.linalg.norm(cam_pts_mm, axis=1)
    
    # 关键：更新全局变量供可视化线程使用
    global latest_world_pts, latest_auv_pose
    latest_world_pts = world_pts.copy()
    latest_auv_pose = (auv_pos.copy(), auv_quat.copy())
    
    return world_pts, distances

def build_pose_matrix(pos, quat):
    r = R.from_quat(quat[[1,2,3,0]])
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3,3] = pos
    return T.reshape(-1).tolist()

# ===================== Nokov & UDP 线程（保持不变）=====================
# （nokov_thread, get_nearest_pose, udp_thread 完全使用你原来的正确版本）
# ===================== Nokov 线程 =====================
def nokov_thread():
    global client, first_pose_received
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
                data = frame.contents
                ts_us = data.iTimeStamp
                for i in range(data.nRigidBodies):
                    rb = data.RigidBodies[i]
                    if rb.x > 9999990: continue
                    pos = np.array([rb.x, rb.y, rb.z])
                    quat = np.array([rb.qw, rb.qx, rb.qy, rb.qz])
                    with pose_lock:
                        pose_cache.append((ts_us, pos.copy(), quat.copy()))
                        cutoff = ts_us - int(POSE_CACHE_SEC * 1e6)
                        while pose_cache and pose_cache[0][0] < cutoff:
                            pose_cache.popleft()
                    if not first_pose_received:
                        first_pose_received = True
                        print(f"[NOKOV] 收到第一帧位姿，时间戳: {ts_us} μs")
            finally:
                client.PyNokovFreeFrame(frame)
        else:
            time.sleep(0.001)
    # client.Uninitialize()

# ===================== 时间戳匹配（关键修复）=====================
def get_nearest_pose(ts_ns: int):
    ts_us = ts_ns // 1_000_000                     # 正确：纳秒 → 微秒
    with pose_lock:
        if not pose_cache:
            return None, None
        ts_arr = np.array([t for t, _, _ in pose_cache])
        idx = np.argmin(np.abs(ts_arr - ts_us))
        dt_us = abs(ts_arr[idx] - ts_us)
        dt_ms = dt_us / 1000.0

        # 即使偏差大一点也继续用（水下常见）
        if dt_ms > SYNC_THRESHOLD_MS:
            print(f"[Warning] 时间戳偏差 {dt_ms:.1f}ms，仍使用最近位姿")

        _, pos, quat = pose_cache[idx]
        return (pos.copy(), quat.copy(), dt_ms), dt_ms

# ===================== UDP 主线程 =====================
def udp_thread():
    global frame_idx, first_image_received
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
    print(f"[UDP] 监听 {UDP_IP}:{UDP_PORT}（非阻塞 + 16MB缓冲）")

    while running:
        latest_msg = None
        while running:
            try:
                data, _ = sock.recvfrom(65535)
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

        if not first_pose_received:
            print("等待 Nokov 位姿就绪...")
            continue

        pose_info, dt_ms = get_nearest_pose(ts_ns)
        if pose_info is None:
            continue

        auv_pos, auv_quat, sync_delay_ms = pose_info

        xs = np.array([p["x"] for p in points], dtype=np.float64)
        depths = compute_depths(xs)
        cam_pts = image_to_camera(points, depths)
        world_pts, distances = camera_to_world_with_distance(cam_pts, auv_pos, auv_quat)

        # 发布 + 保存
        payload = {
            "header": {
                "timestamp": time.time(),
                "frame_id": "laser",
                "frame_idx": frame_idx,
                "pose": build_pose_matrix(auv_pos, auv_quat)
            },
            "points": world_pts.reshape(-1).tolist()
        }
        zmq_socket.send_json(payload, flags=zmq.NOBLOCK)
        append_points_to_ply(world_pts)

        print(f"[Published] Frame {frame_idx:05d} | {len(points)} pts | "
              f"sync_delay: {sync_delay_ms:+.1f}ms | {os.path.basename(current_ply_path)}")

        frame_idx += 1

# ===================== 可视化线程（完美版）=====================
# ===================== 终极可视化线程（永不假死版）=====================
def visualization_thread():
    global vis, pcd, coord_frame, trajectory_line, auv_mesh, running, latest_world_pts, latest_auv_pose, trajectory_points

    # 1. 创建窗口（提前创建，让用户看到“正在启动”）
    vis = o3d.visualization.Visualizer()
    vis.create_window("激光结构光实时点云 + AUV轨迹（启动中...）", width=1600, height=1000)

    # 初始几何体
    pcd = o3d.geometry.PointCloud()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    trajectory_line = o3d.geometry.LineSet()
    auv_mesh = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.2, depth=0.6)
    auv_mesh.paint_uniform_color([0.2, 0.6, 1.0])

    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.add_geometry(trajectory_line)
    vis.add_geometry(auv_mesh)

    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.asarray([0.05, 0.05, 0.1])

    print("[Open3D] 可视化窗口已启动（等待数据...）")

    # 2. 等待第一帧有效数据（关键！）
    print("   等待激光点和动捕位姿就绪...")
    while running:
        if latest_world_pts is not None and len(latest_world_pts) > 100 and latest_auv_pose is not None:
            vis.get_view_control().set_front([0, 0, -1])   # 设置初始视角
            vis.get_view_control().set_lookat([0, 0, 0])
            vis.get_view_control().set_up([0, -1, 0])
            vis.get_view_control().set_zoom(0.5)
            print("[Open3D] 数据就绪，开始实时显示！")
            break
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

    # 3. 主循环（丝滑不卡版）
    while running:
        need_update = False

        # 更新点云
        if latest_world_pts is not None and len(latest_world_pts) > 100:
            pcd.points = o3d.utility.Vector3dVector(latest_world_pts)
            dists = np.linalg.norm(latest_world_pts, axis=1)
            norm = (dists - dists.min()) / (dists.ptp() + 1e-6)
            colors = np.zeros((len(dists), 3))
            colors[:, 0] = norm
            colors[:, 2] = 1 - norm
            pcd.colors = o3d.utility.Vector3dVector(colors)
            need_update = True

        # 更新轨迹 + AUV姿态
        if latest_auv_pose is not None:
            pos, quat = latest_auv_pose
            trajectory_points.append(pos.copy())
            if len(trajectory_points) > 10000:
                trajectory_points.pop(0)

            if len(trajectory_points) > 1:
                points_vec = o3d.utility.Vector3dVector(trajectory_points)
                lines = [[i, i+1] for i in range(len(trajectory_points)-1)]
                trajectory_line.points = points_vec
                trajectory_line.lines = o3d.utility.Vector2iVector(lines)
                trajectory_line.colors = o3d.utility.Vector3dVector([[1, 1, 1]] * len(lines))
                need_update = True

            # 更新AUV小车
            T = np.eye(4)
            r = R.from_quat(quat[[1,2,3,0]])
            T[:3,:3] = r.as_matrix()
            T[:3,3] = pos
            auv_mesh.translate(-auv_mesh.get_center())
            auv_mesh.rotate(np.eye(3), center=False)
            auv_mesh.transform(T)
            need_update = True

        if need_update:
            for geom in [pcd, trajectory_line, auv_mesh]:
                vis.update_geometry(geom)

        # 关键！让Open3D有时间响应鼠标/键盘/关闭事件
        if not vis.poll_events():
            break  # 窗口关闭时优雅退出
        vis.update_renderer()

        time.sleep(0.001)  # 超高刷新率，但不卡！

    vis.destroy_window()
    print("[Open3D] 可视化窗口已关闭")

# ===================== 主
# 程序（关键修改）=====================
if __name__ == "__main__":
    create_new_ply_file()
    
    # 启动所有线程 
    threading.Thread(target=nokov_thread, daemon=False).start()
    threading.Thread(target=udp_thread, daemon=False).start()
    time.sleep(2.0)  # 确保数据线程先跑起来
    threading.Thread(target=visualization_thread, daemon=False).start()  # 最后启动可视化

    print("\n=== 激光结构光实时融合系统 + Open3D可视化 已启动 ===")
    print("点云 + 轨迹 + 坐标系 + AUV姿态 实时显示！\n")

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        time.sleep(2)
        zmq_socket.close()
        context.term()
        sys.exit(0)