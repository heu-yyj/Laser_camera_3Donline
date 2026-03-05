# -*- coding: utf-8 -*-
"""
激光点云转换与PLY输出
读取CSV中的二维图像坐标点，根据PLY中的AUV位姿，将点云转换到世界坐标系，并输出PLY文件。
PLY位姿每隔4帧读取一次，所有转换后的点云合并到一个PLY文件中。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import csv
from datetime import datetime

# --- 配置区 ---
CSV_PATH = "D:\\激光试验数据\\20251225水池测试\\Video_20251219163908005_000029_points_2d.csv"  # 替换为你的CSV文件路径 (包含x, y列)
PLY_PATH = "D:\\激光试验数据\\20251225水池测试\\AUV_frame_time_pose.ply"                 # 替换为你的PLY文件路径
START_FRAME_INDEX = 10                    # 开始读取位姿的帧索引
FRAME_INTERVAL = 4                          # 读取位姿的间隔 (每隔4帧)
MAX_FRAMES_TO_PROCESS = 15000                 # 最多处理多少个位姿 (避免PLY文件太大时处理太久)
PIXEL_SIZE = 0.00345                        # mm/pixel
BASELINE_S = 220.0                          # 激光三角测距基线长度（mm）
A_RAD      = np.deg2rad(19.6)               # 激光发射角（弧度）

# 相机内参矩阵（像素单位）
# 从您之前的代码中获取
fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# 相机相对于AUV的外参 (旋转矩阵和平移向量)
#R_auv2cam = R.from_euler('zyx', [np.deg2rad(199.6), 0, np.deg2rad(90)]).as_matrix() # 从AUV到相机的旋转
#R_auv2cam = R.from_euler('zyx', [np.deg2rad(180.0), np.deg2rad(-19.6), np.deg2rad(-90)]).as_matrix() # 从AUV到相机的旋转
angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])      
R_y = R.from_euler('y', angles_rad[1])     
R_x = R.from_euler('x', angles_rad[2])      

# intrinsic zyx = R_z * R_y * R_x
R_A = (R_z * R_y * R_x)
R_cam2auv = R_A.as_matrix().T  # 转置得到相机到AUV的旋转矩阵

#R_cam2auv = R.from_euler('zyx', [np.deg2rad(180.0), np.deg2rad(19.6), np.deg2rad(90)]).as_matrix() 
# T_auv2cam = np.array([424.0, 27.4, 247.6]) - np.array([35, -12.4, -157.4]) # 从AUV到相机到的平移 (mm)
T_cam_in_auv =np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

# --- PLY 文件读取函数 ---
def read_poses_from_ply(ply_path, start_frame, interval, max_count):
    """
    从PLY文件中读取特定帧索引及其之后每隔interval帧的AUV位姿。
    返回一个字典 {frame_id: {'pos': pos, 'quat': quat}}
    """
    poses = {}
    try:
        with open(ply_path, 'r') as f:
            line = f.readline()
            while line and 'end_header' not in line:
                line = f.readline()
            
            if not line or 'end_header' not in line:
                print(f"[ERROR] 未在PLY文件 {ply_path} 中找到 'end_header'")
                return poses

            all_lines = f.readlines()
            frame_data = []
            for line in all_lines:
                parts = line.strip().split()
                if len(parts) >= 9:
                    try:
                        frame_id = int(float(parts[0]))
                        pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                        quat = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])
                        quat = quat / np.linalg.norm(quat) # 归一化四元数
                        frame_data.append((frame_id, pos, quat))
                    except (ValueError, IndexError) as e:
                        print(f"[WARNING] 解析PLY行时出错: {line.strip()}, 错误: {e}")
                        continue
            
            # 按帧ID排序
            frame_data.sort(key=lambda x: x[0])
            
            target_frames = set()
            current_frame = start_frame
            processed_count = 0
            while processed_count < max_count:
                target_frames.add(current_frame)
                current_frame += interval
                processed_count += 1

            for frame_id, pos, quat in frame_data:
                if frame_id in target_frames:
                    poses[frame_id] = {'pos': pos, 'quat': quat}
                    print(f"[INFO] 读取到帧 {frame_id} 的位姿。")
                    target_frames.discard(frame_id) # 移除已找到的
                    if not target_frames: # 如果都找到了就停止
                        break
            
            if target_frames:
                print(f"[INFO] 以下帧ID在PLY文件中未找到: {sorted(list(target_frames))}")

        return poses
            
    except FileNotFoundError:
        print(f"[ERROR] PLY文件未找到: {ply_path}")
        return poses
    except Exception as e:
        print(f"[ERROR] 读取PLY文件时发生错误: {e}")
        return poses

# --- CSV 文件读取函数 ---
def read_2d_points_from_csv(csv_path):
    """
    从CSV文件中读取二维激光点坐标。
    CSV文件应包含 'x', 'y' 两列。
    """
    points_2d = []
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    x = float(row['x'])
                    y = float(row['y'])
                    points_2d.append((x, y))
                except (ValueError, KeyError) as e:
                    print(f"[WARNING] 解析CSV行时出错: {row}, 错误: {e}")
                    continue
        print(f"[INFO] 从CSV文件 {csv_path} 读取到 {len(points_2d)} 个二维点。")
        return np.array(points_2d)
    except FileNotFoundError:
        print(f"[ERROR] CSV文件未找到: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 读取CSV文件时发生错误: {e}")
        sys.exit(1)


# --- 深度计算函数 ---
def compute_depths(x_coords):
    """根据像素x坐标计算深度 (mm)"""
    d = (x_coords - cx) * PIXEL_SIZE    # mm
    f_mm = fx * PIXEL_SIZE              # 焦距 (mm)
    tanB = d / f_mm
    depths = BASELINE_S / (np.tan(A_RAD) + tanB)
    return depths


# --- 图像坐标到相机坐标系 ---
def image_to_camera(points_img, depths):
    """将图像坐标和深度转换到相机坐标系 (mm)"""
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt[0], pt[1], 1.0]) # (x, y, 1)
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm   # mm
    return pts


# --- 相机坐标系到AUV世界坐标系 ---
def camera_to_auv_world(cam_pts_mm, auv_pos_mm, auv_quat):
    """将相机坐标系下的点转换到AUV世界坐标系"""
    # 1. 相机 -> AUV本地坐标系
    auv_local_pts_mm = (R_cam2auv @ cam_pts_mm.T).T + T_cam_in_auv
    
    # 2. AUV -> AUV_World (根据当前AUV位姿)
    r = R.from_quat(auv_quat)  # [x, y, z, w]
    R_w = r.as_matrix()
    auv_world_pts_mm = (R_w @ auv_local_pts_mm.T).T + auv_pos_mm 
    return auv_world_pts_mm


# --- PLY 文件写入函数 (ASCII) ---
def write_ply_file_ascii(points, filename):
    """
    将点云写入ASCII格式的PLY文件。
    points: Nx3 numpy数组，包含 (x, y, z) 坐标
    filename: 输出的PLY文件名
    """
    try:
        with open(filename, 'w') as f:
            # PLY Header
            header = f"""ply
format ascii 1.0
comment Generated @ {datetime.now().isoformat()}
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
            f.write(header)
            
            # Write vertex data as ASCII floats
            np.savetxt(f, points, fmt='%.6f', delimiter=' ')
        print(f"  - 点云已保存到PLY文件: {filename}")
    except Exception as e:
        print(f"[ERROR] 保存PLY文件 {filename} 时发生错误: {e}")


def main():
    print("=== 激光点云转换与PLY输出程序 ===")
    
    # 1. 读取CSV中的二维激光点
    print(f"[Step 1] 读取CSV文件: {CSV_PATH}")
    laser_points_2d = read_2d_points_from_csv(CSV_PATH)
    if laser_points_2d.size == 0:
        print("[ERROR] 从CSV文件中读取到的点数为0，无法继续。")
        return

    print(f"  - 读取到 {len(laser_points_2d)} 个二维图像坐标点。")

    # 2. 准备数据并计算深度 (只计算一次)
    print("[Step 2] 计算激光点深度...")
    x_coords = laser_points_2d[:, 0] # 提取x坐标
    y_coords = laser_points_2d[:, 1] # 提取y坐标
    
    print(f"  - 激光点数量: {len(laser_points_2d)}")
    print(f"  - X坐标 - min: {x_coords.min():.2f}, max: {x_coords.max():.2f}, cx: {cx:.2f}")
    print(f"  - Y坐标 - min: {y_coords.min():.2f}, max: {y_coords.max():.2f}, cy: {cy:.2f}")

    depths = compute_depths(x_coords)
    print(f"  - 计算深度 - min: {depths.min():.2f}, max: {depths.max():.2f}, mean: {depths.mean():.2f}")
    print(f"  - 深度 - 负值数量: {(depths < 0).sum()}, 零值数量: {(depths == 0).sum()}, 正值数量: {(depths > 0).sum()}")
    
    # 检查是否有无效深度
    if (depths <= 0).any():
        print("[WARNING] 检测到无效深度 (<= 0)，这些点将导致转换结果无效。")
        # 可以选择过滤这些点，或者继续处理
        # valid_indices = depths > 0
        # laser_points_2d = laser_points_2d[valid_indices]
        # depths = depths[valid_indices]
        # print(f"  - 过滤后剩余 {len(laser_points_2d)} 个有效点。")


    # 3. 图像坐标 -> 相机坐标系 (只转换一次)
    print("[Step 3] 转换到相机坐标系...")
    cam_pts_mm = image_to_camera(laser_points_2d, depths)

    print(f"  - 相机坐标系 - Z坐标 min: {cam_pts_mm[:, 2].min():.2f}, max: {cam_pts_mm[:, 2].max():.2f}, mean: {cam_pts_mm[:, 2].mean():.2f}")
    print(f"  - 相机坐标系 - Z负值数量: {(cam_pts_mm[:, 2] < 0).sum()}")

    # 4. 读取PLY文件中的多个位姿
    print(f"[Step 4] 从PLY文件 {PLY_PATH} 读取位姿 (从帧 {START_FRAME_INDEX} 开始，间隔 {FRAME_INTERVAL})...")
    poses_dict = read_poses_from_ply(PLY_PATH, START_FRAME_INDEX, FRAME_INTERVAL, MAX_FRAMES_TO_PROCESS)

    if not poses_dict:
        print("[ERROR] 未能读取到任何AUV位姿，无法继续。")
        return

    # 5. 对每个读取到的位姿进行坐标转换，并将结果合并
    print(f"[Step 5] 使用同一激光点云，对 {len(poses_dict)} 个AUV位姿进行坐标转换并合并...")
    all_world_points = []
    for frame_id, pose_data in sorted(poses_dict.items()):
        print(f"\n--- 处理帧 {frame_id} ---")
        auv_pos_mm = pose_data['pos']
        auv_quat = pose_data['quat']
        
        # 相机坐标系 -> AUV世界坐标系
        auv_world_pts_mm = camera_to_auv_world(cam_pts_mm, auv_pos_mm, auv_quat)

        print(f"  - AUV位姿 - 位置: [{auv_pos_mm[0]:.2f}, {auv_pos_mm[1]:.2f}, {auv_pos_mm[2]:.2f}]")
        print(f"  - AUV坐标系 - Z坐标 min: {auv_world_pts_mm[:, 2].min():.2f}, max: {auv_world_pts_mm[:, 2].max():.2f}, mean: {auv_world_pts_mm[:, 2].mean():.2f}")
        print(f"  - AUV坐标系 - 相对于AUV位置 (Z) (min, max): {auv_world_pts_mm[:, 2].min() - auv_pos_mm[2]:.2f}, {auv_world_pts_mm[:, 2].max() - auv_pos_mm[2]:.2f}")

        # 将当前位姿转换的点云添加到总列表中
        all_world_points.append(auv_world_pts_mm)
        print(f"  - 添加了 {len(auv_world_pts_mm)} 个点到总点云中。")

    # 6. 合并所有点云并保存到一个PLY文件
    if all_world_points:
        final_point_cloud = np.vstack(all_world_points)
        print(f"\n[Step 6] 合并完成，总共有 {len(final_point_cloud)} 个点。")
        
        # 生成输出PLY文件名
        output_ply_filename = f"merged_laser_points_world.ply"
        print(f"  - 正在将合并后的点云保存到PLY文件: {output_ply_filename}")
        write_ply_file_ascii(final_point_cloud, output_ply_filename)
        
        print("    - 前5个点 (X, Y, Z in mm):")
        for i in range(min(5, len(final_point_cloud))):
            pt = final_point_cloud[i]
            print(f"      {i:3d}: ({pt[0]:8.2f}, {pt[1]:8.2f}, {pt[2]:8.2f})")
        if len(final_point_cloud) > 5:
            print(f"      ... (还有 {len(final_point_cloud) - 5} 个点)")
    else:
        print("\n[Step 6] 没有生成任何点云数据，PLY文件未创建。")


    print("\n=== 点云转换与PLY输出完成 ===")

if __name__ == "__main__":
    main()