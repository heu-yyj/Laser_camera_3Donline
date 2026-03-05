# -*- coding: utf-8 -*-
"""
读取图像坐标TXT文件，按激光+动捕逻辑转换到不同坐标系，并保存为PLY文件
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from datetime import datetime
from collections import deque

# ========================= 从原始代码复制的配置区 =========================
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
angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])
R_y = R.from_euler('y', angles_rad[1])
R_x = R.from_euler('x', angles_rad[2])

# intrinsic zyx = R_z * R_y * R_x
R_A = (R_z * R_y * R_x)
R_cam2auv = R_A.as_matrix().T  # 转置得到相机到AUV的旋转矩阵

T_cam_in_auv = np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

# 位姿信息 - 这里需要手动设置，因为没有实时动捕数据
# 示例：AUV在世界坐标系中的位置和姿态 (单位：毫米, 四元数 [x, y, z, w])
DEFAULT_AUV_POS_MM = np.array([0.0, 0.0, 0.0])  # AUV中心在世界坐标系的位置 (mm)
# 示例四元数 [x, y, z, w]，代表无旋转 (identity rotation)
# DEFAULT_AUV_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
# 示例：绕Z轴旋转90度
DEFAULT_AUV_QUAT_NP = R.from_euler('z', 90, degrees=True).as_quat() # [x, y, z, w]

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
    auv_pts_mm = (R_cam2auv @ cam_pts_mm.T).T + T_cam_in_auv  # 转到AUV坐标系下

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
        print("[PLY] 没有数据点可以保存。")
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


def process_coordinates_and_save_ply(txt_file_path, auv_pos, auv_quat):
    """
    主处理函数：读取坐标，转换，保存PLY
    """
    # 1. 读取图像坐标
    image_points = read_coordinates_from_txt(txt_file_path)
    if image_points is None or not image_points:
        print("读取或解析坐标文件失败或文件为空。")
        return

    print(f"成功读取到 {len(image_points)} 个图像坐标点。")

    # 2. 计算深度
    xs = np.array([p["x"] for p in image_points], dtype=np.float64)
    depths = compute_depths(xs)

    # 3. 图像坐标 -> 相机坐标
    cam_pts_mm = image_to_camera(image_points, depths)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_points_as_ply(cam_pts_mm, f"camera_space_{timestamp_str}.ply", "D:/工作/LaserData/CameraSpace")

    # 4. 相机坐标 -> AUV坐标系
    auv_pts_mm, _ = camera_to_world_with_distance(cam_pts_mm, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    save_points_as_ply(auv_pts_mm, f"auv_space_{timestamp_str}.ply", "D:/工作/LaserData/AUVSpace")

    # 5. AUV坐标系 -> 世界坐标系
    world_pts_mm, distances = camera_to_world_with_distance(cam_pts_mm, auv_pos, auv_quat)
    save_points_as_ply(world_pts_mm, f"world_space_{timestamp_str}.ply", "D:/工作/LaserData/WorldSpace")

    # 6. 打印一些统计信息
    print("\n--- 转换结果统计 ---")
    print(f"输入点数量: {len(image_points)}")
    print(f"输出点数量: {len(world_pts_mm)}")
    if len(world_pts_mm) > 0:
        print(f"世界坐标 X - min: {world_pts_mm[:, 0].min():.2f}, max: {world_pts_mm[:, 0].max():.2f}")
        print(f"世界坐标 Y - min: {world_pts_mm[:, 1].min():.2f}, max: {world_pts_mm[:, 1].max():.2f}")
        print(f"世界坐标 Z - min: {world_pts_mm[:, 2].min():.2f}, max: {world_pts_mm[:, 2].max():.2f}")
        print(f"相机坐标 Z (深度) - min: {cam_pts_mm[:, 2].min():.2f}, max: {cam_pts_mm[:, 2].max():.2f}")
    print("--------------------")


if __name__ == "__main__":
    # --- 用户需要修改的部分 ---
    input_txt_file_path = "D:\\激光试验数据\\20260113水池测试\\LaserRawImagePoints_2\\image_points_1768273693250899696.txt"  # 请将 'your_input_coordinates.txt' 替换为你的实际文件名
    # --- END 用户需要修改的部分 ---

    # 可选：修改默认的AUV位姿
    # auv_position = np.array([100.0, 200.0, -50.0]) # 例如，AUV位于 (100, 200, -50) mm
    # auv_quaternion = np.array([0.0, 0.0, 0.0, 1.0]) # 例如，无旋转
    auv_position = DEFAULT_AUV_POS_MM
    auv_quaternion = DEFAULT_AUV_QUAT_NP # 使用示例中的90度Z轴旋转

    print(f"开始处理文件: {input_txt_file_path}")
    print(f"使用的AUV位姿: 位置={auv_position}, 四元数={auv_quaternion}")
    process_coordinates_and_save_ply(input_txt_file_path, auv_position, auv_quaternion)
    print("处理完成。")