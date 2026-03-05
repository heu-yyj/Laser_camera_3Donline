import numpy as np
from scipy.spatial.transform import Rotation as R

def read_ply_poses(filename):
    """从PLY文件读取位姿 (位置 + 四元数)"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == 'end_header':
            header_end = i + 1
            break

    vertices_data = []
    for line in lines[header_end:]:
        values = line.strip().split()
        if len(values) >= 9: # Frame, Timestamp, x, y, z, qx, qy, qz, qw
            vertices_data.append(list(map(float, values[2:]))) # 跳过 Frame 和 Timestamp

    vertices_data = np.array(vertices_data)
    positions = vertices_data[:, :3] # x, y, z
    quaternions = vertices_data[:, 3:] # qx, qy, qz, qw (scipy 格式 [x, y, z, w])
    
    return positions, quaternions

def write_ply_poses(filename, positions, quaternions, frames, timestamps):
    """将位姿 (位置 + 四元数 + Frame + Timestamp) 写入PLY文件"""
    num_points = len(positions)
    if num_points != len(quaternions) or num_points != len(frames) or num_points != len(timestamps):
        raise ValueError("Positions, quaternions, frames, and timestamps must have the same length")

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated camera poses\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float Frame\n")
        f.write("property float Timestamp\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float qx\n") # scipy order [x, y, z, w]
        f.write("property float qy\n")
        f.write("property float qz\n")
        f.write("property float qw\n")
        f.write("end_header\n")
        
        for frame, timestamp, pos, quat in zip(frames, timestamps, positions, quaternions):
            f.write(f"{frame} {timestamp} {pos[0]} {pos[1]} {pos[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")

# --- 主流程 ---
auv_ply_file = "D:\\激光试验数据\\20251225水池测试\\AUV_frame_time_pose.ply"  # 替换为你的 AUV PLY 文件路径
cam_ply_file = "D:\\激光试验数据\\20251225水池测试\\camera_poses.ply" # 输出的相机 PLY 文件路径

# 1. 读取 AUV 位姿
auv_positions, auv_quaternions = read_ply_poses(auv_ply_file)
print(f"Read {len(auv_positions)} AUV poses from {auv_ply_file}")

# --- 读取PLY文件时提取Frame和Timestamp ---
with open(auv_ply_file, 'r') as f:
    lines = f.readlines()

header_end = 0
for i, line in enumerate(lines):
    if line.strip() == 'end_header':
        header_end = i + 1
        break

frames = []
timestamps = []
for line in lines[header_end:]:
    values = line.strip().split()
    if len(values) >= 9:
        frames.append(float(values[0]))
        timestamps.append(float(values[1]))

# 2. 定义相机外参 (根据你的最终判断选择)
angles_deg = [199.6, 0, 90.0]
angles_rad = np.deg2rad(angles_deg)
R_z = R.from_euler('z', angles_rad[0])      
R_y = R.from_euler('y', angles_rad[1])     
R_x = R.from_euler('x', angles_rad[2])      

# intrinsic zyx = R_z * R_y * R_x
R_auv2cam = (R_z * R_y * R_x)
R_cam2auv = R_auv2cam.as_matrix().T  # 转置得到相机到AUV的旋转矩阵

T_cam_in_auv =np.array([389.0, 39.8, 405.0])  # 相机在AUV坐标系中的位置 (mm)

print("Using R_cam2auv:")
print(R_cam2auv)
print(f"Using T_cam_in_auv: {T_cam_in_auv}")

# 3. 计算相机位姿
cam_positions = []
cam_quaternions = []

for auv_pos, auv_q in zip(auv_positions, auv_quaternions):
    # 将 AUV 四元数转为旋转矩阵
    R_w = R.from_quat(auv_q).as_matrix() # 四元数顺序是 [x, y, z, w]
    
    # 计算相机位置
    cam_pos_world = R_w @ T_cam_in_auv + auv_pos

    # 计算相机姿态 (旋转矩阵)
    cam_rot_world = R_w @ R_auv2cam.as_matrix()
    # 将相机姿态旋转矩阵转回四元数
    cam_q = R.from_matrix(cam_rot_world).as_quat() # 输出顺序也是 [x, y, z, w]
    
    cam_positions.append(cam_pos_world)
    cam_quaternions.append(cam_q)

cam_positions = np.array(cam_positions)
cam_quaternions = np.array(cam_quaternions)

# 4. 保存相机位姿 PLY 文件
write_ply_poses(cam_ply_file, cam_positions, cam_quaternions, frames, timestamps)
print(f"Saved {len(cam_positions)} camera poses to {cam_ply_file}")