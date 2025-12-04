# test_send_m_spiral_line_final_right_hand.py
# 发送以 米(m) 为单位的测试数据
# AUV螺旋式左环绕运动 (中心在 (0,0,0))，点云在AUV右侧(Y+)呈竖直线状
# 坐标系约定 (右手坐标系):
#   World (W): +X_Right, +Y_Forward, +Z_Up
#   AUV Body (B): +X_B_Front (motion dir), +Y_B_Right, +Z_B_Down (opposite to +Z_W)
import zmq
import numpy as np
import time
import math

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://127.0.0.1:5557") 

frame_idx = 0
start_time = time.time()

# --- 参数配置 ---
SPIRAL_RADIUS = 3.0           # 螺旋半径 (米)，中心在 (0, 0, 0)
VERTICAL_AMPLITUDE = 1.5      # 垂直方向振幅 (米)
VERTICAL_SPEED = 0.3          # 垂直方向角速度 (弧度/秒)
ANGULAR_SPEED = 0.5           # 水平面角速度 (弧度/秒)，正值为逆时针 (从 +Z_W 轴向下看)
POINTS_PER_FRAME = 1500       # 每帧点数目标
POINT_NOISE_STD = 0.01        # 点云噪声标准差 (米)
# -----------------

def euler_to_rot_matrix(roll, pitch, yaw):
    """计算 ZYX 顺序的欧拉角 (radian) 对应的旋转矩阵 (世界到机体)"""
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy]
    ])
    return R

try:
    while True:
        timestamp = time.time()
        elapsed_time = time.time() - start_time

        # === 1. 计算 AUV 位置 (世界坐标系 W) ===
        # 逆时针圆周运动: x = r*cos(w*t), y = r*sin(w*t)
        angle_xy_rad = elapsed_time * ANGULAR_SPEED
        x_m = SPIRAL_RADIUS * math.cos(angle_xy_rad)
        y_m = SPIRAL_RADIUS * math.sin(angle_xy_rad)
        z_m = VERTICAL_AMPLITUDE * math.sin(elapsed_time * VERTICAL_SPEED)


        x_axis_world = np.array([-math.sin(angle_xy_rad), math.cos(angle_xy_rad), 0.0]) # X_B axis in W
        z_axis_world = np.array([0.0, 0.0, -1.0])                                       # Z_B axis in W
        y_axis_world = np.cross(z_axis_world, x_axis_world)                             # Y_B axis in W

        # 构建旋转矩阵 (世界坐标系向量 -> AUV坐标系向量)
        # R_world_to_body 的每一列是世界坐标系基向量在AUV坐标系中的表示
        R_w_to_b = np.column_stack((x_axis_world, y_axis_world, z_axis_world)).T # Each row is axis vector

        # 构建 4x4 位姿矩阵 (Point_B = Pose^-1 * Point_W, or Point_W = Pose * Point_B)
        # Pose transforms points from B-frame to W-frame
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_w_to_b.T  # R_body_to_world = (R_world_to_body)^T
        pose[:3, 3] = [x_m, y_m, z_m] # Position of B origin in W


        # === 3. 生成点云 (AUV坐标系 B) ===
        # 在 AUV 坐标系下生成点云：沿 AUV 的 +Y_B 轴 (右方) 固定距离，点云竖直，与 Z 轴平行
        local_points_m = np.zeros((POINTS_PER_FRAME, 3))
        local_points_m[:, 0] = 0.0  # X_B: 0 (在AUV Y-Z平面内)
        local_points_m[:, 1] = np.random.uniform(1.0, 1.01, POINTS_PER_FRAME)  # Y_B: 1.0 -> 2.0 (AUV右方)，随机分布以模拟分散性
        local_points_m[:, 2] = np.linspace(-0.5, 0.5, POINTS_PER_FRAME)  # Z_B: -0.5 -> 0.5 (竖直范围)

        # --- 添加微小随机噪声 ---
        noise = np.random.normal(0.0, POINT_NOISE_STD, (POINTS_PER_FRAME, 3))
        noise[:, 1] = 0  # 不给Y_B轴加噪声，保持固定距离
        local_points_m += noise

        # === 4. 转换到世界坐标系 ===
        # Point_W = Pose_Matrix * Point_B (with homogeneous coordinates)
        local_points_homogeneous = np.hstack([local_points_m, np.ones((POINTS_PER_FRAME, 1))])
        # Apply transformation: Pose @ Points_in_B = Points_in_W
        world_points_homogeneous = (pose @ local_points_homogeneous.T).T
        world_points_m = world_points_homogeneous[:, :3]
      

        # === 5. 发送数据 ===
        points_flat = np.round(world_points_m.flatten(), 4).tolist()
        pose_flat = np.round(pose.flatten(), 4).tolist()

        msg = {
            "header": {
                "timestamp": timestamp,
                "frame_id": "laser",
                "frame_idx": frame_idx,
                "pose": pose_flat # 位姿矩阵 (世界坐标系 <- AUV坐标系)
            },
            "points": points_flat # 点云坐标 (世界坐标系)
        }

        sock.send_json(msg)
        print(f"发送帧 {frame_idx:05d}, AUV位置: ({x_m:.2f}, {y_m:.2f}, {z_m:.2f}) m, 点数: {len(points_flat)//3}")

        frame_idx += 1
        time.sleep(0.1) # ~10 FPS

except KeyboardInterrupt:
    print("\n停止发送测试数据")
finally:
    sock.close()
    ctx.term()
    print("ZMQ 资源已释放")