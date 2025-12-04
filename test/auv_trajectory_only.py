# auv_trajectory_only.py
# 功能：仅显示 AUV 坐标系 + 轨迹线
# 从 4x4 齐次矩阵中提取位置 + 四元数，用于验证轨迹

import zmq
import numpy as np
import open3d as o3d
import sys
import threading
import queue
from scipy.spatial.transform import Rotation as R

# ===== 配置 =====
MAX_TRAJ_POINTS = 10000  # 最多保留的轨迹点数

# ===== ZeroMQ 接收 =====
context = zmq.Context()
socket = context.socket(zmq.PULL)
try:
    socket.bind("tcp://192.168.5.110:5557")
    print("[ZMQ] 成功绑定到 tcp://192.168.5.110:5557")
    socket.setsockopt(zmq.RCVHWM, 10)
except Exception as e:
    print(f"[ERROR] ZMQ bind 失败: {e}")
    sys.exit(1)

# ===== 几何体初始化 =====
# AUV 坐标系（初始）
auv_coord = o3d.geometry.LineSet()
auv_coord.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
auv_coord.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
auv_coord.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # RGB

# 轨迹线
traj_line = o3d.geometry.LineSet()
traj_line.points = o3d.utility.Vector3dVector([[0, 0, 0]])
traj_line.lines = o3d.utility.Vector2iVector([])
traj_points_list = []  # 存储历史位置 (list of np.array)

# ===== Open3D 可视化 =====
vis = o3d.visualization.Visualizer()
vis.create_window("AUV 轨迹验证", width=1024, height=768)
vis.add_geometry(auv_coord)
vis.add_geometry(traj_line)

opt = vis.get_render_option()
opt.background_color = np.array([0.1, 0.1, 0.1])
opt.line_width = 2.0
vis.reset_view_point(True)

# ===== 数据队列 & 接收线程 =====
data_queue = queue.Queue(maxsize=20)

def data_receiver():
    while True:
        try:
            msg = socket.recv_json()
            data_queue.put(msg)
        except Exception as e:
            print(f"[RECV ERROR] {e}")
            break

threading.Thread(target=data_receiver, daemon=True).start()

print("\n--- AUV 轨迹验证模式启动 ---")
print("正在监听 192.168.5.110:5557 ...")
print("每帧将打印: [frame_id] position(x,y,z) quaternion(x,y,z,w)")

# ===== 主循环 =====
try:
    while True:
        updated = False
        while not data_queue.empty():
            try:
                msg = data_queue.get_nowait()
                pose_flat = np.array(msg["header"]["pose"], dtype=np.float64)
                T = pose_flat.reshape(4, 4)  # 4x4 齐次矩阵

                # === 提取平移（位置）===
                pos = T[:3, 3]  # [x, y, z]

                # === 提取旋转 → 四元数 ===
                rot_matrix = T[:3, :3]
                quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

                frame_id = msg["header"].get("frame_idx", "N/A")
                print(f"[{frame_id}] pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                      f"quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")

                # === 更新 AUV 坐标系 ===
                origin = pos
                R_mat = rot_matrix
                x_end = R_mat @ np.array([1, 0, 0]) + origin
                y_end = R_mat @ np.array([0, 1, 0]) + origin
                z_end = R_mat @ np.array([0, 0, 1]) + origin
                auv_coord.points = o3d.utility.Vector3dVector([origin, x_end, y_end, z_end])
                vis.update_geometry(auv_coord)

                # === 更新轨迹 ===
                traj_points_list.append(origin.copy())
                if len(traj_points_list) > MAX_TRAJ_POINTS:
                    traj_points_list.pop(0)

                if len(traj_points_list) >= 2:
                    traj_line.points = o3d.utility.Vector3dVector(traj_points_list)
                    lines = [[i, i+1] for i in range(len(traj_points_list)-1)]
                    traj_line.lines = o3d.utility.Vector2iVector(lines)
                    traj_line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]] * len(lines))
                    vis.update_geometry(traj_line)

                updated = True

            except Exception as e:
                print(f"[PROCESS ERROR] {e}")
                import traceback
                traceback.print_exc()

        if updated:
            vis.poll_events()
            vis.update_renderer()

        # 小幅休眠避免 CPU 占用过高
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] 用户中断，退出...")
finally:
    vis.destroy_window()
    socket.close()
    context.term()