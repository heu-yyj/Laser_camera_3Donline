# viewer.py
# 实时可视化：点云 + AUV模型 + AUV坐标系 + 自动跟随视角
import zmq
import numpy as np
import open3d as o3d
import time
import sys

# ==================== ZeroMQ 初始化 ====================
context = zmq.Context()
socket = context.socket(zmq.PULL)
try:
    socket.bind("tcp://127.0.0.1:5557")
    print("[ZMQ] 成功绑定到 tcp://127.0.0.1:5557")
except Exception as e:
    print(f"[ERROR] ZMQ bind 失败: {e}")
    sys.exit(1)

# ==================== 几何体初始化 ====================
# 点云（非空，避免 Open3D 视图异常）
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])
pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]])

# AUV 本体坐标系（动态更新的 LineSet）
auv_coord = o3d.geometry.LineSet()
auv_coord.points = o3d.utility.Vector3dVector([[0, 0, 0]] * 4)
auv_coord.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
auv_coord.colors = o3d.utility.Vector3dVector([
    [1.0, 0.0, 0.0],  # X: 红
    [0.0, 1.0, 0.0],  # Y: 绿
    [0.0, 0.0, 1.0]   # Z: 蓝
])

# 轨迹线
line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0]])
line.lines = o3d.utility.Vector2iVector([[0, 1]])
line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]])

# AUV 小车模型（中心在原点）
box = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.2, depth=0.6)
box.paint_uniform_color([0.2, 0.6, 1.0])
box.translate([-0.15, -0.1, -0.3])  # 使几何中心位于 (0,0,0)

traj = []
first_update = True

# ==================== Open3D 可视化窗口 ====================
vis = o3d.visualization.Visualizer()
vis.create_window("AUV 实时可视化（自动跟随视角）", width=1280, height=720)

# 添加几何体
vis.add_geometry(pcd, reset_bounding_box=True)
vis.add_geometry(auv_coord, reset_bounding_box=False)
vis.add_geometry(line, reset_bounding_box=False)
vis.add_geometry(box, reset_bounding_box=False)

# 渲染设置
opt = vis.get_render_option()
opt.point_size = 2.0
opt.background_color = np.array([0.05, 0.05, 0.1])

# 获取视角控制器
ctr = vis.get_view_control()

print("等待 ZeroMQ 数据...")

# 初始渲染
vis.poll_events()
vis.update_renderer()

# ==================== 主循环 ====================
try:
    while True:
        try:
            # 非阻塞接收数据
            msg = socket.recv_json(flags=zmq.NOBLOCK)
            points = np.array(msg["points"], dtype=np.float32).reshape(-1, 3)
            pose = np.array(msg["header"]["pose"], dtype=np.float64).reshape(4, 4)

            # ---------- 更新点云 ----------
            if len(points) > 0:
                dists = np.linalg.norm(points, axis=1)
                norm = (dists - dists.min()) / (np.ptp(dists) + 1e-6)
                colors = np.zeros_like(points)
                colors[:, 0] = norm          # 红
                colors[:, 2] = 1.0 - norm    # 蓝
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd.points = o3d.utility.Vector3dVector([])
                pcd.colors = o3d.utility.Vector3dVector([])

            # ---------- 更新轨迹 ----------
            pos = pose[:3, 3]
            traj.append(pos.copy())
            if len(traj) > 10000:
                traj.pop(0)
            if len(traj) > 1:
                line.points = o3d.utility.Vector3dVector(traj)
                line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(traj)-1)])
                line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]] * (len(traj)-1))
            else:
                line.points = o3d.utility.Vector3dVector([[0,0,0], [0,0,0]])
                line.lines = o3d.utility.Vector2iVector([[0,1]])

            # ---------- 更新 AUV 模型 ----------
            box_center = box.get_center()
            box.translate(-box_center)
            box.rotate(np.eye(3), center=np.array([0.0, 0.0, 0.0]))
            box.transform(pose)

            # ---------- 更新 AUV 坐标系 ----------
            origin = pose[:3, 3]
            x_end = (pose @ np.array([1.0, 0.0, 0.0, 1.0]))[:3]
            y_end = (pose @ np.array([0.0, 1.0, 0.0, 1.0]))[:3]
            z_end = (pose @ np.array([0.0, 0.0, 1.0, 1.0]))[:3]

            auv_coord.points = o3d.utility.Vector3dVector([origin, x_end, y_end, z_end])
            auv_coord.lines = o3d.utility.Vector2iVector([[0,1], [0,2], [0,3]])
            auv_coord.colors = o3d.utility.Vector3dVector([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])

            # ---------- 自动视角跟随 AUV（关键！）----------
            auv_pos = pose[:3, 3]
            R = pose[:3, :3]  # 旋转矩阵

            # 相机位于 AUV 后方 5 米，上方 1 米（在 AUV 本体系中定义）
            cam_offset_body = np.array([-5.0, 0.0, 1.0])  # 后方5m，高1m
            camera_pos = auv_pos + R @ cam_offset_body

            look_at = auv_pos
            up = np.array([0.0, 0.0, 1.0])  # 世界 Z 轴为上

            # === 修复：使用 numpy array 作为参数 ===
            ctr.set_lookat(look_at)
            ctr.set_front(look_at - camera_pos)  # front = lookat - eye
            ctr.set_up(up)
            ctr.set_zoom(0.05)  # 调整视野（值越小，看得越远）

            # ---------- 更新所有几何体 ----------
            vis.update_geometry(pcd)
            vis.update_geometry(auv_coord)
            vis.update_geometry(line)
            vis.update_geometry(box)

            if first_update:
                vis.reset_view_point(True)
                first_update = False

        except zmq.Again:
            pass

        # 必须调用，否则窗口无响应
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # 控制帧率，降低 CPU 占用

except KeyboardInterrupt:
    print("\n[INFO] 用户中断，正在退出...")
finally:
    vis.destroy_window()
    socket.close()
    context.term()
    print("程序已退出")