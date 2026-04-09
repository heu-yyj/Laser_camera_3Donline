# auv_nokov_full_history_final.py
# 功能：NOKOV 动捕直连 + 完整历史轨迹（所有点保留）+ 严格分段 + 无跨段连线 + 无闪烁 + 自由视角

import sys
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from nokov.nokovsdk import *

# ===== 配置 =====
NOKOV_SERVER_IP = "10.104.21.38"
TARGET_RIGID_BODY_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
AXIS_SCALE = 0.5 # 与你上面的设置保持一致

# ===== 初始化 NOKOV SDK =====
client = PySDKClient()
ret = client.Initialize(bytes(NOKOV_SERVER_IP, "utf-8"))
if ret != 0:
    print(f"[ERROR] 无法连接到 NOKOV 服务器 {NOKOV_SERVER_IP}，错误码: {ret}")
    sys.exit(1)
print(f"[INFO] 已成功连接到 NOKOV 动捕系统 ({NOKOV_SERVER_IP})")
print(f"[INFO] 跟踪刚体 ID: {TARGET_RIGID_BODY_ID}")

# ===== 几何体初始化 =====
# AUV 使用 TriangleMesh（独立，不与 LineSet 共享）
HIDE_POS = np.array([0.0, -1000.0, 0.0])
auv_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SCALE, origin=HIDE_POS)

# 世界坐标系
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# 轨迹相关
traj_line = None
traj_segments = []      # list of segments, each is list of points (np.ndarray)
last_valid_frame_id = None

# ===== Open3D 可视化窗口 =====
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("AUV 位姿 - 完整历史轨迹", width=1024, height=768)
vis.add_geometry(auv_frame, reset_bounding_box=False)
vis.add_geometry(world_frame, reset_bounding_box=False)

opt = vis.get_render_option()
opt.background_color = np.array([0.1, 0.1, 0.1])
opt.line_width = 3.0

def print_camera_info(vis):
    ctr = vis.get_view_control()
    print(f"\n[Camera] Center: {ctr.get_center_of_rotation()}")
    return False

vis.register_key_callback(ord("K"), print_camera_info)

print("\n--- 启动成功 ---")
print("✅ 保留全部历史点 | ✅ 严格分段 | ✅ 无跨段连线 | ✅ 无闪烁黄线")
print("操作: 鼠标控制视角, R=重置视角, K=打印相机参数")

first_data_received = False # 用于首次重置视角
traj_geometry_added = False
last_frame_id = -1

try:
    while True:
        frame = client.PyGetLastFrameOfMocapData()
        if not frame:
            time.sleep(0.01)
            continue

        try:
            d = frame.contents
            current_frame_id = int(d.iFrame)
            if current_frame_id == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = current_frame_id

            target_rb = None
            for i in range(int(d.nRigidBodies)):
                rb = d.RigidBodies[i]
                rb_id = int(rb.ID) & 0xFFF
                if rb_id == TARGET_RIGID_BODY_ID:
                    raw_x, raw_y, raw_z = float(rb.x), float(rb.y), float(rb.z)
                    if (abs(raw_x) > 9999990) or (abs(raw_y) > 9999990) or (abs(raw_z) > 9999990):
                        break
                    x, y, z = raw_x / 1000.0, raw_y / 1000.0, raw_z / 1000.0
                    qx, qy, qz, qw = float(rb.qx), float(rb.qy), float(rb.qz), float(rb.qw)
                    target_rb = {
                        "frame": current_frame_id,
                        "pos": np.array([x, y, z]),
                        "quat": np.array([qx, qy, qz, qw])
                    }
                    break

            if target_rb is not None:
                pos = target_rb["pos"]
                quat = target_rb["quat"]
                frame_id = target_rb["frame"]

                print(f"\r[{frame_id}] pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]", end="", flush=True)

                # === 更新 AUV 坐标系：remove + add 新 frame（防闪烁）===
                vis.remove_geometry(auv_frame, reset_bounding_box=False)
                T = np.eye(4)
                T[:3, :3] = R.from_quat(quat).as_matrix()
                T[:3, 3] = pos.copy()
                auv_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SCALE, origin=[0,0,0])
                auv_frame.transform(T)
                vis.add_geometry(auv_frame, reset_bounding_box=False)

                # === 轨迹分段逻辑 ===
                if (np.linalg.norm(pos) > 1e-6 and 
                    np.all(np.isfinite(pos)) and 
                    np.linalg.norm(pos) < 10.0):

                    # 判断是否新开段
                    should_new_segment = False
                    if last_valid_frame_id is None:
                        should_new_segment = True
                    elif frame_id != last_valid_frame_id + 1:
                        should_new_segment = True

                    if should_new_segment:
                        traj_segments.append([])

                    traj_segments[-1].append(pos.copy())
                    last_valid_frame_id = frame_id

                    # === 构建轨迹：插入 NaN 分隔不同段 ===
                    all_points = []
                    all_lines = []
                    point_offset = 0

                    for segment in traj_segments:
                        n = len(segment)
                        if n < 2:
                            # 单点段：只加点（不连线），但仍计入 offset
                            all_points.extend(segment)
                            point_offset += n
                            continue

                        # 添加真实点
                        all_points.extend([p.copy() for p in segment])

                        # 添加段内连线
                        for i in range(n - 1):
                            all_lines.append([point_offset + i, point_offset + i + 1])
                        point_offset += n

                        # 插入 NaN 分隔符（防止下一段连接）
                        all_points.append(np.array([np.nan, np.nan, np.nan]))
                        point_offset += 1  # NaN 占一个位置

                    # 更新轨迹几何体
                    if len(all_lines) > 0: # 检查 lines 数量，更安全
                        points_vec = o3d.utility.Vector3dVector(all_points)
                        lines_vec = o3d.utility.Vector2iVector(all_lines)
                        colors_vec = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]] * len(all_lines))

                        if not traj_geometry_added:
                            traj_line = o3d.geometry.LineSet()
                            traj_line.points = points_vec
                            traj_line.lines = lines_vec
                            traj_line.colors = colors_vec
                            vis.add_geometry(traj_line, reset_bounding_box=False)
                            traj_geometry_added = True
                        else:
                            # 直接赋值，更新几何体
                            traj_line.points = points_vec
                            traj_line.lines = lines_vec
                            traj_line.colors = colors_vec
                            vis.update_geometry(traj_line)

                    # --- 添加首次重置视角代码 ---
                    if not first_data_received:
                        # 强制重置视角
                        vis.reset_view_point(True)
                        print(f"\n[INFO] 首个有效数据点 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]，已重置视角")
                        first_data_received = True
                    # --- 代码结束 ---

        finally:
            client.PyNokovFreeFrame(frame)

        # 必须调用，否则窗口卡死
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n\n[INFO] 用户中断，正在退出...")
finally:
    vis.destroy_window()
    cleanup_method = getattr(client, 'Finalize', getattr(client, 'Uninitialize', None))
    if cleanup_method:
        try:
            cleanup_method()
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")