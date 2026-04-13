# auv_zmq_full_history_final.py
# 功能：ZMQ位姿接收 + 完整历史轨迹（所有点保留）+ 严格分段 + 无跨段连线 + 无闪烁 + 自由视角

import sys
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import zmq
import json

# ===== 配置 =====
ZMQ_ADDRESS = "tcp://127.0.0.1:5556"  # 默认连接到本地的处理后数据端口
TARGET_RIGID_BODY_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 保留此参数用于兼容性
AXIS_SCALE = 0.5  # 与你上面的设置保持一致

# ===== 初始化 ZMQ 订阅者 =====
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(ZMQ_ADDRESS)
socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息

print(f"[INFO] 正在连接到 ZMQ 服务器 ({ZMQ_ADDRESS})")
print(f"[INFO] 跟踪位姿数据")

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
vis.create_window("AUV 位姿 - ZMQ数据完整历史轨迹", width=1024, height=768)
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

# 用于存储最新接收到的位姿数据
latest_pose_data = None

try:
    while True:
        # 尝试接收ZMQ消息
        try:
            # 设置非阻塞接收，超时10ms
            if socket.poll(10):  # 10ms超时
                message = socket.recv_string(flags=zmq.NOBLOCK)
                if message:
                    try:
                        pose_data = json.loads(message)
                        latest_pose_data = {
                            "frame": pose_data.get("t", 0),  # 时间戳作为帧号
                            "pos": np.array([pose_data.get("x", 0), pose_data.get("y", 0), pose_data.get("z", 0)]),
                            "quat": np.array([pose_data.get("qw", 1), pose_data.get("qx", 0), pose_data.get("qy", 0), pose_data.get("qz", 0)])
                        }
                        
                        # 验证数据有效性
                        pos = latest_pose_data["pos"]
                        if (abs(pos[0]) > 9999990) or (abs(pos[1]) > 9999990) or (abs(pos[2]) > 9999990):
                            continue  # 跳过异常数据
                        
                        current_frame_id = latest_pose_data["frame"]
                        if current_frame_id == last_frame_id:
                            continue
                        last_frame_id = current_frame_id
                        
                    except json.JSONDecodeError:
                        continue  # 跳过无效JSON
                    except KeyError:
                        continue  # 跳过缺少字段的数据
            else:
                # 没有接收到新数据，继续使用最新的数据
                pass
        except zmq.Again:
            # 没有消息可读
            pass
        
        if latest_pose_data is not None:
            pos = latest_pose_data["pos"]
            quat = latest_pose_data["quat"]
            frame_id = latest_pose_data["frame"]

            print(f"\r[{frame_id}] pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]", end="", flush=True)

            # === 更新 AUV 坐标系：remove + add 新 frame（防闪烁）===
            vis.remove_geometry(auv_frame, reset_bounding_box=False)
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()  # 调整四元数顺序 wxyz -> xyzw
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

        # 必须调用，否则窗口卡死
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n\n[INFO] 用户中断，正在退出...")
finally:
    vis.destroy_window()
    socket.close()
    context.term()