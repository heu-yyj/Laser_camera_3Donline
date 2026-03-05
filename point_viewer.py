# viewer_optimized.py
# 实时可视化：米单位点云 + AUV模型 + AUV坐标系 (支持交互缩放/平移/旋转)
# 功能：显示历史点云累积 + 水池环境 + 世界坐标系
# 优化：使用队列解耦数据接收与渲染，批处理点云更新
import zmq
import numpy as np
import open3d as o3d
import time
import sys
import threading
import queue
import collections

# ===== 配置参数 =====
FOLLOW_AUV = False
MAX_HISTORY_POINTS = 500000 # 最多保留的历史点数
BATCH_UPDATE_SIZE = 10      # 累积多少帧后更新一次历史点云 (优化项)
# ==================== ZeroMQ 初始化 ====================
context = zmq.Context()
socket = context.socket(zmq.PULL)
try:
    socket.bind("tcp://192.168.1.101:5557")
    print("[ZMQ] 成功绑定到 tcp://192.168.1.101:5557")
    socket.setsockopt(zmq.RCVHWM, 10) # 设置接收缓冲区上限，防止内存暴涨
    socket.setsockopt(zmq.SNDHWM, 10)
except Exception as e:
    print(f"[ERROR] ZMQ bind 失败: {e}")
    sys.exit(1)

# ==================== 辅助函数 (不变) ====================
# ... (create_water_tank, create_world_coordinate_system 保持不变) ...

def create_water_tank(width_m, length_m, height_m, grid_spacing_m):
    """
    创建表示水池的线条几何体。
    """
    lineset = o3d.geometry.LineSet()
    points = []
    lines = []
    colors = []

    w_half = width_m / 2.0
    l_half = length_m / 2.0
    h = height_m
    gs = grid_spacing_m

    # --- 底部网格 ---
    x_lines_base = np.arange(-w_half, w_half + gs/2, gs)
    y_lines_base = np.arange(-l_half, l_half + gs/2, gs)
    
    for y_val in y_lines_base:
        start_idx = len(points)
        points.extend([[x, y_val, 0] for x in x_lines_base])
        end_idx = len(points) - 1
        lines.extend([[i, i+1] for i in range(start_idx, end_idx)])
        
    for x_val in x_lines_base:
        start_idx = len(points)
        points.extend([[x_val, y, 0] for y in y_lines_base])
        end_idx = len(points) - 1
        lines.extend([[i, i+1] for i in range(start_idx, end_idx)])

    # --- 侧壁网格 ---
    corners_bottom = [
        [-w_half, -l_half, 0], [w_half, -l_half, 0],
        [w_half, l_half, 0], [-w_half, l_half, 0]
    ]
    corners_top = [[x, y, h] for x, y, _ in corners_bottom]

    all_corners = corners_bottom + corners_top
    base_corner_idx = len(points)
    points.extend(all_corners)

    lines.extend([
        [base_corner_idx + 0, base_corner_idx + 1],
        [base_corner_idx + 1, base_corner_idx + 2],
        [base_corner_idx + 2, base_corner_idx + 3],
        [base_corner_idx + 3, base_corner_idx + 0],
        [base_corner_idx + 4, base_corner_idx + 5],
        [base_corner_idx + 5, base_corner_idx + 6],
        [base_corner_idx + 6, base_corner_idx + 7],
        [base_corner_idx + 7, base_corner_idx + 4],
        [base_corner_idx + 0, base_corner_idx + 4],
        [base_corner_idx + 1, base_corner_idx + 5],
        [base_corner_idx + 2, base_corner_idx + 6],
        [base_corner_idx + 3, base_corner_idx + 7],
    ])

    num_vertical_lines_x = int(width_m / grid_spacing_m) + 1
    num_vertical_lines_y = int(length_m / grid_spacing_m) + 1

    for i in range(num_vertical_lines_x):
        x_val = -w_half + i * gs
        p1_idx = len(points)
        points.append([x_val, -l_half, 0])
        p2_idx = len(points)
        points.append([x_val, -l_half, h])
        lines.append([p1_idx, p2_idx])
        
        p3_idx = len(points)
        points.append([x_val, l_half, 0])
        p4_idx = len(points)
        points.append([x_val, l_half, h])
        lines.append([p3_idx, p4_idx])

    for i in range(num_vertical_lines_y):
        y_val = -l_half + i * gs
        p1_idx = len(points)
        points.append([-w_half, y_val, 0])
        p2_idx = len(points)
        points.append([-w_half, y_val, h])
        lines.append([p1_idx, p2_idx])
        
        p3_idx = len(points)
        points.append([w_half, y_val, 0])
        p4_idx = len(points)
        points.append([w_half, y_val, h])
        lines.append([p3_idx, p4_idx])

    z_levels = np.arange(gs, h, gs)
    for z_val in z_levels:
        p1_idx = len(points)
        points.append([-w_half, -l_half, z_val])
        p2_idx = len(points)
        points.append([w_half, -l_half, z_val])
        lines.append([p1_idx, p2_idx])
        p3_idx = len(points)
        points.append([-w_half, l_half, z_val])
        p4_idx = len(points)
        points.append([w_half, l_half, z_val])
        lines.append([p3_idx, p4_idx])
    for z_val in z_levels:
        p1_idx = len(points)
        points.append([-w_half, -l_half, z_val])
        p2_idx = len(points)
        points.append([-w_half, l_half, z_val])
        lines.append([p1_idx, p2_idx])
        p3_idx = len(points)
        points.append([w_half, -l_half, z_val])
        p4_idx = len(points)
        points.append([w_half, l_half, z_val])
        lines.append([p3_idx, p4_idx])

    colors = [[0.5, 0.5, 0.5] for _ in range(len(lines))] 

    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return lineset

def create_world_coordinate_system(axis_length_m=3.0):
    """
    创建一个世界坐标系 LineSet。
    Args:
       axis_length_m (float): 坐标轴长度（米）。
    """
    coord = o3d.geometry.LineSet()
    axis_len = axis_length_m 
    
    points = [
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] # XYZ -> RGB
    
    coord.points = o3d.utility.Vector3dVector(points)
    coord.lines = o3d.utility.Vector2iVector(lines)
    coord.colors = o3d.utility.Vector3dVector(colors)
    return coord

# ==================== 几何体初始化 (不变) ====================
TANK_WIDTH_M = 10.0; TANK_LENGTH_M = 10.0; TANK_HEIGHT_M = 6.0; GRID_SPACING_M = 1.0
tank_lineset = create_water_tank(TANK_WIDTH_M, TANK_LENGTH_M, TANK_HEIGHT_M, GRID_SPACING_M)
world_coord = create_world_coordinate_system(axis_length_m=5.0)

# --- 历史点云: 使用 NumPy 数组缓冲区 (优化项) ---
history_buffer_points = np.empty((MAX_HISTORY_POINTS, 3), dtype=np.float32)
history_buffer_colors = np.empty((MAX_HISTORY_POINTS, 3), dtype=np.float32)
buffer_count = 0
history_pcd = o3d.geometry.PointCloud() # 实际用于可视化的点云对象
history_pcd_added_to_vis = False

# --- AUV相关 (不变) ---
auv_coord = o3d.geometry.LineSet()
auv_coord.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
auv_coord.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
auv_coord.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0]])
line.lines = o3d.utility.Vector2iVector([[0, 1]])
line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]])
width_m, height_m, depth_m = 0.3, 0.2, 0.6
box = o3d.geometry.TriangleMesh.create_box(width=width_m, height=height_m, depth=depth_m)
box.paint_uniform_color([0.2, 0.6, 1.0])
box.translate([-width_m/2, -height_m/2, -depth_m/2])

traj_points_list = []

# ==================== Open3D 可视化窗口 (不变) ====================
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("AUV 实时可视化 (优化版)", width=1280, height=720)
static_geometries = [tank_lineset, world_coord, auv_coord, line, box]
for geom in static_geometries:
    vis.add_geometry(geom, reset_bounding_box=False)

opt = vis.get_render_option()
opt.point_size = 1.0
opt.line_width = 2.0
opt.background_color = np.array([0.1, 0.1, 0.1])
opt.show_coordinate_frame = False
print("[INFO] 正在设置初始视角...")
vis.reset_view_point(True) 
print("[INFO] 初始视角设置完成.")
print(f"\n--- 可视化环境已准备就绪 (优化版) ---")
print("提示: ")
print("  - 使用鼠标左键拖拽旋转视角")
print("  - 使用鼠标右键拖拽平移视角")
print("  - 使用鼠标滚轮缩放")
print("  - 按 'R' 或 'Ctrl+R' 重置视角")
print("  - 按 'C' 切换相机跟随模式")
print("--------------------------------\n")

# ==================== 回调函数 (不变) ====================
def toggle_follow_mode(vis):
    global FOLLOW_AUV
    FOLLOW_AUV = not FOLLOW_AUV
    print(f"[INFO] 相机跟随模式已切换为: {'开启' if FOLLOW_AUV else '关闭'}")
    return False
vis.register_key_callback(ord("C"), toggle_follow_mode)

# ==================== 数据接收线程 ====================
data_queue = queue.Queue(maxsize=50) # 限制队列大小，防止内存无限制增长

def data_receiver_thread():
    """在后台线程中接收ZeroMQ数据并放入队列"""
    print("[THREAD] 数据接收线程启动")
    try:
        while True:
            try:
                # 使用阻塞接收，但设置超时以便线程可以响应中断
                msg = socket.recv_json(flags=0) # flags=0 表示阻塞直到收到
                data_queue.put(msg) # 如果队列满，put会阻塞
                # print(f"[THREAD] 收到消息并放入队列，队列大小: {data_queue.qsize()}") # 调试用
            except zmq.Again:
                # 通常不会触发，因为我们是阻塞接收。但如果设置了NOBLOCK或超时，可能会用到。
                continue
            except Exception as e:
                print(f"[THREAD ERROR] 接收数据时出错: {e}")
                break # 出错则退出线程
    except KeyboardInterrupt:
        pass # 允许线程被主程序中断
    finally:
        print("[THREAD] 数据接收线程结束")

# ==================== 主循环 (大幅修改) ====================
batch_points_buffer = [] # 临时缓存待更新的点
batch_colors_buffer = [] # 临时缓存待更新的颜色
frames_since_last_update = 0

# 启动数据接收线程
receiver_thread = threading.Thread(target=data_receiver_thread, daemon=True)
receiver_thread.start()

try:
    while True:
        # --- 处理来自队列的数据 (非阻塞) ---
        processed_any_data = False
        while not data_queue.empty(): # 尽可能多地处理队列中的数据
            try:
                msg = data_queue.get_nowait() # 非阻塞获取
                processed_any_data = True
                
                # 解析数据
                pose_flat = np.array(msg["header"]["pose"], dtype=np.float64)
                pose_matrix_vis = pose_flat.reshape(4, 4)
                points_flat = np.array(msg["points"], dtype=np.float32)
                
                if len(points_flat) % 3 != 0:
                     print(f"[WARN] 点数量不是3的倍数，跳过此帧。点数: {len(points_flat)}")
                     continue
                     
                current_frame_points_m = points_flat.reshape(-1, 3) # 形状 (N, 3)
                num_points = current_frame_points_m.shape[0]
                print(f"[INFO] 处理帧 {msg['header']['frame_idx']}, 点数: {num_points}")

                # --- 缓冲当前帧点云用于批量更新 ---
                if num_points > 0:
                    batch_points_buffer.append(current_frame_points_m)
                    new_colors = np.tile([0.0, 1.0, 0.0], (num_points, 1)) # 绿色
                    batch_colors_buffer.append(new_colors)
                    frames_since_last_update += 1

                # --- 更新 AUV 模型、坐标系、轨迹 (每次收到数据都更新) ---
                # 更新 AUV 模型
                box_center = box.get_center()
                box.translate(-box_center, relative=False)
                box.rotate(np.eye(3), center=(0, 0, 0))
                box.transform(pose_matrix_vis)

                # 更新 AUV 坐标系
                origin = pose_matrix_vis[:3, 3]
                R = pose_matrix_vis[:3, :3]
                local_x_end = np.array([1.0, 0.0, 0.0]); local_y_end = np.array([0.0, 1.0, 0.0]); local_z_end = np.array([0.0, 0.0, 1.0])
                world_x_end = R @ local_x_end + origin; world_y_end = R @ local_y_end + origin; world_z_end = R @ local_z_end + origin
                auv_coord.points = o3d.utility.Vector3dVector([origin, world_x_end, world_y_end, world_z_end])

                # 更新 AUV 轨迹
                traj_points_list.append(origin.copy())
                if len(traj_points_list) > 10000: traj_points_list.pop(0)
                if len(traj_points_list) >= 2:
                    line.points = o3d.utility.Vector3dVector(traj_points_list)
                    lines = [[i, i+1] for i in range(len(traj_points_list)-1)]
                    line.lines = o3d.utility.Vector2iVector(lines)
                    line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]] * len(lines))
                else:
                    line.points = o3d.utility.Vector3dVector([origin, origin])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])

            except queue.Empty:
                break # 队列已空，跳出内层循环
            except Exception as e:
                print(f"[ERROR] 处理队列中的消息时出错: {e}")
                import traceback
                traceback.print_exc()

        # --- 批量更新历史点云 (定期) ---
        if frames_since_last_update >= BATCH_UPDATE_SIZE or (processed_any_data and data_queue.empty()):
            if batch_points_buffer:
                # 合并批次数据
                combined_points = np.vstack(batch_points_buffer)
                combined_colors = np.vstack(batch_colors_buffer)
                n_new_points = combined_points.shape[0]
                
                # --- 更新历史缓冲区 (优化项) ---
                if buffer_count + n_new_points <= MAX_HISTORY_POINTS:
                    # 缓冲区未满，直接追加
                    history_buffer_points[buffer_count:buffer_count+n_new_points] = combined_points
                    history_buffer_colors[buffer_count:buffer_count+n_new_points] = combined_colors
                    buffer_count += n_new_points
                else:
                    # 缓冲区满了，采用循环覆盖或丢弃策略
                    available_space = MAX_HISTORY_POINTS - buffer_count
                    if available_space > 0:
                        # 先填充剩余空间
                        history_buffer_points[buffer_count:] = combined_points[:available_space]
                        history_buffer_colors[buffer_count:] = combined_colors[:available_space]
                        buffer_count = MAX_HISTORY_POINTS
                    
                    # 如果还有新点，则开始循环覆盖 (FIFO)
                    remaining_points = combined_points[available_space:]
                    remaining_colors = combined_colors[available_space:]
                    n_remaining = remaining_points.shape[0]
                    
                    if n_remaining > 0:
                         # 计算实际需要覆盖的点数 (可能超过缓冲区大小，只取最后的部分)
                         overwrite_start = 0 # 从头开始覆盖
                         actual_overwrite_n = min(n_remaining, MAX_HISTORY_POINTS)
                         
                         # 将新点复制到缓冲区开头
                         history_buffer_points[overwrite_start:overwrite_start+actual_overwrite_n] = \
                             remaining_points[-actual_overwrite_n:] # 取最新的点
                         history_buffer_colors[overwrite_start:overwrite_start+actual_overwrite_n] = \
                             remaining_colors[-actual_overwrite_n:]
                         
                         # buffer_count 保持为 MAX_HISTORY_POINTS
                         
                         # 注意：这种简单的循环覆盖会导致点云跳跃显示。
                         # 更复杂的滑动窗口管理可以解决这个问题，但增加了复杂度。
                         # 这里采用简化策略：一旦满，就只保留最后 MAX_HISTORY_POINTS 个点。
                         # 如果希望平滑滚动，需要维护一个指针记录有效数据的起始位置。
                         # 为简单起见，这里采用“填满后只保留最后N个”的策略。
                         # 如果需要更精确的滑动窗口，需要额外的状态管理。
                         # 一个折衷方案是：当缓冲区满时，丢弃最早的一部分数据，
                         # 然后将新数据追加到末尾。但这需要移动现有数据，开销较大。
                         # 当前实现倾向于简单性和性能。

                # --- 更新 Open3D PointCloud 对象 (优化项) ---
                # 直接使用缓冲区的视图，避免拷贝
                history_pcd.points = o3d.utility.Vector3dVector(history_buffer_points[:buffer_count])
                history_pcd.colors = o3d.utility.Vector3dVector(history_buffer_colors[:buffer_count])
                
                if not history_pcd_added_to_vis:
                    vis.add_geometry(history_pcd, reset_bounding_box=False)
                    history_pcd_added_to_vis = True
                    print("[INFO] 历史点云首次添加到可视化器。")
                
                vis.update_geometry(history_pcd)
                print(f"[INFO] 批量更新历史点云: 新增 {n_new_points} 个点，总计 {buffer_count} 个点")

                # 清空批次缓存
                batch_points_buffer.clear()
                batch_colors_buffer.clear()
                frames_since_last_update = 0

        # --- 更新 AUV 相关几何体 (如果数据已处理) ---
        if processed_any_data:
            vis.update_geometry(auv_coord)
            vis.update_geometry(line)
            vis.update_geometry(box)

        # --- 处理视角跟随 ---
        if FOLLOW_AUV and processed_any_data and 'pose_matrix_vis' in locals():
            auv_pos = pose_matrix_vis[:3, 3]
            R = pose_matrix_vis[:3, :3]
            cam_offset_body = np.array([-5.0, 0.0, 1.0]) 
            camera_pos = auv_pos + R @ cam_offset_body
            look_at = auv_pos
            up = np.array([0.0, 0.0, 1.0])
            ctr = vis.get_view_control()
            ctr.set_lookat(look_at)
            ctr.set_front(look_at - camera_pos) 
            ctr.set_up(up)

        # --- Open3D 渲染循环 ---
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.01) # 可根据需要调整或移除

except KeyboardInterrupt:
    print("\n[INFO] 用户中断，正在退出...")
except Exception as e:
    import traceback
    print(f"\n[ERROR] 发生未预期错误: {e}")
    traceback.print_exc()
finally:
    # 等待接收线程结束 (它应该是守护线程，会随主程序退出)
    # receiver_thread.join(timeout=1.0) # 可选的等待
    vis.destroy_window()
    socket.close()
    context.term()
    print("程序已退出")