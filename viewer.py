# viewer.py
# 实时可视化：米单位点云 + AUV模型 + AUV坐标系 (支持交互缩放/平移/旋转)
# 功能：显示历史点云累积 + 水池环境 + 世界坐标系
import zmq
import numpy as np
import open3d as o3d
import time
import sys
import collections # <<< 新增导入

# ===== 配置参数 =====
# 移除 GLOBAL_SCALE，假设输入数据已经是米
FOLLOW_AUV = False        # 设置为 False 以允许自由交互视角
MAX_HISTORY_POINTS = 500000 # 最多保留的历史点数，防止内存溢出

# 水池尺寸 (米) - 注意单位变化
TANK_WIDTH_M = 10.0   # X方向 10m
TANK_LENGTH_M = 10.0  # Y方向 10m
TANK_HEIGHT_M = 6.0   # Z方向 6m
GRID_SPACING_M = 1.0  # 网格间距 1m

# ==================== ZeroMQ 初始化 ====================
context = zmq.Context()
socket = context.socket(zmq.PULL)
try:
    socket.bind("tcp://127.0.0.1:5557")
    print("[ZMQ] 成功绑定到 tcp://127.0.0.1:5557")
except Exception as e:
    print(f"[ERROR] ZMQ bind 失败: {e}")
    sys.exit(1)

# ==================== 辅助函数 ====================
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
    x_lines_base = np.arange(-w_half, w_half + gs/2, gs) # 包括边界
    y_lines_base = np.arange(-l_half, l_half + gs/2, gs)
    
    # 沿Y方向的X线
    for y_val in y_lines_base:
        start_idx = len(points)
        points.extend([[x, y_val, 0] for x in x_lines_base])
        end_idx = len(points) - 1
        lines.extend([[i, i+1] for i in range(start_idx, end_idx)])
        
    # 沿X方向的Y线
    for x_val in x_lines_base:
        start_idx = len(points)
        points.extend([[x_val, y, 0] for y in y_lines_base])
        end_idx = len(points) - 1
        lines.extend([[i, i+1] for i in range(start_idx, end_idx)])

    # --- 侧壁网格 ---
    # 定义四个角点
    corners_bottom = [
        [-w_half, -l_half, 0], [w_half, -l_half, 0],
        [w_half, l_half, 0], [-w_half, l_half, 0]
    ]
    corners_top = [[x, y, h] for x, y, _ in corners_bottom]

    # 将顶点加入点列表
    all_corners = corners_bottom + corners_top
    base_corner_idx = len(points)
    points.extend(all_corners)

    # 底面四边
    lines.extend([
        [base_corner_idx + 0, base_corner_idx + 1],
        [base_corner_idx + 1, base_corner_idx + 2],
        [base_corner_idx + 2, base_corner_idx + 3],
        [base_corner_idx + 3, base_corner_idx + 0],
    ])
    # 顶面四边
    lines.extend([
        [base_corner_idx + 4, base_corner_idx + 5],
        [base_corner_idx + 5, base_corner_idx + 6],
        [base_corner_idx + 6, base_corner_idx + 7],
        [base_corner_idx + 7, base_corner_idx + 4],
    ])
    # 四根立柱
    lines.extend([
        [base_corner_idx + 0, base_corner_idx + 4],
        [base_corner_idx + 1, base_corner_idx + 5],
        [base_corner_idx + 2, base_corner_idx + 6],
        [base_corner_idx + 3, base_corner_idx + 7],
    ])

    # 垂直网格线
    num_vertical_lines_x = int(width_m / grid_spacing_m) + 1
    num_vertical_lines_y = int(length_m / grid_spacing_m) + 1

    # X方向侧壁上的垂直线 (前后两面)
    for i in range(num_vertical_lines_x):
        x_val = -w_half + i * gs
        # 前面 (y = -l_half)
        p1_idx = len(points)
        points.append([x_val, -l_half, 0])
        p2_idx = len(points)
        points.append([x_val, -l_half, h])
        lines.append([p1_idx, p2_idx])
        
        # 后面 (y = l_half)
        p3_idx = len(points)
        points.append([x_val, l_half, 0])
        p4_idx = len(points)
        points.append([x_val, l_half, h])
        lines.append([p3_idx, p4_idx])

    # Y方向侧壁上的垂直线 (左右两面)
    for i in range(num_vertical_lines_y):
        y_val = -l_half + i * gs
        # 左面 (x = -w_half)
        p1_idx = len(points)
        points.append([-w_half, y_val, 0])
        p2_idx = len(points)
        points.append([-w_half, y_val, h])
        lines.append([p1_idx, p2_idx])
        
        # 右面 (x = w_half)
        p3_idx = len(points)
        points.append([w_half, y_val, 0])
        p4_idx = len(points)
        points.append([w_half, y_val, h])
        lines.append([p3_idx, p4_idx])

    # 水平网格线 (侧壁上)
    z_levels = np.arange(gs, h, gs)
    # 前后面
    for z_val in z_levels:
        # 前面
        p1_idx = len(points)
        points.append([-w_half, -l_half, z_val])
        p2_idx = len(points)
        points.append([w_half, -l_half, z_val])
        lines.append([p1_idx, p2_idx])
        # 后面
        p3_idx = len(points)
        points.append([-w_half, l_half, z_val])
        p4_idx = len(points)
        points.append([w_half, l_half, z_val])
        lines.append([p3_idx, p4_idx])
    # 左右面
    for z_val in z_levels:
        # 左面
        p1_idx = len(points)
        points.append([-w_half, -l_half, z_val])
        p2_idx = len(points)
        points.append([-w_half, l_half, z_val])
        lines.append([p1_idx, p2_idx])
        # 右面
        p3_idx = len(points)
        points.append([w_half, -l_half, z_val])
        p4_idx = len(points)
        points.append([w_half, l_half, z_val])
        lines.append([p3_idx, p4_idx])

    # 所有点和线都已添加，现在设置颜色
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
    axis_len = axis_length_m # 直接使用米
    
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


# ==================== 几何体初始化 ====================
# --- 水池 ---
tank_lineset = create_water_tank(
    TANK_WIDTH_M, TANK_LENGTH_M, TANK_HEIGHT_M, GRID_SPACING_M
)

# --- 世界坐标系 ---
world_coord = create_world_coordinate_system(axis_length_m=5.0) # 5m长轴

# --- 用于累积历史点云 (使用 deque) ---
# 使用 maxlen 可以自动实现 FIFO，但我们手动控制以更好地管理颜色
history_points_deque = collections.deque(maxlen=MAX_HISTORY_POINTS)
history_colors_deque = collections.deque(maxlen=MAX_HISTORY_POINTS)
history_pcd = o3d.geometry.PointCloud() # 实际用于可视化的点云对象

# --- AUV坐标系 ---
auv_coord = o3d.geometry.LineSet()
# 初始长度设为1米
auv_coord.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
auv_coord.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
auv_coord.colors = o3d.utility.Vector3dVector([
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
])

# --- AUV轨迹线 ---
line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0]])
line.lines = o3d.utility.Vector2iVector([[0, 1]])
line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]])

# --- AUV模型 ---
# 直接使用米定义尺寸
width_m = 0.3   # 30cm
height_m = 0.2  # 20cm
depth_m = 0.6   # 60cm
box = o3d.geometry.TriangleMesh.create_box(width=width_m, height=height_m, depth=depth_m)
box.paint_uniform_color([0.2, 0.6, 1.0])
box.translate([-width_m/2, -height_m/2, -depth_m/2]) # 中心对齐

# --- 其他辅助变量 ---
traj_points_list = [] # 使用普通 list 即可，因为只在尾部追加和偶尔 pop(0)

# ==================== Open3D 可视化窗口 ====================
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("AUV 实时可视化 (含历史点云 & 水池环境)", width=1280, height=720)

# --- 添加所有静态几何体 ---
static_geometries = [tank_lineset, world_coord, auv_coord, line, box]
for geom in static_geometries:
    vis.add_geometry(geom, reset_bounding_box=False)

# --- history_pcd 初始为空，暂不添加 ---

opt = vis.get_render_option()
opt.point_size = 1.0   # 点大小设为1.0以适应米单位
opt.line_width = 2.0    # AUV线宽设为2.0
opt.background_color = np.array([0.1, 0.1, 0.1])
opt.show_coordinate_frame = False # 我们有自己的世界坐标系

print("[INFO] 正在设置初始视角...")
vis.reset_view_point(True) 
print("[INFO] 初始视角设置完成.")

print(f"\n--- 可视化环境已准备就绪 ---")
print(f"等待 ZeroMQ 数据... (输入单位: m, 最大历史点数: {MAX_HISTORY_POINTS})")
print("提示: ")
print("  - 使用鼠标左键拖拽旋转视角")
print("  - 使用鼠标右键拖拽平移视角")
print("  - 使用鼠标滚轮缩放")
print("  - 按 'R' 或 'Ctrl+R' 重置视角")
print("  - 按 'C' 切换相机跟随模式")
print("--------------------------------\n")

# ==================== 回调函数 ====================
def toggle_follow_mode(vis):
    global FOLLOW_AUV
    FOLLOW_AUV = not FOLLOW_AUV
    print(f"[INFO] 相机跟随模式已切换为: {'开启' if FOLLOW_AUV else '关闭'}")
    return False

vis.register_key_callback(ord("C"), toggle_follow_mode)

# ==================== 辅助函数：更新历史点云 ====================
def update_history_pointcloud():
    """从 deque 更新 Open3D 点云对象"""
    global history_pcd, history_points_deque, history_colors_deque, history_pcd_added_to_vis
    
    if len(history_points_deque) == 0:
        # 如果没有点，确保点云是空的
        history_pcd.points = o3d.utility.Vector3dVector([])
        history_pcd.colors = o3d.utility.Vector3dVector([])
    else:
        # 将 deque 转换为 numpy 数组
        # np.asarray(list(deque)) 是一种常见做法
        points_np = np.asarray(list(history_points_deque))
        colors_np = np.asarray(list(history_colors_deque))
        
        # 更新点云对象
        history_pcd.points = o3d.utility.Vector3dVector(points_np)
        history_pcd.colors = o3d.utility.Vector3dVector(colors_np)
        
    # 如果还没添加到可视化器，则添加；否则更新
    if not history_pcd_added_to_vis:
        vis.add_geometry(history_pcd, reset_bounding_box=False)
        history_pcd_added_to_vis = True
        print("[INFO] 历史点云首次添加到可视化器。")
    else:
        vis.update_geometry(history_pcd)
    print(f"[DEBUG] 已更新历史点云总数: {len(history_pcd.points)} 个点")


# ==================== 主循环 ====================
history_pcd_added_to_vis = False 

try:
    while True:
        try:
            msg = socket.recv_json(flags=zmq.NOBLOCK)
            
            # 假设 pose 和 points 都是以米为单位
            pose_flat = np.array(msg["header"]["pose"], dtype=np.float64)
            pose_matrix_vis = pose_flat.reshape(4, 4) # 直接使用，无需缩放

            points_flat = np.array(msg["points"], dtype=np.float32)
            if len(points_flat) % 3 != 0:
                 print(f"[WARN] 点数量不是3的倍数，跳过此帧。点数: {len(points_flat)}")
                 continue
                 
            original_points_m = points_flat.reshape(-1, 3) # 形状 (N, 3)
            num_points = original_points_m.shape[0]
            print(f"[INFO] 接收到一帧点云，共 {num_points} 个点")

            # --- 更新历史点云 (累积) 使用 deque ---
            if num_points > 0:
                # --- 给新点云一个醒目的颜色 (例如绿色) ---
                new_colors = np.tile([0.0, 1.0, 0.0], (num_points, 1))  # 纯绿色 (N, 3)
                
                # 将新点和颜色逐个添加到 deque 的右侧
                for i in range(num_points):
                    # append 自动处理 maxlen，超过会从左侧弹出
                    history_points_deque.append(original_points_m[i])
                    history_colors_deque.append(new_colors[i])
                
                # 更新 Open3D 点云对象 (批量更新，而非每帧都做大量数组操作)
                update_history_pointcloud()
                # 注意：update_history_pointcloud 内部已经处理了 add_geometry 和 update_geometry

            # --- 更新轨迹 ---
            auv_position_m = pose_matrix_vis[:3, 3].copy() # 提取平移部分
            traj_points_list.append(auv_position_m)
            # 限制轨迹点数量，防止无限增长
            if len(traj_points_list) > 10000: 
                 traj_points_list.pop(0) # 从头部移除旧点
                 
            if len(traj_points_list) >= 2:
                line.points = o3d.utility.Vector3dVector(traj_points_list)
                lines = [[i, i+1] for i in range(len(traj_points_list)-1)]
                line.lines = o3d.utility.Vector2iVector(lines)
                line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]] * len(lines))
                print(f"[DEBUG] 已更新轨迹: {len(traj_points_list)} 个点")
            else:
                # 至少保证有两个点，即使相同
                line.points = o3d.utility.Vector3dVector([auv_position_m, auv_position_m])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                print("[DEBUG] 轨迹点不足...")

            # --- 更新 AUV 模型 ---
            # 重置变换
            box_center = box.get_center()
            box.translate(-box_center, relative=False)
            box.rotate(np.eye(3), center=(0, 0, 0))
            # 应用新的位姿变换
            box.transform(pose_matrix_vis)
            print("[DEBUG] 已更新 AUV 模型")

            # --- 更新 AUV 坐标系 ---
            # 提取位姿矩阵中的平移和旋转部分
            origin = pose_matrix_vis[:3, 3] # 平移 [x, y, z]
            R = pose_matrix_vis[:3, :3]     # 旋转矩阵 3x3
            
            # 定义局部坐标轴的终点 (1米长)
            local_x_end = np.array([1.0, 0.0, 0.0])
            local_y_end = np.array([0.0, 1.0, 0.0])
            local_z_end = np.array([0.0, 0.0, 1.0])
            
            # 变换到世界坐标系
            world_x_end = R @ local_x_end + origin
            world_y_end = R @ local_y_end + origin
            world_z_end = R @ local_z_end + origin
            
            # 更新 LineSet 的点
            auv_coord.points = o3d.utility.Vector3dVector([origin, world_x_end, world_y_end, world_z_end])
            print(f"[DEBUG] 已更新 AUV 坐标系")

            # --- 更新几何体 ---
            # history_pcd 已在 update_history_pointcloud 中处理
            vis.update_geometry(auv_coord)
            vis.update_geometry(line)
            vis.update_geometry(box)
            # 静态几何体（水池、坐标系）不需要在循环中更新
            
            # --- 处理视角跟随 ---
            if FOLLOW_AUV:
                auv_pos = pose_matrix_vis[:3, 3]
                R = pose_matrix_vis[:3, :3]
                # 在AUV坐标系中，相机在后上方 (例如 -5m 后, 1m 上)
                cam_offset_body = np.array([-5.0, 0.0, 1.0]) 
                camera_pos = auv_pos + R @ cam_offset_body
                look_at = auv_pos
                up = np.array([0.0, 0.0, 1.0])
                ctr = vis.get_view_control()
                # 注意 set_front 是指向相机的向量，即 lookat - eye
                ctr.set_lookat(look_at)
                ctr.set_front(look_at - camera_pos) 
                ctr.set_up(up)

            print("---")

        except zmq.Again:
            pass

        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.01) # 可根据需要调整，或者移除以获得最大帧率

except KeyboardInterrupt:
    print("\n[INFO] 用户中断，正在退出...")
except Exception as e:
    import traceback
    print(f"\n[ERROR] 发生未预期错误: {e}")
    traceback.print_exc() # 打印详细堆栈跟踪，便于调试
finally:
    vis.destroy_window()
    socket.close()
    context.term()
    print("程序已退出")