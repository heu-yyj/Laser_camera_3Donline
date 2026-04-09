import open3d as o3d
import numpy as np
import sys
import os
import threading
import queue
import time
import json

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("❌ 未找到 scipy 模块。请运行 'pip install scipy' 安装。")
    sys.exit(1)

try:
    import zmq
except ImportError:
    print("❌ 未找到 pyzmq 模块。请运行 'pip install pyzmq' 安装。")
    sys.exit(1)

# 预设颜色列表（用于多个点云区分），RGB 范围 [0,1]
COLORS = [
    [1, 1, 1],    # 白
    # [1, 0, 0],    # 红
    # [0, 0, 1],    # 蓝
    # [0, 1, 0],    # 绿
    # [1, 1, 0],    # 黄
    # [1, 0, 1],    # 品红
    # [0, 1, 1],    # 青
    # [1, 0.5, 0],  # 橙
    # [0.5, 0, 1],  # 紫
]

class RealTimePointCloudVisualizer:
    def __init__(self, zmq_address="tcp://192.168.5.110:5557", max_trajectory_points=10000):
        self.zmq_bind_address = "tcp://*:5557"
        self.max_trajectory_points = max_trajectory_points
        self.data_queue = queue.Queue(maxsize=100) # 队列大小限制，防止积压
        
        self.vis = None
        self.geometries = {} # 存储所有几何体对象，便于更新
        self.lock = threading.Lock() # 保护共享数据结构

        # 存储历史轨迹和姿态坐标系
        self.trajectory_points = []
        self.pose_frame_points = [] # 存储所有姿态坐标系的点
        self.pose_frame_lines_indices = [] # 存储所有姿态坐标系的线索引
        self.pose_frame_colors = [] # 存储所有姿态坐标系的线颜色

        # 存储累积的历史点云
        self.accumulated_points_np = np.empty((0, 3)) # 累积点云的 NumPy 数组
        self.accumulated_colors_np = np.empty((0, 3)) # 累积点云的颜色数组

        # ZeroMQ 接收线程
        self.zmq_thread = threading.Thread(target=self._zmq_receive_loop, daemon=True)
        self.running = True

        # --- 初始化可能在 _update_dynamic_geometries 中被访问的属性 ---
        with self.lock:
            # 初始化点云数据和颜色
            self.current_points_np = np.empty((0, 3)) # 空数组
            self.current_color = COLORS[0] # 默认颜色


    def _zmq_receive_loop(self):
        """ZeroMQ 接收数据的后台线程"""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(self.zmq_bind_address) # 使用新的 bind 地址
        print(f"[ZMQ] 服务器已启动，绑定到: {self.zmq_bind_address}, 等待数据...")

        while self.running:
            try:
                # 非阻塞接收，避免线程卡死
                message_json = socket.recv_string(flags=zmq.NOBLOCK)
                if message_json:
                    try:
                        message_data = json.loads(message_json)
                        # 将数据放入队列，如果队列满了就丢弃旧数据
                        try:
                            self.data_queue.put_nowait(message_data)
                        except queue.Full:
                            print("[ZMQ] 数据队列已满，丢弃一帧数据。")
                    except json.JSONDecodeError as e:
                        print(f"[ZMQ] JSON 解析错误: {e}")
            except zmq.Again:
                # 没有收到消息，短暂休眠
                time.sleep(0.001)
            except Exception as e:
                print(f"[ZMQ] 接收错误: {e}")
                time.sleep(0.1) # 出错后稍作休眠再继续
        socket.close()
        context.term()
        print("[ZMQ] 接收线程结束。")

    def _process_received_data(self, data):
        """处理接收到的一帧数据"""
        header = data.get("header", {})
        points_flat = data.get("points", [])
        frame_idx = header.get("frame_idx", -1)

        # 解析姿态
        pose_matrix_list = header.get("pose", None)
        if pose_matrix_list is None:
            print(f"[WARN] 帧 {frame_idx} 缺少 pose 信息，跳过。")
            return
        try:
            T = np.array(pose_matrix_list, dtype=np.float64)
            pos = T[:3, 3] # 位置 [x, y, z]
            R_mat = T[:3, :3] # 旋转矩阵
        except (ValueError, IndexError) as e:
            print(f"[WARN] 帧 {frame_idx} 姿态矩阵解析失败: {e}, 跳过。")
            return

        # 解析点云
        if not points_flat:
            print(f"[INFO] 帧 {frame_idx} 无点云数据。")
            points_np = np.empty((0, 3)) # 空数组
        else:
            try:
                points_np = np.array(points_flat, dtype=np.float64).reshape(-1, 3)
            except (ValueError, IndexError) as e:
                print(f"[ERROR] 帧 {frame_idx} 点云数据解析失败: {e}")
                return

        print(f"[INFO] 处理帧 {frame_idx} - 位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), 点数: {len(points_np)}")

        # 更新内部状态（轨迹、姿态坐标系、当前点云数据、累积点云数据）
        with self.lock:
            # --- 1. 更新轨迹 ---
            self.trajectory_points.append(pos.copy())
            if len(self.trajectory_points) > self.max_trajectory_points:
                self.trajectory_points.pop(0)

            # --- 2. 更新姿态坐标系 (每50帧添加一个) ---
            if len(self.trajectory_points) % 50 == 0:
                axis_length = 500.0
                local_axes = np.array([
                    [0, 0, 0],             # Origin
                    [axis_length, 0, 0],     # X-axis end
                    [0, axis_length, 0],     # Y-axis end
                    [0, 0, axis_length]      # Z-axis end
                ])
                world_axes = (R_mat @ local_axes.T).T + pos

                base_idx = len(self.pose_frame_points)
                self.pose_frame_points.extend(world_axes)
                # Add lines (indices relative to the growing point list)
                self.pose_frame_lines_indices.extend([
                    [base_idx, base_idx+1], # X
                    [base_idx, base_idx+2], # Y
                    [base_idx, base_idx+3]  # Z
                ])
                # Add colors (red, green, blue)
                self.pose_frame_colors.extend([
                    [1, 0, 0], [0, 1, 0], [0, 0, 1]
                ])

            # --- 3. 更新当前点云数据 ---
            if len(points_np) > 0:
                color_idx = frame_idx % len(COLORS)
                self.current_points_np = points_np
                self.current_color = COLORS[color_idx]

            # --- 4. 更新累积点云数据 ---
            if len(points_np) > 0:
                # 计算当前帧点云的颜色
                color_idx = frame_idx % len(COLORS)
                current_frame_color = COLORS[color_idx]
                # 创建对应数量的颜色数组
                num_new_points = len(points_np)
                current_frame_colors = np.tile(current_frame_color, (num_new_points, 1))

                # 将新点和新颜色追加到累积数组
                self.accumulated_points_np = np.vstack((self.accumulated_points_np, points_np))
                self.accumulated_colors_np = np.vstack((self.accumulated_colors_np, current_frame_colors))


    def _create_xy_grid(self, size=50000, step=500, color=[105/255, 105/255, 105/255]):
        points = []
        lines = []
        half = size / 2
        for y in np.arange(-half, half + step, step):
            points.append([-half, y, 0])
            points.append([ half, y, 0])
            lines.append([len(points)-2, len(points)-1])
        for x in np.arange(-half, half + step, step):
            points.append([x, -half, 0])
            points.append([x,  half, 0])
            lines.append([len(points)-2, len(points)-1])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return line_set

    def run(self):
        # --- 1. 初始化可视化器 ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="实时点云与AUV轨迹/姿态查看器", width=1280, height=720)

        render_opt = self.vis.get_render_option()
        render_opt.point_size = 1.0
        render_opt.background_color = np.array([0, 0, 0])

        # --- 2. 添加固定的辅助元素 (只需添加一次) ---
        grid = self._create_xy_grid(size=10000, step=1000)
        coord_global = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)
        self.vis.add_geometry(grid, reset_bounding_box=True) # Add once
        self.vis.add_geometry(coord_global, reset_bounding_box=False) # Add once
        self.geometries.update({'grid': grid, 'coord_global': coord_global})

        # --- 3. 添加动态几何体对象 (只需添加一次) ---
        # 当前点云 (最新一帧)
        self.current_pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.current_pcd, reset_bounding_box=False) 
        self.geometries['current_pcd'] = self.current_pcd

        # 累积点云 (所有历史点云)
        self.accumulated_pcd = o3d.geometry.PointCloud() # 新增累积点云对象
        self.vis.add_geometry(self.accumulated_pcd, reset_bounding_box=False) # Add once
        self.geometries['accumulated_pcd'] = self.accumulated_pcd # Store reference

        # 轨迹线
        self.trajectory_line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.trajectory_line_set, reset_bounding_box=False) 
        self.geometries['trajectory'] = self.trajectory_line_set

        # 姿态坐标系
        self.pose_frames_line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.pose_frames_line_set, reset_bounding_box=False) 
        self.geometries['pose_frames'] = self.pose_frames_line_set

        # 设置初始视角
        view_control = self.vis.get_view_control()
        view_control.set_front([0, 0, -1]) # Looking down from positive Z
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0]) # Y up
        view_control.set_zoom(0.5)

        # --- 4. 启动ZMQ接收线程 ---
        self.zmq_thread.start()

        print("🟢 实时可视化器已启动。")
        print("   🔹 辅助元素: XY网格平面, 全局坐标系 (已显示)")
        print("   🔹 动态元素: AUV轨迹, 姿态坐标系, 最新点云, 累积点云")
        print("   🔹 左键拖动：旋转视角")
        print("   🔹 右键拖动：平移")
        print("   🔹 滚轮：缩放")
        print("   🔹 按 'q' 或关闭窗口退出")
        print("   📡 等待来自 ZeroMQ 的数据...")

        # --- 5. 主可视化循环 ---
        prev_update_time = time.time()
        update_interval = 0.033  # ~30 FPS 更新频率

        while self.running:
            try:
                # 检查是否有新数据需要处理
                new_data_processed = False
                while not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    self._process_received_data(data)
                    new_data_processed = True
                
                # 如果有新数据，或者达到更新间隔，则更新几何体
                current_time = time.time()
                if new_data_processed or (current_time - prev_update_time) >= update_interval:
                    self._update_dynamic_geometries()
                    prev_update_time = current_time

                # 刷新视图
                self.vis.poll_events()
                self.vis.update_renderer()

                # 小幅休眠，避免CPU占用过高
                time.sleep(0.005)

            except KeyboardInterrupt:
                print("\n[INFO] 接收到中断信号，正在关闭...")
                break

        # --- 6. 清理 ---
        self.running = False
        self.zmq_thread.join(timeout=2) # 等待ZMQ线程结束
        self.vis.destroy_window()
        print("🔴 可视化器已关闭。")

    def _update_dynamic_geometries(self):
        """在主线程中安全地更新动态几何体（轨迹、姿态、点云）的属性"""
        with self.lock:
            # --- 1. 更新当前点云 (最新一帧) ---
            self.current_pcd.points = o3d.utility.Vector3dVector(self.current_points_np)
            if len(self.current_points_np) > 0:
                 self.current_pcd.paint_uniform_color(self.current_color)
            self.vis.update_geometry(self.current_pcd)

            # --- 2. 更新累积点云 (所有历史点云) ---
            # 将累积的点和颜色应用到累积点云对象
            self.accumulated_pcd.points = o3d.utility.Vector3dVector(self.accumulated_points_np)
            self.accumulated_pcd.colors = o3d.utility.Vector3dVector(self.accumulated_colors_np) # 应用颜色
            # 通知可视化器累积点云已更新
            self.vis.update_geometry(self.accumulated_pcd) # 关键：更新累积点云几何体

            # --- 3. 更新轨迹线 ---
            if len(self.trajectory_points) > 1:
                traj_points_np = np.array(self.trajectory_points)
                traj_lines_np = np.array([[i, i+1] for i in range(len(traj_points_np)-1)])
                traj_colors_np = np.tile([0.5, 0.5, 0.5], (len(traj_lines_np), 1)) # 灰色轨迹

                self.trajectory_line_set.points = o3d.utility.Vector3dVector(traj_points_np)
                self.trajectory_line_set.lines = o3d.utility.Vector2iVector(traj_lines_np)
                self.trajectory_line_set.colors = o3d.utility.Vector3dVector(traj_colors_np)
                self.vis.update_geometry(self.trajectory_line_set)

            # --- 4. 更新姿态坐标系 ---
            if len(self.pose_frame_points) > 0:
                self.pose_frames_line_set.points = o3d.utility.Vector3dVector(np.array(self.pose_frame_points))
                self.pose_frames_line_set.lines = o3d.utility.Vector2iVector(np.array(self.pose_frame_lines_indices))
                self.pose_frames_line_set.colors = o3d.utility.Vector3dVector(np.array(self.pose_frame_colors))
                self.vis.update_geometry(self.pose_frames_line_set)


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    ZMQ_CONNECT_ADDRESS_DEFAULT = "tcp://192.168.1.101:5557"
    MAX_TRAJECTORY_POINTS = 10000000

    if len(sys.argv) > 1:
        provided_addr = sys.argv[1]
        print(f"📌 忽略命令行提供的 ZMQ 地址 '{provided_addr}' (因为接收端现在是服务器并绑定到固定端口)。")
        print(f"   服务器将绑定到 'tcp://*:5557'")
    else:
        print(f"   服务器将绑定到 'tcp://*:5557'")

    # 创建可视化器实例并运行
    visualizer = RealTimePointCloudVisualizer(zmq_address=ZMQ_CONNECT_ADDRESS_DEFAULT, max_trajectory_points=MAX_TRAJECTORY_POINTS)
    visualizer.run()