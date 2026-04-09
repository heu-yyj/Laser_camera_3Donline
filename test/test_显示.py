import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# 设置参数
frame_size = 500
spacing = 750
frames = []

# 初始（无旋转）
R_cumulative = R.from_matrix(np.eye(3))  # 单位旋转
frame0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
frame0.translate([0, 0, 0])
frames.append(frame0)

# 第1步：绕 Z 轴 +180° (修正了角度值)
R_z = R.from_euler('z', np.deg2rad(199.6))
R_cumulative = R_z  # 当前累积 = R_z
frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
frame1.rotate(R_cumulative.as_matrix(), center=(0, 0, 0))
frame1.translate([spacing, 0, 0])
frames.append(frame1)

# 第2步：在当前姿态下，绕 **自身 Y 轴** 0
R_y = R.from_euler('y', np.deg2rad(0))  # 修正了角度值和旋转轴
R_cumulative = R_z * R_y  # intrinsic: 先 z，再 y
frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
frame2.rotate(R_cumulative.as_matrix(), center=(0, 0, 0))
frame2.translate([2 * spacing, 0, 0])
frames.append(frame2)

# 第3步：在当前姿态下，绕 **自身 X 轴** 90° (修正了角度符号)
R_x = R.from_euler('x', np.deg2rad(90))
R_cumulative = R_z * R_y * R_x  # 完整 intrinsic ZYX
frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
frame3.rotate(R_cumulative.as_matrix(), center=(0, 0, 0))
frame3.translate([3 * spacing, 0, 0])
frames.append(frame3)


# frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
# frame3.rotate(R_cumulative.as_matrix().T, center=(0, 0, 0))
# frame3.translate([-1 * spacing, 0, 0])
# frames.append(frame3)

# --- 增加点云部分 ---
# 加载你的点云数据
point_cloud = o3d.io.read_point_cloud("D:\\工作\\LaserData\\CameraSpace\\converted_pointclouds\\camera_space_20260114_172914.ply") # 替换为你的点云文件路径

# 如果点云没有颜色信息，则为其生成一种非白色的颜色
if not point_cloud.has_colors():
    colors = np.random.rand(len(point_cloud.points), 3) * 0.8 + 0.2 # 随机颜色，范围 [0.2, 1)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

# 根据累积旋转矩阵变换点云
R_cumulative_matrix = R_cumulative.as_matrix()  # 获取累积旋转矩阵
T = np.eye(4)  # 创建一个单位齐次变换矩阵
T[:3, :3] = R_cumulative_matrix  # 设置旋转部分
T[0, 3] = 3 * spacing  # 设置平移部分，与 frame3 的位置一致

# 应用变换
transformed_point_cloud = point_cloud.transform(T)

# 将变换后的点云添加到 frames 列表中以便一起可视化
frames.append(transformed_point_cloud)
# --- 点云部分结束 ---

# 可视化所有几何体
o3d.visualization.draw_geometries(frames, window_name="Step-by-step Intrinsic Rotations (Z → Y → X) with Point Cloud")