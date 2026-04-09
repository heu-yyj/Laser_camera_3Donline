import zmq
import json
import numpy as np
import open3d as o3d

def zmq_viewer():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)  # 因为你用 recv_string，且对方是 PUSH，所以 PULL 正确
    
    # viewer.py 中
    socket.bind("tcp://192.168.5.110:5557")  # ← 显式指定 IP
    print("ZMQ服务器已启动，绑定到 192.168.5.110:5557")
    
    print("ZeroMQ 显示端已启动，等待数据...")

    # 初始化 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer", width=1024, height=768)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    first_time = True

    try:
        while True:
            # 接收 JSON 字符串
            msg_str = socket.recv_string()
            
            try:
                data = json.loads(msg_str)
                points_list = data.get("points", [])
                
                if len(points_list) == 0:
                    print("[警告] 点云为空")
                    continue

                # 转为 NumPy 数组，并 reshape 为 (N, 3)
                points = np.array(points_list, dtype=np.float32)
                if points.size % 3 != 0:
                    print(f"[错误] 点数量不是3的倍数，无法构成XYZ：{points.size}")
                    continue

                points = points.reshape(-1, 3)
                print(f"收到点云：{points.shape[0]} 个点")

                # 更新点云
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # 首次添加几何体后，后续只需 update
                if first_time:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    first_time = False
                else:
                    vis.update_geometry(pcd)
                
                vis.poll_events()
                vis.update_renderer()

            except json.JSONDecodeError as e:
                print(f"[错误] JSON 解析失败: {e}")
            except Exception as e:
                print(f"[错误] 处理点云时出错: {e}")

    except KeyboardInterrupt:
        print("\nViewer 已停止")
    finally:
        vis.destroy_window()
        socket.close()
        context.term()

if __name__ == "__main__":
    zmq_viewer()