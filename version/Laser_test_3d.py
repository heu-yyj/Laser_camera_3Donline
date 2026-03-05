# -*- coding: utf-8 -*-
"""
激光结构光 → 相机坐标系点云 → ZeroMQ 实时发布
已解决：
1. UDP 缓冲区溢出（永不卡）
2. Ctrl+C 无法退出（完美退出）
"""

import socket
import json
import threading
import time
import numpy as np
import zmq
import signal
import sys

# ========================= 配置区 =========================
UDP_IP        = "0.0.0.0"
UDP_PORT      = 8888

ZMQ_SERVER_IP = "10.101.30.221"
ZMQ_PORT      = 5557

fx, fy = 4308.8624, 4302.9958
cx, cy = 1379.5081, 1031.0359
K = np.array([[fx,   0,  cx],
              [ 0,  fy,  cy],
              [ 0,   0,   1]], dtype=np.float64)

PIXEL_SIZE = 0.00345
f_mm       = fx * PIXEL_SIZE
BASELINE_S = 220.0
A_RAD      = np.deg2rad(19.6)
# =========================================================

# 全局标志位 + 资源
context    = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.set_hwm(0)
zmq_socket.set(zmq.CONFLATE, 1)
zmq_socket.connect(f"tcp://{ZMQ_SERVER_IP}:{ZMQ_PORT}")
print(f"[ZMQ] 已连接 {ZMQ_SERVER_IP}:{ZMQ_PORT}")

frame_idx = 0
running = True                                          # 全局运行标志

# ===================== 计算函数（不变）=====================
def compute_depths(x_coords: np.ndarray) -> np.ndarray:
    d = (x_coords - cx) * PIXEL_SIZE
    tanB = d / f_mm
    return BASELINE_S / (np.tan(A_RAD) + tanB)

def image_to_camera(points_img: list, depths: np.ndarray) -> np.ndarray:
    K_inv = np.linalg.inv(K)
    pts = np.zeros((len(points_img), 3))
    for i, pt in enumerate(points_img):
        uv1 = np.array([pt['x'], pt['y'], 1.0])
        norm = K_inv @ uv1
        pts[i] = depths[i] * norm
    return pts

# ===================== UDP 主线程 =====================
def udp_thread():
    global frame_idx, running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    print(f"[UDP] 监听 {UDP_IP}:{UDP_PORT}（非阻塞 + 16MB 缓冲）")

    while running:
        latest_msg = None
        while running:
            try:
                data, _ = sock.recvfrom(65535)
                try:
                    latest_msg = json.loads(data.decode('utf-8'))
                except:
                    pass
            except BlockingIOError:
                break
            except Exception:
                break

        if latest_msg and running:
            points = latest_msg.get("laser_points", [])
            if points:
                xs = np.array([p["x"] for p in points], dtype=np.float64)
                depths = compute_depths(xs)
                cam_pts_mm = image_to_camera(points, depths)
                points_flat = cam_pts_mm.reshape(-1).tolist()

                payload = {
                    "header": {
                        "timestamp": time.time(),
                        "frame_id": "laser_camera",
                        "frame_idx": frame_idx
                    },
                    "points": points_flat
                }
                zmq_socket.send_json(payload, flags=zmq.NOBLOCK)
                print(f"[Published] Frame {frame_idx:05d} | {len(points)} pts")
                frame_idx += 1

        time.sleep(0.001)      # 防止空转 100% CPU

    sock.close()
    print("[UDP] 线程已安全退出")

# ===================== 优雅退出处理 =====================
def signal_handler(sig, frame):
    global running
    print("\n\n收到 Ctrl+C，正在优雅退出...")
    running = False
    time.sleep(0.5)                     # 给线程一点时间收尾
    zmq_socket.close()
    context.term()
    print("所有资源已释放，程序完全退出！")
    sys.exit(0)

# 注册信号
signal.signal(signal.SIGINT, signal_handler)      # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)     # kill

# ===================== 主程序 =====================
if __name__ == "__main__":
    threading.Thread(target=udp_thread, daemon=False).start()   # 改成非 daemon！

    print("\n=== 激光结构光实时点云发布系统（终极稳定版）已启动 ===")
    print("   支持 Ctrl+C 完美退出！\n")

    try:
        while running:
            time.sleep(1)
    except:
        pass
    finally:
        # 保险起见再执行一次退出流程
        signal_handler(None, None)