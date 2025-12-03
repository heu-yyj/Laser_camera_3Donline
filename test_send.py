# test_send_random.py
# 持续向 ZeroMQ 发送随机点云 + 移动的位姿，用于测试 viewer

import zmq
import numpy as np
import time
import json

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://127.0.0.1:5557")
    print("[TEST] 已连接到 viewer，开始发送随机点云...")

    frame_idx = 0
    try:
        while True:
            # 生成随机点数（50 ~ 500 个点）
            num_points = np.random.randint(50, 500)
            points = (np.random.rand(num_points, 3) - 0.5) * 10  # 范围 [-5, 5]

            # 生成沿 X 轴前进的位姿（模拟AUV移动）
            x = frame_idx * 0.1  # 每帧前进 0.1 米
            pose = np.eye(4)
            pose[0, 3] = x       # 平移X
            pose[1, 3] = np.sin(x * 0.5) * 2  # 正弦波Y
            pose[2, 3] = 0.0

            # 构造消息
            msg = {
                "header": {
                    "pose": pose.flatten().tolist()  # 转为16元素列表
                },
                "points": points.astype(np.float32).flatten().tolist()
            }

            sock.send_json(msg, flags=zmq.NOBLOCK)
            print(f"发送帧 {frame_idx}: {num_points} 个点, AUV位置=({x:.2f}, {pose[1,3]:.2f}, 0)")

            frame_idx += 1
            time.sleep(0.1)  # 10 FPS

    except KeyboardInterrupt:
        print("\n[INFO] 停止发送")
    finally:
        sock.close()
        ctx.term()

if __name__ == "__main__":
    main()