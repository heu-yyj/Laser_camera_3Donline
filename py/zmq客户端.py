import zmq
import time

def zmq_loop_client():
    # 创建ZeroMQ上下文
    context = zmq.Context()
    
    # 创建PUSH类型套接字（用于向服务器发送消息）
    socket = context.socket(zmq.PUSH)
    
    # 连接到服务器的局域网IP（替换为你的服务器实际局域网IP）
    server_ip = "192.168.1.110"  # 例如：服务器的局域网IP
    server_port = 5557
    socket.connect(f"tcp://{server_ip}:{server_port}")
    print(f"已连接到服务器 {server_ip}:{server_port}，开始循环发送消息...")
    print("（按Ctrl+C停止发送）")
    
    try:
        # 消息计数器（用于区分不同消息）
        message_count = 1
        
        # 循环发送消息
        while True:
            # 自定义消息内容（可根据需求修改，比如加入时间戳、计数器等）
            message = f"这是第 {message_count} 条消息 | 发送时间：{time.strftime('%H:%M:%S')}"
            
            # 发送消息（字符串类型）
            socket.send_string(message)
            print(f"已发送：{message}")
            
            # 递增计数器
            message_count += 1
            
            # 控制发送间隔（单位：秒，可根据需求调整）
            time.sleep(1)  # 每1秒发送一条消息
            
    except KeyboardInterrupt:
        print("\n客户端已手动停止发送")
    finally:
        # 关闭套接字和上下文
        socket.close()
        context.term()

if __name__ == "__main__":
    zmq_loop_client()