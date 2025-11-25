import zmq

def zmq_server():
    # 创建ZeroMQ上下文
    context = zmq.Context()
    
    # 创建PULL类型套接字（用于接收消息，无回复）
    # PULL可以接收多个PUSH客户端发送的消息，适合单向接收场景
    socket = context.socket(zmq.PULL)
    
    # 绑定到所有网络接口的5555端口（*表示接受所有来源的连接）
    # 端口可根据需要修改（1024-65535之间未占用的端口）
    socket.bind("tcp://*:5557")
    print("ZMQ服务器已启动，绑定端口：5557，等待接收消息...")
    print("（按Ctrl+C停止服务器）")
    
    try:
        # 循环接收消息
        while True:
            # 接收消息（默认接收字节类型，可根据需要转换）
            # 若发送方用send_string()，这里可用recv_string()直接接收字符串
            message = socket.recv_string()  # 接收字符串类型消息
            # 若需要接收字节类型，用：message = socket.recv()
            
            print(f"收到消息：{message}")
            
    except KeyboardInterrupt:
        print("\n服务器已手动停止")
    finally:
        # 关闭套接字和上下文
        socket.close()
        context.term()

if __name__ == "__main__":
    zmq_server()