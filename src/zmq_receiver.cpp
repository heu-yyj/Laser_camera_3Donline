// zmq_pose_receiver.cpp
#include "zmq_receiver.h"
#include <zmq.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

// 分割字符串
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}
struct TimedPose; // 前向声明

// 解析 ZMQ 消息（CSV格式：timestamp,x,y,z,qw,qx,qy,qz）
bool ZmqPoseReceiver::parsePose(const std::string& line, TimedPose& timedPose) {
    auto parts = split(line, ',');
    if (parts.size() != 8) return false;

    try {
        timedPose.timestamp_msec = std::stoll(parts[0]);
        timedPose.pose.x = std::stod(parts[1]);
        timedPose.pose.y = std::stod(parts[2]);
        timedPose.pose.z = std::stod(parts[3]);
        timedPose.pose.qw = std::stod(parts[4]);
        timedPose.pose.qx = std::stod(parts[5]);
        timedPose.pose.qy = std::stod(parts[6]);
        timedPose.pose.qz = std::stod(parts[7]);
        return true;
    }
    catch (...) {
        return false;
    }
}

// 构造函数
ZmqPoseReceiver::ZmqPoseReceiver(const std::string& addr)
    : address(addr), running(false) {}

// 析构函数
ZmqPoseReceiver::~ZmqPoseReceiver() {
    if (running) {
        stop();
    }
}

// 启动接收线程
void ZmqPoseReceiver::start() {
    if (running) return;
    running = true;
    receiverThread = std::thread(&ZmqPoseReceiver::receiverThreadFunc, this);
}

// 停止接收线程
void ZmqPoseReceiver::stop() {
    if (!running) return;
    running = false;
    if (receiverThread.joinable()) {
        receiverThread.join();
    }
}

// 接收线程函数
void ZmqPoseReceiver::receiverThreadFunc() {
    std::cout << "📡 正在连接 ZMQ: " << address << std::endl;

    zmq::context_t context(1);
    zmq::socket_t subscriber(context, ZMQ_SUB);
    subscriber.connect(address);
    subscriber.set(zmq::sockopt::subscribe, "");  // 订阅所有消息

    std::cout << "✅ 已连接，等待 ZMQ 数据..." << std::endl;

    while (running) {
        zmq::message_t message;
        auto result = subscriber.recv(message, zmq::recv_flags::dontwait);

        if (result && message.size() > 0) {
            std::string msg_str(static_cast<char*>(message.data()), message.size());
            TimedPose timedPose{};
            if (parsePose(msg_str, timedPose)) {
                std::lock_guard<std::mutex> lock(mutex);
                poseMap[timedPose.timestamp_msec] = timedPose;

                // 可选：打印调试
                std::cout << "📥 ZMQ 收到位姿: "
                          << "Time=" << timedPose.timestamp_msec << "ms"
                          << " | Pos=(" << timedPose.pose.x << "," << timedPose.pose.y << "," << timedPose.pose.z << ")"
                          << std::endl;
            } else {
                std::cerr << "❌ ZMQ 解析失败: " << msg_str << std::endl;
            }
        } else {
            // 避免CPU空转
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "🛑 ZMQ 接收器已停止。" << std::endl;
}

// 获取最接近的时间戳对应的位姿
// 注意：ZMQ 数据是 毫秒，所以需要转换比较
bool ZmqPoseReceiver::getClosestPose(int64_t timestamp_ms, TimedPose& out_pose) {
    std::lock_guard<std::mutex> lock(mutex);
    if (poseMap.empty()) return false;

    auto it = poseMap.lower_bound(timestamp_ms);
    if (it == poseMap.end()) {
        out_pose = std::prev(it)->second;
    }
    else if (it == poseMap.begin()) {
        out_pose = it->second;
    }
    else {
        auto prevIt = std::prev(it);
        int64_t diff1 = std::abs(it->first - timestamp_ms);
        int64_t diff2 = std::abs(prevIt->first - timestamp_ms);
        out_pose = (diff2 <= diff1) ? prevIt->second : it->second;
    }
    return true;
}
