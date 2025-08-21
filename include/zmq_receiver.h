// zmq_receiver.h
#pragma once

#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include "common_types.h"

class ZmqPoseReceiver {
public:
    explicit ZmqPoseReceiver(const std::string& address = "tcp://localhost:5556");
    ~ZmqPoseReceiver();

    // 启动接收线程
    void start();

    // 停止接收线程
    void stop();

    // 获取最接近指定时间戳的位姿（输入时间单位：毫秒）
    bool getClosestPose(int64_t timestamp_mesc, TimedPose& out_pose) ;

private:
    std::string address;
    std::atomic<bool> running;
    std::thread receiverThread;

    // 时间戳 -> 位姿 映射（缓存）
    std::map<int64_t, TimedPose> poseMap;  // key 是 timestamp_msec
    std::mutex mutex;

    // 内部线程函数
    void receiverThreadFunc();

    // 解析 ZMQ 消息
    bool parsePose(const std::string& line, TimedPose& timedPose);
};
