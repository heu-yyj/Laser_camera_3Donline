#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <fstream>

#include "zmq_receiver.h"
#include "camera_capture.h"
#include "laser_pointcloud.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// 全局标志用于响应 Ctrl+C
static std::atomic<bool> exit_requested(false);

// 全局 receiver 对象（用于 signal_handler 中停止）
std::unique_ptr<ZmqPoseReceiver> g_zmqReceiver;
CameraCapture *g_camera = nullptr;  //  全局相机指针

// 信号处理函数
void signal_handler(int sig) {
    std::cout << "\n🛑 收到退出信号 (" << sig << ")，正在关闭...\n";
    exit_requested = true;
    g_camera->stop();
    g_zmqReceiver->stop();
}

int main() {
    // 注册信号处理
    std::signal(SIGINT, signal_handler);   // Ctrl+C
    std::signal(SIGTERM, signal_handler);  // 终止信号

    // 启动 ZMQ 接收器,5556为处理后的数据，5555为原始UDP位姿数据
    g_zmqReceiver = std::make_unique<ZmqPoseReceiver>("tcp://localhost:5556");
    g_zmqReceiver->start();

    // 等待 ZMQ 接收器启动
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 初始化相机
    CameraCapture camera("192.168.10.101", "192.168.5.105");
    if (!camera.init()) {
        std::cerr << "相机初始化失败!" << std::endl;
         // ❌ 不 return，不重试，只是挂起等待
        while (!exit_requested) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        }
         return -1;
    
    }

    camera.setSavePath("C:/Users/15158/Desktop/Image_Input/");
    if (!camera.start()) {
        std::cerr << "相机启动失败!" << std::endl;
        return -1;
    }

    std::cout << "相机已启动. 正在处理图像..." << std::endl;

    // 相机内参矩阵 K（根据你的标定结果填写）
    Eigen::Matrix3d K;
    K << 4308.8624, 0, 1379.5081, // fx, cx
        0, 4302.9958, 1031.0359, // fy, cy
        0, 0, 1;

    int frameCount = 0;
    int64_t lastTimestampMs = 0;

    while (!exit_requested) {
        // 获取图像和时间戳
        cv::Mat image;
        uint64_t timestamp_ns;
        if (!camera.getImage(image, timestamp_ns)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 转换为毫秒（与ZMQ时间戳单位一致）
        int64_t timestamp_ms = timestamp_ns / 1000000;

        // 防止重复处理同一帧（可选）
        if (timestamp_ms == lastTimestampMs) {
            continue;
        }
        lastTimestampMs = timestamp_ms;

        // 获取最接近的位姿
        TimedPose timedPose;
        if (!g_zmqReceiver->getClosestPose(timestamp_ms, timedPose)) {
            std::cerr << "[Frame " << ++frameCount << "] 未找到有效位姿，时间戳: " << timestamp_ms << std::endl;
            continue;
        }

        // 提取点云
        int min_Point_Count = 400;
        auto pointCloud = fetchLaserPointCloud(image, min_Point_Count, K, timedPose.pose);

        if (!pointCloud.empty()) {
            std::cout << "[Frame " << ++frameCount << "] 获取到点云数量: " << pointCloud.size() << std::endl;

            // 保存为 txt 文件（可选：保存为 .ply）
            std::ofstream ofs("pointcloud_" + std::to_string(frameCount) + ".txt");
            for (const auto& [point, distance] : pointCloud) {
                ofs << point.x() << " " << point.y() << " " << point.z() << " " << distance << "\n";
            }
            ofs.close();
        }

        // 控制 CPU 占用
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
   
    // 主线程等待退出信号
    while (!exit_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Goodbye!" << std::endl;
    return 0;
}