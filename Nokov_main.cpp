#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// 替换 ZMQ 为 NOKOV
#include "nokov_receiver.h"   // 👈 新增头文件
#include "camera_capture.h"
#include "laser_pointcloud.h"
#include "point_publisher.h"


// 全局标志用于响应 Ctrl+C
static std::atomic<bool> exit_requested(false);

// 全局 receiver 对象（用于 signal_handler 中停止）
std::unique_ptr<NokovPoseReceiver> g_nokovReceiver;  // 👈 改为 Nokov
CameraCapture* g_camera = nullptr;

// 信号处理函数
void signal_handler(int sig) {
    std::cout << "\n🛑 收到退出信号 (" << sig << ")，正在关闭...\n";
    exit_requested = true;
    if (g_camera) g_camera->stop();
    if (g_nokovReceiver) g_nokovReceiver->stop();  // 👈 停止 NOKOV
}

// // 激光点云发布器（保持不变）
// std::unique_ptr<LaserPublisher> laser_pub = std::make_unique<LaserPublisher>("tcp://*:5557");

int main() {
    // 注册信号处理
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ==============================
    // 🔧 启动 NOKOV 接收器
    // ==============================
    const std::string NOKOV_SERVER_IP = "192.168.1.101";  // 👈 改为你的真实 IP
    const std::string TARGET_RIGID_BODY = "AUV";         // 👈 改为你动捕软件中的刚体名

    g_nokovReceiver = std::make_unique<NokovPoseReceiver>(NOKOV_SERVER_IP, TARGET_RIGID_BODY);
    if (!g_nokovReceiver->start()) {
        std::cerr << "❌ NOKOV 接收器启动失败！\n";
        return -1;
    }

    // 等待 NOKOV 初始化完成（可选）
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    //接收点云的iP 
    const std::string SERVER_ADDR = "tcp://10.101.30.221:5557";
    LaserPublisher laserpub(SERVER_ADDR); // 传入地址

    // ==============================
    // 📷 初始化相机
    // ==============================
    CameraCapture camera("192.168.10.101", "192.168.5.105");
    g_camera = &camera;

    if (!camera.init()) {
        std::cerr << "相机初始化失败!" << std::endl;
        // 不立即退出，等待用户中断
        while (!exit_requested) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return -1;
    }

    camera.setSavePath("D:/激光数据/Image_Input/");
    if (!camera.start()) {
        std::cerr << "相机启动失败!" << std::endl;
        return -1;
    }

    std::cout << "✅ 相机与 NOKOV 动捕系统均已就绪，开始处理...\n";

    // 相机内参矩阵 K（根据你的标定结果填写）
    Eigen::Matrix3d K;
    K << 4308.8624, 0, 1379.5081,
         0, 4302.9958, 1031.0359,
         0, 0, 1;

    int frameCount = 0;
    int64_t lastTimestampMs = 0;

    // ==============================
    // 🔄 主循环：图像 + 位姿对齐 + 点云生成
    // ==============================
    while (!exit_requested) {
        cv::Mat image;
        uint64_t timestamp_ns;
        if (!camera.getImage(image, timestamp_ns)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 转换为毫秒（与 NOKOV 时间戳单位一致）
        int64_t timestamp_ms = static_cast<int64_t>(timestamp_ns / 1'000'000);

        // 防止重复帧（可选）
        if (timestamp_ms == lastTimestampMs) {
            continue;
        }
        lastTimestampMs = timestamp_ms;

        frameCount++;

        // ✅ 从 NOKOV 获取最接近的位姿
        TimedPose timedPose;
        if (!g_nokovReceiver->getClosestPose(timestamp_ms, timedPose)) {
            std::cerr << "[Frame " << frameCount << "] 未找到有效位姿，时间戳: " << timestamp_ms << " ms\n";
            continue;
        }

        // 提取点云
        int min_Point_Count = 400;
        auto pointCloud = fetchLaserPointCloud(image, min_Point_Count, K, timedPose.pose);

        // 发布点云
        laserpub.publish(frameCount, timestamp_ms, timedPose.pose, pointCloud);

        if (!pointCloud.empty()) {
            std::cout << "[Frame " << frameCount << "] 获取点云数量: " << pointCloud.size() << "\n";

            // 保存为 txt（可选）
            std::ofstream ofs("pointcloud_" + std::to_string(frameCount) + ".txt");
            for (const auto& [point, distance] : pointCloud) {
                ofs << point.x() << " " << point.y() << " " << point.z() << " " << distance << "\n";
            }
            ofs.close();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "👋 Goodbye!\n";
    return 0;
}