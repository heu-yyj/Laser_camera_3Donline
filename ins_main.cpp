#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <fstream>

#include "ins_receiver.h"
#include "camera_capture.h"
#include "laser_pointcloud.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

std::atomic<bool> g_running(true);

// 全局标志用于响应 Ctrl+C
static std::atomic<bool> exit_requested(false);


void signalHandler(int sig) {
    std::cout << "\nReceived shutdown signal (" << sig << ")..." << std::endl;
     exit_requested = true;
    g_running = false;
}

int main() {
    // 注册信号处理（Ctrl+C）
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // 启动 INS 接收器
    INSReceiver insReceiver(8006);
    insReceiver.start();

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
        std::cerr << "Failed to start camera capture!" << std::endl;
        return -1;
    }

    std::cout << "Camera started. Processing images..." << std::endl;

    // 相机内参矩阵 K（根据你的标定结果填写）
    Eigen::Matrix3d K;
    K << 4308.8624, 0, 1379.5081, // fx, cx
        0, 4302.9958, 1031.0359, // fy, cy
        0, 0, 1;

    int frameCount = 0;
    int64_t lastTimestamp = 0;

    while (g_running) {
        // 获取图像和时间戳
        cv::Mat image;
        uint64_t timestamp_ns;
        if (!camera.getImage(image, timestamp_ns)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 转换为毫秒（与 INS 时间戳单位一致）
        int64_t timestamp_ms = timestamp_ns / 1000000;

        // 防止重复处理同一帧（可选）
        if (timestamp_ms == lastTimestamp) {
            continue;
        }
        lastTimestamp = timestamp_ms;

        // 获取最接近的 INS 位姿
        TimedPose timedPose;
        if (!insReceiver.getClosestPose(timestamp_ms, timedPose)) {
            std::cerr << "[Frame " << ++frameCount << "] No valid pose found for timestamp: " << timestamp_ms << std::endl;
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

    // 程序退出前保存数据
    std::cout << "Saving INS data..." << std::endl;
    insReceiver.saveRawInsDataToFile("ins_raw_data.txt");
    insReceiver.savePoseToCSV("ins_poses.csv");

    std::cout << "Shutting down..." << std::endl;
    camera.stop();
    insReceiver.stop();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}