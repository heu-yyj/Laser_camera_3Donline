#pragma once    
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include "camera_capture.h"
#include "ins_receiver.h"

// 定义 HSV 阈值范围组
extern const std::array<std::pair<cv::Scalar, cv::Scalar>, 3> HSV_RANGES;

// 函数声明

// 角度转换
double degreesToRadians(double degrees);

// 提取激光点
void extractLaserCoordinates(const cv::Mat& img,
    const std::pair<cv::Scalar, cv::Scalar>& hsvRange,
    std::vector<cv::Point>& filteredLaserPoints);

// 深度计算
std::vector<float> computeDepths(const std::vector<cv::Point>& laserPoints,
    const Eigen::Matrix3d& K);

// 图像坐标转世界坐标
std::vector<std::pair<Eigen::Vector3d, double>> imagePointsToWorld(
    const std::vector<cv::Point>& imgPoints,
    const std::vector<float>& depths,
    const Eigen::Matrix3d& K,
    const Pose& pose);

// 获取激光点云主函数
std::vector<std::pair<Eigen::Vector3d, double>> fetchLaserPointCloud(
    const cv::Mat& image,
    int minPointCount,
    const Eigen::Matrix3d& K,
    const Pose & pose);