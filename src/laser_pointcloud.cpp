#include "laser_pointcloud.h"
#include "ins_receiver.h"
#include "zmq_receiver.h"

using namespace cv;
using namespace std;

double degreesToRadians(double degrees) {
    return degrees * M_PI / 180.0;
}

// 定义多个可能的HSV阈值范围
const array<pair<Scalar, Scalar>, 3> HSV_RANGES = { {
    {Scalar(35, 100, 100), Scalar(85, 255, 255)}, // 默认范围
    {Scalar(35, 100, 60), Scalar(85, 255, 250)}, // 较暗的绿色
    {Scalar(35, 30, 30), Scalar(85, 255, 150)}   // 更暗的绿色
} };

// 定义一个函数用于提取激光坐标并进行滤波
void extractLaserCoordinates(const Mat& img, const pair<Scalar, Scalar>& hsvRange, vector<Point>& filteredLaserPoints) {
    Mat hsv, mask;
    // 转换到HSV颜色空间  
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // 阈值处理，提取特定范围的颜色部分  
    inRange(hsv, hsvRange.first, hsvRange.second, mask);

    // 形态学操作，去除噪声和连接断裂的条纹  
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel); // 先做开运算去除小噪点
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);   //闭运算连接可能断开的条纹

    vector<int> xCoordinates; // 存储所有激光点的X坐标
    vector<int> yCoordinates; // 存储所有激光点的Y坐标

    // 遍历每一行
    for (int y = 0; y < mask.rows; ++y) {
        uchar* maskRow = mask.ptr<uchar>(y); // 获取当前行在mask中的数据指针
        Vec3b* hsvRow = hsv.ptr<Vec3b>(y); // 获取当前行在原始HSV图像中的数据指针

        int best_x = -1; // 初始化为-1，表示尚未找到有效点
        uchar maxBrightness = 0; // 记录当前行的最大亮度值

        // 遍历当前行的所有列
        for (int x = 0; x < mask.cols; ++x) {
            if (maskRow[x] > 0) { // 找到非黑色像素即为有效点
                uchar brightness = hsvRow[x][2]; // 获取该点的亮度值（Value通道）
                if (brightness > maxBrightness) { // 检查是否是最亮的点
                    maxBrightness = brightness;
                    best_x = x;
                }
            }
        }

        // 如果找到了有效的点（即非黑色），则添加其X坐标和Y坐标到列表中
        if (best_x != -1) {
            xCoordinates.push_back(best_x);
            yCoordinates.push_back(y);
        }
    }

    // 设置最大允许的Y坐标间隔（可以根据实际情况调整）
    int max_y_gap = 200;

    // 设置最小Y坐标和最大Y坐标（避免靠近图像边缘的噪声）
    int min_y = 0;
    int max_y = mask.rows;

    // 初始化前一个点的坐标
    int prev_y = -1;

    // 初步筛选，去除边界点并检查连续性
    vector<Point> prelimFilteredLaserPoints;
    for (size_t i = 0; i < yCoordinates.size(); ++i) {
        int current_x = xCoordinates[i];
        int current_y = yCoordinates[i];

        // 检查当前点是否在合理的Y范围内
        if (current_y < min_y || current_y > max_y) {
            continue;
        }

        // 如果这是第一个点或者与前一个点在Y方向上连续
        if (prev_y == -1 || abs(current_y - prev_y) <= max_y_gap) {
            prelimFilteredLaserPoints.push_back(Point(current_x, current_y));
            prev_y = current_y;
        }
        else {
            // 如果当前点与前一个点不连续，则跳过此点
            // 可以根据需要决定是否重置prev_y
            prev_y = current_y;
        }
    }

    // 计算X坐标的统计信息
    if (!prelimFilteredLaserPoints.empty()) {
        vector<int> prelimXCoordinates;
        for (const auto& pt : prelimFilteredLaserPoints) {
            prelimXCoordinates.push_back(pt.x);
        }

        sort(prelimXCoordinates.begin(), prelimXCoordinates.end());
        double medianX = prelimXCoordinates[prelimXCoordinates.size() / 2]; // 使用中位数作为中心参考

        double stddev = 0;
        for (int x : prelimXCoordinates) {
            stddev += (x - medianX) * (x - medianX);
        }
        stddev = sqrt(stddev / prelimXCoordinates.size());

        // 根据标准差定义一个合理的范围
        double lowerBound = medianX - 2 * stddev;
        double upperBound = medianX + 2 * stddev;

        // 再次遍历初步筛选后的激光点，这次仅保留符合X坐标范围的点
        for (const auto& pt : prelimFilteredLaserPoints) {
            if (pt.x >= lowerBound && pt.x <= upperBound) {
                filteredLaserPoints.push_back(pt);
            }
        }
    }

}

std::vector<float> computeDepths(
    const std::vector<cv::Point>& laserPoints,
    const Eigen::Matrix3d& K)
{
    std::vector<float> depths(laserPoints.size());

    double cx = K(0, 2); // 内参标定获取的cx
    double fx = K(0, 0); // 内参标定获取的fx
    double u = 0.00345;  // 每像素代表的距离
    double f = fx * u;
    double s = 220;      // 实测相机-激光器距离
    double A_degrees = 19.6; // 激光线与其法线夹角
    double A = degreesToRadians(A_degrees);

    for (size_t i = 0; i < laserPoints.size(); ++i) {
        double c = laserPoints[i].x; // 激光点横坐标
        double d = (c - cx) * u; // 计算 d
        double tanB = d / f;     // 计算 tan B
        double B = std::atan(tanB); // 计算 B
        double H = s / (std::tan(A) + tanB); // 计算 H
        depths[i] = static_cast<float>(H);
    }

    return depths;
}

// 将图像坐标转换为世界坐标系的三维点
std::vector<std::pair<Eigen::Vector3d,double>>imagePointsToWorld(
    const std::vector<cv::Point>& imgPoints,
    const std::vector<float>& depths,
    const Eigen::Matrix3d& K,
    const Pose& pose) 
{
    std::vector<std::pair<Eigen::Vector3d, double>> world_points_d;
    for (size_t i = 0; i < imgPoints.size(); ++i) {
        Eigen::Vector3d cam_point;
        if (depths.empty() || i >= depths.size()) {
            continue;  // 跳过该点的处理
        }
        else {
            cam_point = depths[i] * (K.inverse() * Eigen::Vector3d(imgPoints[i].x, imgPoints[i].y, 1.0));
        }
        //AUV(INS)坐标系到相机的欧拉角
		double yaw = 199.6 * M_PI / 180.0; // 假设的偏航角度（弧度制）
		double pitch = 0.0; // 假设的俯仰角度（弧度制）
		double roll = 90 * M_PI / 180.0; // 假设的翻滚角度（弧度制）

        Eigen::Matrix3d Rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Matrix3d Ry = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d Rx = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();

        Eigen::Matrix3d R_wc = Rz * Ry * Rx; // ZYX顺序
		Eigen::Vector3d T_cw(424.0, 27.4, 247.6); // 相机坐标系在AUV坐标系中的位置 mm

        //每个位姿下的点(AUV坐标系下)
        Eigen::Vector3d auv_point = R_wc * cam_point + T_cw; //转换到AUV(INS)坐标系
        

        // 将四元数转换为旋转矩阵
        Eigen::Quaterniond quat(pose.qw, pose.qx, pose.qy, pose.qz);
        Eigen::Matrix3d R_auv = quat.toRotationMatrix();
        // 构造平移向量
        Eigen::Vector3d T_auv(pose.x, pose.y, pose.z);

        // 世界坐标系下的点
		Eigen::Vector3d world_point = R_auv * auv_point + T_auv;

        // 计算到相机原点的距离
        double distance = std::sqrt(cam_point.squaredNorm());
       
        world_points_d.emplace_back(world_point,distance);
    }
    return world_points_d;
}

std::vector<std::pair<Eigen::Vector3d, double>> fetchLaserPointCloud(
    const cv::Mat &image,
    int minPointCount,
    const Eigen::Matrix3d& K,
    const Pose & pose)
{
    bool found_good_range = false;
    vector<Point> laser_points;

    for (const auto& [lower_green, upper_green] : HSV_RANGES) {
        laser_points.clear();
        extractLaserCoordinates(image, { lower_green, upper_green }, laser_points);
        if (laser_points.size() >= minPointCount) {
            found_good_range = true;
            break;
        }
    }

    if (!found_good_range) {
        cout << "未能找到合适的HSV范围，无法提取足够的激光点。" << endl;
        return {};
    }

    // 单独调用深度计算函数
    std::vector<float> depths = computeDepths(laser_points, K);

    // 调用坐标转换函数
    return imagePointsToWorld(laser_points, depths, K, pose);
}