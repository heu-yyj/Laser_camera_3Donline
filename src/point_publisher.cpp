#include "point_publisher.h"
#include <iostream>

using json = nlohmann::json;

LaserPublisher::LaserPublisher(const std::string& target_address)
    : context_(1), socket_(context_, ZMQ_PUSH) {
    socket_.connect(target_address);
    std::cout << "[LaserPublisher] 连接到融合节点: " << target_address << std::endl;
}

LaserPublisher::~LaserPublisher() {
    // socket_.set(zmq::sockopt::linger, 0);
    socket_.close();
}

void LaserPublisher::publish(
    int frame_idx,
    int64_t timestamp_ms,
    const Pose& pose,
    const std::vector<std::pair<Eigen::Vector3d, double>>& pointCloud
) {
    json j;
    j["header"]["timestamp"] = timestamp_ms;
    j["header"]["frame_id"] = "laser";
    j["header"]["frame_idx"] = frame_idx;

   // 确保四元数是单位四元数（规范化总是安全的）
    Eigen::Quaterniond q_normalized(pose.qw, pose.qx, pose.qy, pose.qz);
    q_normalized.normalize();

    // 2. 构造 4x4 变换矩阵 T = | R t |
    //                          | 0 1 |
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity(); // 初始化为单位矩阵
    T.block<3,3>(0,0) = q_normalized.toRotationMatrix();        // 左上角 3x3 块设置为旋转矩阵 R
    T.block<3,1>(0,3) = Eigen::Vector3d(pose.x, pose.y, pose.z);                           // 右上角 3x1 块设置为平移向量 t
    // 最后一行 [0, 0, 0, 1] 已由 Identity() 初始化

     // 3. 将 4x4 矩阵转换为 JSON 数组 (行优先)
    json pose_matrix_json = json::array();
    for (int i = 0; i < 4; ++i) { // 遍历行
        json row = json::array();
        for (int j = 0; j < 4; ++j) { // 遍历列
             row.push_back(T(i, j));
        }
        pose_matrix_json.push_back(row);
    }
    
    // 4. 将矩阵存入 JSON
    j["header"]["pose_matrix"] = pose_matrix_json;
    for (const auto& [pt, _] : pointCloud) {
        j["points"].push_back({pt.x(), pt.y(), pt.z()});
    }

    std::string payload = j.dump();
    zmq::message_t msg(payload.size());
    std::memcpy(msg.data(), payload.data(), payload.size());
    socket_.send(msg, zmq::send_flags::none);

    std::cout << "[LaserPublisher] 发布帧 " << frame_idx 
              << "，点数: " << pointCloud.size() << std::endl;
}
