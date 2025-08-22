#include "point_publisher.h"
#include <iostream>

using json = nlohmann::json;

LaserPublisher::LaserPublisher(const std::string& endpoint)
    : context_(1), socket_(context_, ZMQ_PUB) {
    socket_.bind(endpoint);
    std::cout << "[LaserPublisher] 绑定到: " << endpoint << std::endl;
}

LaserPublisher::~LaserPublisher() {
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

    // ✅ 直接访问 Pose 的成员变量
    Eigen::Vector3d t(pose.x, pose.y, pose.z);
    Eigen::Quaterniond q(pose.qw, pose.qx, pose.qy, pose.qz); // 注意：qw 在前！

    j["header"]["pose"] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};

    for (const auto& [pt, dist] : pointCloud) {
        j["points"].push_back({pt.x(), pt.y(), pt.z(), dist});
    }

    std::string payload = j.dump();
    zmq::message_t msg(payload.size());
    std::memcpy(msg.data(), payload.data(), payload.size());
    socket_.send(msg, zmq::send_flags::none);

    std::cout << "[LaserPublisher] 发布帧 " << frame_idx 
              << "，点数: " << pointCloud.size() << std::endl;
}
