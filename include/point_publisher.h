#pragma once
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <vector>
#include "common_types.h"

class LaserPublisher {
public:
    explicit LaserPublisher(const std::string& target_address);
    ~LaserPublisher();

    // 发布点云
    void publish(
        int frame_idx,
        int64_t timestamp_ms,
        const Pose& pose,
        const std::vector<std::pair<Eigen::Vector3d, double>>& pointCloud
    );

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
};