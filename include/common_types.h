// common_types.h
#pragma once
#include <cstdint>

// 纯位姿（位置 + 四元数）
struct Pose {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
};

// 带时间戳的位姿
struct TimedPose {
    int64_t timestamp_msec;
    Pose pose;

    // 便于构造
    TimedPose() = default;
    TimedPose(int64_t ts, const Pose& p) : timestamp_msec(ts), pose(p) {}
};