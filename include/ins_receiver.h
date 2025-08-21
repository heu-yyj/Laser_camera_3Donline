#pragma once
#include <Eigen/Geometry>
#include <map>
#include <mutex>
#include <thread>
#include <cstdint>
#include <numbers>
#include "common_types.h"

// C++20 标准方式（推荐）
inline constexpr double M_PI = std::numbers::pi;

struct InsData {
    double nav_x;
    double nav_y;
    double nav_depth;
    double nav_heading; // yaw (degree)
    double nav_pitch;   // pitch (degree)
    double nav_roll;    // roll (degree)
    double utc_time;    // timestamp (microseconds)
};

//四元数结构体
struct Quaternion {
    double w, x, y, z;
};

class INSReceiver  {
public: 
    INSReceiver(int port);
    ~INSReceiver();

    void start();
    void stop();
    bool getClosestPose(int64_t timestamp_msec, TimedPose& out_pose);

	void saveRawInsDataToFile(const std::string& filename);
	void savePoseToCSV(const std::string& filename);

private:
    int port;
    std::atomic<bool> running;
    std::thread receiverThread;

    std::map<int64_t, TimedPose> poseMap;
	std::map<int64_t, std::string> rawDataMap; // 用于存储原始 INS 数据
    std::mutex mutex;

    void receiverThreadFunc();
    bool parseInsData(const std::string& msg, InsData& data);
    static int64_t GpsWeekSecondsToUnixTimestamp(double seconds_into_week);
    static Quaternion eulerToQuaternion(double yaw_deg, double pitch_deg, double roll_deg);
    Pose buildPose(const InsData& insData);

};