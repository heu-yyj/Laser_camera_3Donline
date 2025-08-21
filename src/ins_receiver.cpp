#include "ins_receiver.h"
#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <sstream>
#include <chrono>  
#include <fstream>

#pragma comment(lib, "ws2_32.lib")

// 常量定义
namespace TimeConstants {
    const int64_t GPS_EPOCH_SEC = 315964800;  // 1980-01-06 UTC 的 Unix 时间戳（秒）
    const int64_t SECS_PER_WEEK = 7LL * 24 * 60 * 60;
}

// 解析 INS 文本数据
bool INSReceiver::parseInsData(const std::string& msg, InsData& data) {
    try {
        int matched = sscanf_s(
            msg.c_str(),
            "%*[^N]N" "AV_X:%lf;"     // 匹配 NAV_X
            "%*[^N]N" "AV_Y:%lf;"     // 匹配 NAV_Y
            "%*[^N]N" "AV_DEPTH:%lf;" // 匹配 NAV_DEPTH
            "%*[^N]N" "AV_HEADING:%lf;" // 匹配 NAV_HEADING
            "%*[^N]N" "AV_PITCH:%lf;"   // 匹配 NAV_PITCH
            "%*[^N]N" "AV_ROLL:%lf;"    // 匹配 NAV_ROLL
            "%*[^U]U" "TC_TIME:%lf;",   // 匹配 UTC_TIME
            &data.nav_x,
            &data.nav_y,
            &data.nav_depth,
            &data.nav_heading,
            &data.nav_pitch,
            &data.nav_roll,
            &data.utc_time);

        if (matched != 7) {
            std::cerr << "Failed to match all required fields in INS data." << std::endl;
            return false;
        }

        return true;
    }
    catch (...) {
        std::cerr << "Exception during parsing INS data." << std::endl;
        return false;
    }
}

// 欧拉角转四元数（ZYX顺序：yaw → pitch → roll）
Quaternion INSReceiver::eulerToQuaternion(double yaw_deg, double pitch_deg, double roll_deg) {
    double yaw = yaw_deg * M_PI / 180.0;
    double pitch = pitch_deg * M_PI / 180.0;
    double roll = roll_deg * M_PI / 180.0;

    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

Pose INSReceiver::buildPose(const InsData& insData) {

	// 归一化 heading 到 [0, 360)
	// 对 pitch 取反（因为它是左手坐标系定义）  
    double heading = fmod(insData.nav_heading, 360.0);
    if (heading < 0) heading += 360.0;

    double corrected_pitch = -insData.nav_pitch;

    Quaternion quat = eulerToQuaternion(heading, corrected_pitch, insData.nav_roll);
 
    // 构造 Pose
    Pose pose;
    pose.x = insData.nav_x * 1000;  // 转换为毫米
    pose.y = insData.nav_y * 1000;
    pose.z = insData.nav_depth * 1000; // 深度转为 Z 轴
    pose.qw = quat.w;
    pose.qx = quat.x;
    pose.qy = quat.y;
    pose.qz = quat.z;

    return pose;
}

// 周内秒转 Unix 时间戳（毫秒）
int64_t INSReceiver::GpsWeekSecondsToUnixTimestamp(double seconds_into_week) {
    auto now = std::chrono::system_clock::now();
    time_t now_sec = std::chrono::system_clock::to_time_t(now);

    int64_t secs_since_gps_epoch = now_sec - TimeConstants::GPS_EPOCH_SEC;
    int64_t weeks = secs_since_gps_epoch / TimeConstants::SECS_PER_WEEK;

    int64_t millis_into_week = static_cast<int64_t>(seconds_into_week * 1000.0 + 0.5);

    int64_t total_millis = TimeConstants::GPS_EPOCH_SEC * 1000LL +
        weeks * TimeConstants::SECS_PER_WEEK * 1000LL +
        millis_into_week;

    return total_millis; // 返回 Unix 毫秒时间戳

}

void INSReceiver::receiverThreadFunc() {
    WSADATA wsaData;
    SOCKET sock;
    sockaddr_in serverAddr, clientAddr;
    char buffer[4096];
    int clientAddrLen = sizeof(clientAddr);

    WSAStartup(MAKEWORD(2, 2), &wsaData);

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed!" << std::endl;
        return;
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);

    if (bind(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed!" << std::endl;
        closesocket(sock);
        WSACleanup();
        return;
    }

    std::cout << "INS Receiver started, waiting for UDP packets on port " << port << "..." << std::endl;

    while (running) {
        int len = recvfrom(sock, buffer, sizeof(buffer), 0, (sockaddr*)&clientAddr, &clientAddrLen);
        if (len > 0) {
            buffer[len] = '\0'; // 确保字符串结尾
            std::string data(buffer);

            InsData insData;
            if (!parseInsData(data, insData)) {
                std::cerr << "Failed to parse INS data." << std::endl;
                continue;
            }
           
            Pose pose = buildPose(insData);

            // 转换为 Unix 时间戳（毫秒）
            int64_t gps_unix_msec = GpsWeekSecondsToUnixTimestamp(insData.utc_time);

            // 存储到 map
            std::lock_guard<std::mutex> lock(mutex);
            poseMap[gps_unix_msec] = TimedPose(gps_unix_msec, pose);
			rawDataMap[gps_unix_msec] = data; // 保存原始数据

            // 打印调试信息
            std::cout << "Received INS data: " << std::fixed << std::setprecision(6)
                << "Time: " << gps_unix_msec
                << " | Pos: (" << insData.nav_x << ", " << insData.nav_y << ", " << -insData.nav_depth << ")"
                << " | YPR: (" << insData.nav_heading << ", " << insData.nav_pitch << ", " << insData.nav_roll << ")"
                << std::endl;
        }
    }

    closesocket(sock);
    WSACleanup();
}

INSReceiver::INSReceiver(int port)
    : port(port), running(false) {
}

INSReceiver::~INSReceiver() {
    if (running) {
        stop();
    }
}

void INSReceiver::start() {
    running = true;
    receiverThread = std::thread(&INSReceiver::receiverThreadFunc, this);
}

void INSReceiver::stop() {
    running = false;
    if (receiverThread.joinable()) {
        receiverThread.join();
    }
}

// ins_receiver.cpp
bool INSReceiver::getClosestPose(int64_t timestamp_ms, TimedPose& out_pose) {
    std::lock_guard<std::mutex> lock(mutex);
    if (poseMap.empty()) return false;

    const int64_t MAX_TIME_DIFF = 150; // 最大时间差，单位毫秒

    auto it = poseMap.lower_bound(timestamp_ms);
    if (it == poseMap.end()) {
        out_pose = std::prev(it)->second;
    }
    else if (it == poseMap.begin()) {
        out_pose = it->second;
    }
    else {
        auto prevIt = std::prev(it);
        int64_t diff1 = std::abs(it->first - timestamp_ms);
        int64_t diff2 = std::abs(prevIt->first - timestamp_ms);
        out_pose = (diff2 <= diff1) ? prevIt->second : it->second;
    }

    // 检查时间差是否过大
    int64_t actual_diff = llabs(out_pose.timestamp_msec - timestamp_ms);
    if (actual_diff > MAX_TIME_DIFF) {
        return false;
    }
    return true;
}

// 保存原始 INS 数据到 txt 文件
void INSReceiver::saveRawInsDataToFile(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::app); // 以追加方式写入
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for raw INS data: " << filename << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [timestamp, rawData] : rawDataMap) {
        ofs << timestamp << " | " << rawData << "\n"; // 可扩展为保存原始字符串
    }

    ofs.close();
}

// 保存位姿为 CSV 文件（时间戳, x, y, z, qx, qy, qz, qw）
void INSReceiver::savePoseToCSV(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::app); // 追加写入
    if (!ofs.is_open()) {
        std::cerr << "Failed to open CSV file for saving poses: " << filename << std::endl;
        return;
    }

    // 写入 CSV 头（仅第一次写入）
    static bool headerWritten = false;
    if (!headerWritten) {
        ofs << "timestamp_ms,x,y,z,qx,qy,qz,qw\n";
        headerWritten = true;
    }

    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [timestamp, timedPose] : poseMap) {
        ofs << timestamp << ","
            << timedPose.pose.x << "," << timedPose.pose.y << "," << timedPose.pose.z << ","
            << timedPose.pose.qx << "," << timedPose.pose.qy << "," << timedPose.pose.qz << "," << timedPose.pose.qw << "\n";
    }

    ofs.close();
}

