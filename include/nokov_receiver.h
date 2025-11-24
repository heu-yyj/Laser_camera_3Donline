// NokovPoseReceiver.h
#pragma once
#include <string>
#include <thread>
#include <mutex>
#include <map>
#include "common_types.h"
#include "NokovSDKTypes.h"
#include "NokovSDKClient.h"
#include <unordered_map>

class NokovPoseReceiver {
public:
    // 构造函数：
    // - serverIP: NOKOV 服务器地址（如 "192.168.1.100"）
    // - rigidBodyName: 要监听的刚体名称（如 "AUV1"），若为空则取第一个刚体
    explicit NokovPoseReceiver(const std::string& serverIP, const std::string& rigidBodyName = "");
    ~NokovPoseReceiver();

    bool start();   // 启动接收
    void stop();    // 停止接收
    bool isRunning() const { return running; }

    // 获取最接近 timestamp_ms 的位姿（单位：毫秒）
    bool getClosestPose(int64_t timestamp_ms, TimedPose& out_pose);

private:
    static void __cdecl DataHandler(sFrameOfMocapData* data, void* pUserData);
    void loadRigidBodyNames();

    std::string serverIP;
    std::string targetRigidBodyName;
    int targetRigidBodyId = -1;  // 解析后填充

    bool running = false;
    std::thread receiverThread;
    class NokovSDKClient* client = nullptr;

    // 缓存：仅一个刚体的历史位姿（按时间戳排序）
    std::map<int64_t, TimedPose> poseBuffer;
    mutable std::mutex bufferMutex;

    bool winsockInitialized = false;
};