// NokovPoseReceiver.cpp
#include "nokov_receiver.h" // 确保这个包含了 sFrameOfMocapData 和 NokovSDKClient 的定义
#include <winsock2.h>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <thread>        // 添加缺失的头文件
#include <chrono>        // 添加缺失的头文件
#include <mutex>         // 添加缺失的头文件 (虽然 mutex 在 nokov_receiver.h 中用了，但最好也包含)

#pragma warning(disable : 4996)

// --- 工具函数：字符串转小写（用于名称匹配） ---
// (这部分看起来是正确的)
std::string toLower(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
    return r;
}

// --- 静态回调函数 ---
void __cdecl NokovPoseReceiver::DataHandler(sFrameOfMocapData* data, void* pUserData) {
    // --- 1. 获取 NokovPoseReceiver 实例指针 ---
    // pUserData 就是调用 SetDataCallback 时传入的 'this' 指针
    if (pUserData == nullptr) {
        std::cerr << "NokovPoseReceiver::DataHandler: pUserData is null!" << std::endl;
        return;
    }
    // 将 void* 转换回 NokovPoseReceiver*
    NokovPoseReceiver* receiver_instance = static_cast<NokovPoseReceiver*>(pUserData);

    // --- 2. 安全检查 ---
    if (data == nullptr || !receiver_instance->isRunning()) { // 使用 isRunning() 更安全
        return;
    }

    // --- 3. 处理数据 ---
    // NOKOV 时间戳 (假设单位是毫秒，需要根据 SDK 文档确认)
    int64_t ts_ms = data->iTimeStamp;

    for (int i = 0; i < data->nRigidBodies; ++i) {
        const sRigidBodyData& rb = data->RigidBodies[i];

        // 如果指定了名称，但尚未解析出 ID，则跳过
        // 注意：loadRigidBodyNames 应该在 start() 中调用一次，确保 ID 被设置
        // 如果 loadRigidBodyNames 是异步的或可能失败，这里需要更健壮的处理
        if (!receiver_instance->targetRigidBodyName.empty() && receiver_instance->targetRigidBodyId == -1) {
            // 可以选择在这里再次尝试加载，或者记录日志
            // std::cerr << "Target rigid body ID not resolved yet.\n";
            continue; // 或者 break; 取决于逻辑
        }

        // 匹配刚体
        bool matched = false;
        if (!receiver_instance->targetRigidBodyName.empty()) {
            // 如果指定了名称，匹配 ID
            matched = (rb.ID == receiver_instance->targetRigidBodyId);
        } else {
            // 未指定名称：取第一个刚体 (索引为0的)
            // 注意：你原来的逻辑是只要 matched=true 就处理，这会处理第一个。
            // 但如果目标ID是-1且未指定名称，也会处理第一个。
            // 更清晰的写法可能是 if (i == 0) matched = true;
            matched = (i == 0); // 明确处理第一个
        }

        if (matched) {
            // --- 4. 创建并填充 Pose 和 TimedPose ---
            // a. 创建 Pose 对象并填充 NOKOV 数据
            Pose current_pose;
            current_pose.x = rb.x / 1000.0;   // mm → m
            current_pose.y = rb.y / 1000.0;
            current_pose.z = rb.z / 1000.0;
            current_pose.qw = rb.qw;
            current_pose.qx = rb.qx;
            current_pose.qy = rb.qy;
            current_pose.qz = rb.qz;

            // b. 使用时间戳和 Pose 对象创建 TimedPose 对象
            // 注意：你的结构体成员是 timestamp_msec，不是 timestamp_ms
            TimedPose timed_pose(ts_ms, current_pose);

            // --- 5. 存入缓冲区 ---
            {
                std::lock_guard<std::mutex> lock(receiver_instance->bufferMutex);
                // 存储时也请注意 key 是 timestamp_msec
                receiver_instance->poseBuffer[ts_ms] = timed_pose;
            }

            // 调试输出（可选，记得同步成员名）
            // std::cout << "📥 NOKOV: t=" << timed_pose.timestamp_msec << "ms, pos=("
            //           << timed_pose.pose.x << "," << timed_pose.pose.y << "," << timed_pose.pose.z << ")\n";
            break; // 只处理一个匹配的刚体
        }
    }
}

// --- loadRigidBodyNames (这部分看起来基本正确，但注意线程安全) ---
void NokovPoseReceiver::loadRigidBodyNames() {
    if (!client) return;

    // 这个函数通常在 start() 中调用一次，此时只有一个线程，所以通常不需要锁
    // 但如果担心，可以在访问 targetRigidBodyId/Name 时加锁
    // std::lock_guard<std::mutex> lock(bufferMutex); // 如果需要

    // 重置 ID，以防重新连接
    targetRigidBodyId = -1;

    std::unordered_map<int, std::string> nameMap;
    sDataDescriptions* desc = nullptr;
    // 假设 ErrorCode_OK 和 Descriptor_RigidBody 是 SDK 定义的
    if (client->GetDataDescriptions(&desc) == ErrorCode_OK && desc) {
        for (int i = 0; i < desc->nDataDescriptions; ++i) {
            auto& d = desc->arrDataDescriptions[i];
            if (d.type == Descriptor_RigidBody) {
                int id = d.Data.RigidBodyDescription->ID;
                std::string name(d.Data.RigidBodyDescription->szName);
                nameMap[id] = name;

                // 如果名称匹配（不区分大小写）
                if (!targetRigidBodyName.empty() &&
                    toLower(name) == toLower(targetRigidBodyName)) {
                    targetRigidBodyId = id;
                    std::cout << "✅ 找到刚体 \"" << name << "\" (ID=" << id << ")\n";
                }
            }
        }
        // 假设 FreeDataDescriptions 是 SDK 提供的清理函数
        client->FreeDataDescriptions(desc);
    } else {
         std::cerr << "⚠️ 获取刚体描述信息失败或无描述信息。\n";
    }

    // 若指定了名称但未找到
    if (!targetRigidBodyName.empty() && targetRigidBodyId == -1) {
        std::cerr << "⚠️ 未找到刚体 \"" << targetRigidBodyName << "\"，将监听所有刚体中的第一个。\n";
    } else if (targetRigidBodyName.empty()) {
         std::cout << "ℹ️ 未指定刚体名称，将监听第一个可用刚体。\n";
    }
}

// --- 构造/析构函数 ---
NokovPoseReceiver::NokovPoseReceiver(const std::string& ip, const std::string& name)
    : serverIP(ip), targetRigidBodyName(name), targetRigidBodyId(-1), // 明确初始化
      running(false), client(nullptr), winsockInitialized(false) { // 明确初始化
    // 构造函数体可以为空
}

NokovPoseReceiver::~NokovPoseReceiver() {
    stop(); // 确保资源被释放
    if (winsockInitialized) {
        WSACleanup();
        winsockInitialized = false; // 重置标志
    }
}

// --- start 函数 ---
bool NokovPoseReceiver::start() {
    if (running) {
        std::cout << "NOKOV 接收器已在运行。\n";
        return false; // 已经在运行
    }

    // 初始化 Winsock
    if (!winsockInitialized) {
        WSADATA wsaData;
        int wsaResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (wsaResult != 0) {
            std::cerr << "❌ Winsock 初始化失败，错误码: " << wsaResult << "\n";
            return false;
        }
        winsockInitialized = true;
    }

    // 创建客户端实例
    client = new(std::nothrow) NokovSDKClient(); // 使用 nothrow 避免异常
    if (!client) {
        std::cerr << "❌ 创建 NokovSDKClient 实例失败 (内存不足?)\n";
        return false;
    }

    // 获取 SDK 版本
    unsigned char ver[4] = {0}; // 初始化数组
    client->NokovSDKVersion(ver);
    std::cout << "ℹ️ NOKOV SDK 版本: " << (int)ver[0] << "." << (int)ver[1]
              << "." << (int)ver[2] << "." << (int)ver[3] << "\n";

    // 连接到服务器
    int ret = client->Initialize(const_cast<char*>(serverIP.c_str()));
    if (ret != ErrorCode_OK) { // 确保 ErrorCode_OK 已定义
        std::cerr << "❌ 连接 NOKOV 服务器失败，错误码: " << ret << "\n";
        delete client; // 清理
        client = nullptr;
        return false;
    }

    // 加载刚体名称和 ID
    loadRigidBodyNames(); // 确保在设置回调前完成

    // 设置数据回调
    // 传递静态函数指针 DataHandler 和 this 指针
    int callbackResult = client->SetDataCallback(DataHandler, this);
    if (callbackResult != ErrorCode_OK) { // 检查回调设置是否成功
         std::cerr << "❌ 设置数据回调失败，错误码: " << callbackResult << "\n";
         client->Uninitialize();
         delete client;
         client = nullptr;
         return false;
    }

    // 启动后台线程 (如果需要)
    // 你的原始代码启动了一个空循环线程，这通常是不必要的，
    // 因为 SDK 会在内部处理数据接收。
    // 如果你确实需要一个后台任务，放入有意义的逻辑。
    // 否则，可以移除 receiverThread 相关代码。
    /*
    receiverThread = std::thread([this]() {
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // 这里可以放入周期性任务，如果没有，这个线程是多余的
        }
    });
    */

    running = true; // 设置运行标志

    std::cout << "📡 NOKOV 接收器已启动（目标刚体: "
              << (targetRigidBodyName.empty() ? "第一个" : targetRigidBodyName)
              << (targetRigidBodyId != -1 ? " (ID=" + std::to_string(targetRigidBodyId) + ")" : " (ID未解析)")
              << "）\n";
    return true;
}

// --- stop 函数 ---
void NokovPoseReceiver::stop() {
    if (!running) return; // 如果没在运行，直接返回
    running = false; // 设置停止标志

    // 停止并清理 SDK 客户端
    if (client) {
        client->Uninitialize(); // 通知 SDK 停止接收
        delete client; // 释放内存
        client = nullptr;
    }

    // 等待后台线程结束 (如果启动了的话)
    if (receiverThread.joinable()) {
        receiverThread.join();
    }

    std::cout << "⏹️ NOKOV 接收器已停止。\n";
}

// --- getClosestPose 函数 ---
bool NokovPoseReceiver::getClosestPose(int64_t timestamp_ms, TimedPose& out_pose) {
    std::lock_guard<std::mutex> lock(bufferMutex); // 保护缓冲区访问
    if (poseBuffer.empty()) {
        return false; // 缓冲区空，无法获取
    }

    // 使用 std::map 的 lower_bound 查找 >= timestamp_ms 的第一个元素
    auto it = poseBuffer.lower_bound(timestamp_ms);

    if (it == poseBuffer.end()) {
        // timestamp_ms 比所有时间戳都大，取最后一个（最新的）
        out_pose = std::prev(it)->second;
    } else if (it == poseBuffer.begin()) {
        // timestamp_ms 比所有时间戳都小，取第一个（最早的）
        out_pose = it->second;
    } else {
        // it 指向 >= timestamp_ms 的元素
        // prev 指向 < timestamp_ms 的元素
        auto prev = std::prev(it);
        int64_t dt1 = it->first - timestamp_ms;     // 距离 >= timestamp_ms 的时间差
        int64_t dt2 = timestamp_ms - prev->first;   // 距离 < timestamp_ms 的时间差

        // 选择时间差更小的那个
        if (dt2 <= dt1) {
            out_pose = prev->second;
        } else {
            out_pose = it->second;
        }
    }
    return true; // 成功找到并赋值
}