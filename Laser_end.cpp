#include <iostream>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"         // 使用相对路径
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "MvCameraControl.h"

using json = nlohmann::json;

// 全局控制变量
std::atomic<bool> g_exitRequested{false};

// 行间隔参数 (用户输入的“隔N行” -> 实际处理间隔为 N+1)
// ROW_INTERVAL = N+1, 其中 N 是用户想要隔开的行数
// 例如，隔1行(N=1) -> ROW_INTERVAL=2 (处理0, 跳1, 处理2, 跳3, ...)
const int ROW_INTERVAL = 0 + 1; // 修改这里，例如：想要隔1行，设置为 1+1=2；想要隔2行，设置为 2+1=3；想要不隔(每行都处理)，设置为 0+1=1

// 帧间隔参数（用户输入的“隔N帧” -> 实际处理间隔为 N+1）
// FRAME_INTERVAL = M+1, 其中 M 是用户想要隔开的帧数
// 例如，隔1帧(M=1) -> FRAME_INTERVAL=2 (处理0, 跳1, 处理2, 跳3, ...)
const int FRAME_INTERVAL = 1 + 1; // 修改这里，例如：想要隔1帧，设置为 1+1=2；想要隔2帧，设置为 2+1=3；想要不隔(每帧都处理)，设置为 0+1=1
std::atomic<int> g_frameCounter{0}; // 帧计数器，用于跟踪接收的帧数

// 最小点数要求
const int MIN_POINTS_REQUIRED = 400;

// --- 新增：相机标定参数 ---
const double CAMERA_MATRIX_DATA[] = { 4308.8624, 0.0,       1379.5081,
                                      0.0,       4302.9958, 1031.0359,
                                      0.0,       0.0,       1.0 };
const cv::Matx33d CAMERA_MATRIX(CAMERA_MATRIX_DATA);

const double DIST_COEFFS_DATA[] = { 0.26674, -0.15964, -0.00157, -0.01023 };
const cv::Mat DIST_COEFFS(1, 4, CV_64F, (void*)DIST_COEFFS_DATA);
// --- END ---

// 图像帧结构
struct ImageFrame {
    uint64_t timestamp;
    std::vector<uint8_t> data;
    MV_FRAME_OUT_INFO_EX info;

    ImageFrame() = default;
    ImageFrame(uint64_t ts, const uint8_t* src, const MV_FRAME_OUT_INFO_EX& frameInfo)
        : timestamp(ts), data(src, src + frameInfo.nFrameLen), info(frameInfo) {}
};

// 激光点结构 (修改为Point类型)
struct LaserPoints {
    uint64_t timestamp;
    std::vector<cv::Point> points; // 使用整数坐标
};

// 线程安全的图像缓存（最多保留 N 帧）
class ImageBuffer {
public:
    static constexpr size_t MAX_FRAMES = 30;

    void addFrame(uint64_t ts, const uint8_t* data, const MV_FRAME_OUT_INFO_EX& info) {
        if (!data) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.size() >= MAX_FRAMES) buffer_.erase(buffer_.begin());
        buffer_.emplace(ts, ImageFrame(ts, data, info));
    }

private:
    mutable std::mutex mutex_;
    std::map<uint64_t, ImageFrame> buffer_;
};
ImageBuffer g_imageBuffer;

// 激光点缓存 (修改为Point类型)
class LaserBuffer {
public:
    static constexpr size_t MAX_ENTRIES = 30;
    void addLaserPoints(uint64_t ts, const std::vector<cv::Point>& pts) { // 参数类型改为Point
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.size() >= MAX_ENTRIES) buffer_.erase(buffer_.begin());
        buffer_[ts] = {ts, pts};
    }

    std::optional<LaserPoints> getLatest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) return std::nullopt;
        return buffer_.rbegin()->second;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::map<uint64_t, LaserPoints> buffer_;
};
LaserBuffer g_laserBuffer;

// HSV 阈值范围
const std::array<std::pair<cv::Scalar, cv::Scalar>, 3> HSV_RANGES = {{
    {cv::Scalar(35, 100, 100), cv::Scalar(85, 255, 255)},
    {cv::Scalar(35, 100, 60),  cv::Scalar(85, 255, 250)},
    {cv::Scalar(35, 30, 30),   cv::Scalar(85, 255, 150)}
}};

// 提取激光坐标函数 (返回Point类型的点)
void extractLaserCoordinates(const cv::Mat& img, const std::pair<cv::Scalar, cv::Scalar>& hsvRange,
                             std::vector<cv::Point>& filteredLaserPoints, int rowInterval = 1) { // 返回类型改为Point
    cv::Mat hsv, mask;
    cvtColor(img, hsv, cv::COLOR_BGR2HSV); // BGR 转 HSV
    inRange(hsv, hsvRange.first, hsvRange.second, mask); // 应用阈值，生成掩码

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 定义形态学操作核
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);  // 开运算，去除噪点
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel); // 闭运算，连接断裂

    for (int y = 0; y < mask.rows; y += rowInterval) { // 根据间隔遍历每一行
        uchar* maskRow = mask.ptr<uchar>(y); // 获取掩码行指针
        cv::Vec3b* hsvRow = hsv.ptr<cv::Vec3b>(y); // 获取HSV行指针
        int best_x = -1;
        uchar maxV = 0;
        for (int x = 0; x < mask.cols; ++x) { // 遍历行内每个像素
            if (maskRow[x] && hsvRow[x][2] > maxV) { // 如果在掩码中，且V值更高
                maxV = hsvRow[x][2]; // 记录最大V值
                best_x = x;          // 记录对应的x坐标
            }
        }
        if (best_x != -1) { // 如果该行找到了符合条件的点
            filteredLaserPoints.emplace_back(best_x, y); // 存储坐标，直接使用整数
        }
    }
}

// 去畸变函数 (修改为处理Point类型)
void undistortLaserPoints(std::vector<cv::Point>& points) {
    if (points.empty()) return;

    // 将整数点转换为浮点点用于undistortPoints
    std::vector<cv::Point2f> points_f;
    points_f.reserve(points.size());
    for (const auto& pt : points) {
        points_f.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }

    std::vector<cv::Point2f> undistorted_points_f;
    cv::undistortPoints(points_f, undistorted_points_f, CAMERA_MATRIX, DIST_COEFFS, cv::noArray(), CAMERA_MATRIX);

    // 将去畸变后的浮点点转换回整数点
    points.clear(); // 清空原容器
    points.reserve(undistorted_points_f.size()); // 预分配空间
    for (const auto& pt_f : undistorted_points_f) {
        // 四舍五入到最近的整数
        points.emplace_back(cv::Point(static_cast<int>(std::round(pt_f.x)), static_cast<int>(std::round(pt_f.y))));
    }
}


void publishLaserDataAsJson(const LaserPoints& laserData, const char* targetIp, int targetPort) {
    try {
        json j;
        j["timestamp"] = laserData.timestamp;
        json pointsArray = json::array();
        for (const auto& pt : laserData.points) {
            // 发送整数坐标
            pointsArray.push_back({{"x", pt.x}, {"y", pt.y}}); // 直接使用整数
        }
        j["laser_points"] = pointsArray;

        std::string jsonString = j.dump();

        int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) return;

        struct sockaddr_in serverAddr;
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(targetPort);
        inet_pton(AF_INET, targetIp, &serverAddr.sin_addr);

        sendto(sockfd, jsonString.c_str(), jsonString.size(), 0,
               (struct sockaddr*)&serverAddr, sizeof(serverAddr));

        close(sockfd);
        std::cout << "[Published] TS=" << laserData.timestamp
                  << ", Points=" << laserData.points.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "JSON publish error: " << e.what() << std::endl;
    }
}

void __stdcall imageCallback(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser) {
    if (!pData || !pFrameInfo) return;

    // 增加帧计数器
    int currentFrameCount = ++g_frameCounter;

    // 根据帧间隔判断是否处理当前帧
    // FRAME_INTERVAL = M+1 (用户输入的M是想隔开的帧数)
    // 则 (currentFrameCount - 1) % FRAME_INTERVAL == 0 时处理
    // 例如 M=1 (隔1帧), FRAME_INTERVAL=2: (0%2==0), (1%2!=0), (2%2==0), (3%2!=0), ...
    if ((currentFrameCount - 1) % FRAME_INTERVAL != 0) {
        // std::cout << "Skipped frame " << (currentFrameCount - 1) << std::endl; // 可选：打印跳过的帧
        return; // 跳过不处理当前帧
    }

    // 打印一次像素格式（可选，用于验证）
    static bool fmtPrinted = false;
    if (!fmtPrinted) {
        std::cout << "Camera pixel format: 0x" << std::hex << pFrameInfo->enPixelType << std::dec
                  << " (BayerRG8 expected)" << std::endl;
        fmtPrinted = true;
    }

    uint64_t devTs = (static_cast<uint64_t>(pFrameInfo->nDevTimeStampHigh) << 32) |
                     static_cast<uint64_t>(pFrameInfo->nDevTimeStampLow);

    // 创建原始 Bayer 图像（单通道）
    cv::Mat raw(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);

    // 转换为 BGR（注意：BayerRG8 对应 COLOR_BAYER_RG2BGR）
    cv::Mat bgr;
    try {
        cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV cvtColor error: " << e.what() << std::endl;
        return;
    }

    if (bgr.empty()) {
        std::cerr << "Bayer to BGR conversion failed." << std::endl;
        return;
    }

    // --- 动态HSV范围切换逻辑 ---
    std::vector<cv::Point> laserPoints;
    for (const auto& range : HSV_RANGES) {
        // 清空上一轮的点
        laserPoints.clear();
        extractLaserCoordinates(bgr, range, laserPoints, ROW_INTERVAL); // 使用设定的行间隔
        std::cout << "Attempted HSV range, got " << laserPoints.size() << " points." << std::endl;

        if (laserPoints.size() >= MIN_POINTS_REQUIRED) {
            std::cout << "Got enough points (" << laserPoints.size() << ") with current HSV range." << std::endl;
            break; // 足够多点就停止尝试其他阈值
        }
    }

    // 如果遍历完所有范围都未能满足最小点数要求，可以选择处理不足的点或跳过
    if (laserPoints.size() < MIN_POINTS_REQUIRED) {
        std::cout << "Warning: Only " << laserPoints.size() << " points extracted after trying all HSV ranges. Publishing anyway." << std::endl;
        // 可以选择 return; 来跳过发布，如果点太少无法接受的话
    }

    // --- END ---

    // --- 新增：对提取到的激光点进行去畸变 ---
    undistortLaserPoints(laserPoints);
    // --- END ---

    // 发布结果 (注意：传递的是Point类型的laserPoints)
    g_laserBuffer.addLaserPoints(devTs, laserPoints);
    publishLaserDataAsJson({devTs, laserPoints}, "192.168.5.110", 8888); // 替换为目标IP
}

void waitForExit() {
    std::cout << "\nPress Enter to exit...\n";
    std::cin.ignore();
    g_exitRequested = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void printDeviceInfo(const MV_CC_DEVICE_INFO* dev) {
    if (!dev) return;
    if (dev->nTLayerType == MV_GIGE_DEVICE) {
        auto& gige = dev->SpecialInfo.stGigEInfo;
        int ip1 = (gige.nCurrentIp >> 24) & 0xFF;
        int ip2 = (gige.nCurrentIp >> 16) & 0xFF;
        int ip3 = (gige.nCurrentIp >> 8) & 0xFF;
        int ip4 = gige.nCurrentIp & 0xFF;
        std::cout << "GigE Camera: " << gige.chModelName
                  << " @ " << ip1 << "." << ip2 << "." << ip3 << "." << ip4 << std::endl;
    } else if (dev->nTLayerType == MV_USB_DEVICE) {
        std::cout << "USB Camera: " << dev->SpecialInfo.stUsb3VInfo.chModelName << std::endl;
    }
}

int main() {
    void* handle = nullptr;
    int nRet = MV_OK;

    do {
        MV_CC_DEVICE_INFO_LIST deviceList;
        memset(&deviceList, 0, sizeof(deviceList));

        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList);
        if (nRet != MV_OK) {
            std::cerr << "Failed to enumerate devices. Error: 0x" << std::hex << nRet << std::dec << std::endl;
            break;
        }

        if (deviceList.nDeviceNum == 0) {
            std::cout << "No camera found.\n";
            return 0;
        }

        std::cout << "Found " << deviceList.nDeviceNum << " camera(s):\n";
        for (unsigned int i = 0; i < deviceList.nDeviceNum; ++i) {
            std::cout << "[" << i << "] ";
            printDeviceInfo(deviceList.pDeviceInfo[i]);
        }

        unsigned int index = 0; // 需要根据序列号修改
        if (index >= deviceList.nDeviceNum) {
            std::cerr << "Invalid camera index.\n";
            break;
        }

        nRet = MV_CC_CreateHandle(&handle, deviceList.pDeviceInfo[index]);
        if (nRet != MV_OK) { std::cerr << "CreateHandle failed.\n"; break; }

        nRet = MV_CC_OpenDevice(handle);
        if (nRet != MV_OK) { std::cerr << "OpenDevice failed.\n"; break; }

        if (deviceList.pDeviceInfo[index]->nTLayerType == MV_GIGE_DEVICE) {
            int packetSize = MV_CC_GetOptimalPacketSize(handle);
            if (packetSize > 0) {
                MV_CC_SetIntValue(handle, "GevSCPSPacketSize", packetSize);
            }
        }

        MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF);

        nRet = MV_CC_RegisterImageCallBackEx(handle, imageCallback, handle);
        if (nRet != MV_OK) { std::cerr << "Register callback failed.\n"; break; }

        nRet = MV_CC_StartGrabbing(handle);
        if (nRet != MV_OK) { std::cerr << "StartGrabbing failed.\n"; break; }

        std::cout << "Streaming started. Capturing frames...\n";
        std::cout << "Processing every " << FRAME_INTERVAL << " frame(s) (skip " << (FRAME_INTERVAL - 1) << "). "
                  << "Extracting every " << ROW_INTERVAL << " row(s) (skip " << (ROW_INTERVAL - 1) << ").\n";

        waitForExit();

        MV_CC_StopGrabbing(handle);
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        handle = nullptr;

    } while (0);

    if (handle) {
        MV_CC_DestroyHandle(handle);
    }

    std::cout << "Exit.\n";
    return 0;
}