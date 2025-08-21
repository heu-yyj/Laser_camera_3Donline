#pragma once   
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

// SDK 提供的互斥锁类（假设）
class CMVMutex {
public:
    CMVMutex() {}
    ~CMVMutex() {}
    void Lock() {}
    void Unlock() {}
};

// 内部图像数据结构
struct ImageData {
    int width;
    int height;
    int frameLen;
    int frameNum;
    uint64_t timestamp;  // 微秒级时间戳
    std::vector<unsigned char> data;
};

// 相机采集类
class CameraCapture {
public:
    CameraCapture(const std::string& ip, const std::string& netExport);
    ~CameraCapture();

    bool init();
    bool start();
    bool stop();

    // 获取最新图像（无时间戳）
    bool getImage(cv::Mat& outImage);

    // 获取最新图像和时间戳（推荐用于同步）
    bool getImage(cv::Mat& outImage, uint64_t& outTimestamp);

    // 获取最新图像的时间戳（用于匹配 INS）
    uint64_t getLatestTimestamp();

    void setSavePath(const std::string& path);

private:
    std::string m_ip;
    std::string m_netExport;
    void* handle;
    CMVMutex* m_mutex;
    bool m_isRunning;
    std::mutex m_saveMutex;

    // 图像数据
    ImageData m_latestImage;
    std::mutex m_dataMutex;
    std::condition_variable m_dataCond;
    std::thread m_captureThread;

    std::string m_savePath;

    // 静态回调函数
    static void ImageCallback(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);

    // 图像处理线程
    void processImages();

    // 保存图像
    void saveImage(const ImageData& imgData);
};
