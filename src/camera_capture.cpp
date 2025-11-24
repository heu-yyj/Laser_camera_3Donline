#include "camera_capture.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
  
CameraCapture::CameraCapture(const std::string& ip, const std::string& netExport)
    : m_ip(ip), m_netExport(netExport), handle(nullptr), m_mutex(nullptr),
    m_isRunning(false), m_savePath("C:/Images/") {
}

CameraCapture::~CameraCapture() {
    stop();
    if (m_mutex) {
        delete m_mutex;
        m_mutex = nullptr;
    }
}

bool CameraCapture::init() {
    m_mutex = new CMVMutex();
    if (!m_mutex) {
        std::cerr << "Failed to create mutex." << std::endl;
        return false;
    }

    MV_CC_DEVICE_INFO stDevInfo = { 0 };
    MV_GIGE_DEVICE_INFO stGigEDev = { 0 };

    unsigned int nIp1, nIp2, nIp3, nIp4, nIp;
    sscanf_s(m_ip.c_str(), "%d.%d.%d.%d", &nIp1, &nIp2, &nIp3, &nIp4);
    nIp = (nIp1 << 24) | (nIp2 << 16) | (nIp3 << 8) | nIp4;
    stGigEDev.nCurrentIp = nIp;

    sscanf_s(m_netExport.c_str(), "%d.%d.%d.%d", &nIp1, &nIp2, &nIp3, &nIp4);
    nIp = (nIp1 << 24) | (nIp2 << 16) | (nIp3 << 8) | nIp4;
    stGigEDev.nNetExport = nIp;

    stDevInfo.nTLayerType = MV_GIGE_DEVICE;
    stDevInfo.SpecialInfo.stGigEInfo = stGigEDev;

    //初始化SDK
    int nRet = MV_CC_Initialize();
    if (MV_OK != nRet) {
        std::cerr << "Initialize SDK failed! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    //选择设备创建句柄
    nRet = MV_CC_CreateHandle(&handle, &stDevInfo);
    if (MV_OK != nRet) {
        std::cerr << "Create Handle failed! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    //打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        std::cerr << "Open Device failed! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }

    MVCC_INTVALUE_EX stIntEx = { 0 };
    nRet = MV_CC_GetIntValueEx(handle, "TriggerMode", &stIntEx);
    if (MV_OK == nRet) {
        if (stIntEx.nCurValue != MV_TRIGGER_MODE_OFF) {
            MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF);
        }
    }

    nRet = MV_CC_RegisterImageCallBackEx(handle, ImageCallback, this);
    if (MV_OK != nRet) {
        std::cerr << "Register image callback failed! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }

    return true;
}

bool CameraCapture::start() {
    if (!handle) return false;

    //开始取流
    int nRet = MV_CC_StartGrabbing(handle);
    if (nRet != MV_OK) {
        std::cerr << "Start grabbing failed! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }

    m_isRunning = true;
    m_captureThread = std::thread(&CameraCapture::processImages, this);

    return true;
}

bool CameraCapture::stop() {
    if (!handle) return false;

    m_isRunning = false;
    if (m_captureThread.joinable()) {
        m_captureThread.join();
    }
    //停止取流
    // 
    //关闭设备
    //销毁句柄
    MV_CC_StopGrabbing(handle);
    MV_CC_RegisterImageCallBackEx(handle, NULL, NULL);
    MV_CC_CloseDevice(handle);
    MV_CC_DestroyHandle(handle);
    handle = nullptr;

    return true;
}

void CameraCapture::setSavePath(const std::string& path) {
    m_savePath = path;
}

bool CameraCapture::getImage(cv::Mat& outImage) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    if (m_latestImage.data.empty()) return false;

    outImage = cv::Mat(m_latestImage.height, m_latestImage.width, CV_8UC1, m_latestImage.data.data());
    return true;
}

void CameraCapture::ImageCallback(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser) {
    CameraCapture* capture = static_cast<CameraCapture*>(pUser);
    if (!capture || !pData || !pFrameInfo) return;

    ImageData img;
    img.width = pFrameInfo->nExtendWidth;
    img.height = pFrameInfo->nExtendHeight;
    img.frameLen = pFrameInfo->nFrameLenEx;
    img.frameNum = pFrameInfo->nFrameNum;

    uint64_t devTimeStamp = pFrameInfo->nDevTimeStampHigh;
    devTimeStamp = (devTimeStamp << 32) + pFrameInfo->nDevTimeStampLow;
    img.timestamp = devTimeStamp;

    img.data.assign(pData, pData + img.frameLen);

    std::lock_guard<std::mutex> lock(capture->m_dataMutex);
    capture->m_latestImage = std::move(img);
    capture->m_dataCond.notify_one();

    capture->saveImage(img);
}

// camera_capture.cpp 实现
bool CameraCapture::getImage(cv::Mat& outImage, uint64_t& outTimestamp) {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    if (m_latestImage.data.empty()) return false;

    outImage = cv::Mat(m_latestImage.height, m_latestImage.width, CV_8UC1, m_latestImage.data.data());
    outTimestamp = m_latestImage.timestamp;
    return true;
}

uint64_t CameraCapture::getLatestTimestamp() {
    std::lock_guard<std::mutex> lock(m_dataMutex);
    return m_latestImage.data.empty() ? 0 : m_latestImage.timestamp;
}

void CameraCapture::processImages() {
    while (m_isRunning) {
        std::unique_lock<std::mutex> lock(m_dataMutex);
        m_dataCond.wait(lock, [this] {
            return !m_latestImage.data.empty() || !m_isRunning;
            });

        if (!m_isRunning) break;

        cv::Mat img(m_latestImage.height, m_latestImage.width, CV_8UC1, m_latestImage.data.data());
        cv::imshow("CameraCapture", img);
        cv::waitKey(1);
    }
}

void CameraCapture::saveImage(const ImageData& imgData) {
    std::lock_guard<std::mutex> lock(m_saveMutex); // 线程安全

    // 创建目录
    std::error_code ec;
    std::filesystem::path saveDir(m_savePath);
    std::filesystem::create_directories(saveDir, ec);
    if (ec) {
        std::cerr << "Failed to create directory: " << m_savePath
            << " Error: " << ec.message() << std::endl;
        return;
    }

    // 生成安全文件名
    std::string timestampStr = std::to_string(imgData.timestamp);
    std::filesystem::path filePath = saveDir / ("Image_" + timestampStr + ".png");

    // 复制数据避免 const_cast
    cv::Mat img(imgData.height, imgData.width, CV_8UC1);
    std::copy(imgData.data.begin(), imgData.data.end(), img.data);

    // 保存并检查错误
    if (!cv::imwrite(filePath.string(), img)) {
        std::cerr << "Failed to save image: " << filePath << std::endl;
    }
}