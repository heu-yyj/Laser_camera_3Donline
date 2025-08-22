cmake_minimum_required(VERSION 3.16)
project(Laser_Camera_3D-online VERSION 1.0)

# 启用 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

include_directories(D:/Software)
# -----------------------------
# 查找并链接 OpenCV
# -----------------------------
# ========== 设置 OpenCV 路径（无需 .vscode/settings.json）==========
set(OpenCV_DIR "D:/Software/OpenCV_4.10/opencv/build")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR.")
endif()

# -----------------------------
# 查找并链接 Eigen
set(EIGEN_INCLUDE_DIR "D:/Software/eigen-3.4.0")
if(NOT EXISTS ${EIGEN_INCLUDE_DIR}/Eigen/Core)
    message(FATAL_ERROR "Eigen not found at: ${EIGEN_INCLUDE_DIR}")
endif()

include_directories(${EIGEN_INCLUDE_DIR})

# -----------------------------
# 手动配置 MVSDK (MVCameraControl)
# -----------------------------
# 设置相机 SDK 路径
set(MVSDK_ROOT "D:/Software/MVS/Development" CACHE PATH "Path to MVS SDK")

# 选择 64 位库路径（根据系统自动判断或手动指定）
set(MVSDK_LIB_DIR ${MVSDK_ROOT}/Libraries/win64)


# 查找 MvCameraControl.lib
find_library(MV_CAMERA_LIB
    NAMES MvCameraControl
    PATHS ${MVSDK_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT MV_CAMERA_LIB)
    message(FATAL_ERROR "MvCameraControl.lib not found in ${MVSDK_LIB_DIR}")
endif()

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_search_module(ZMQ QUIET libzmq)
endif()

if(NOT ZMQ_FOUND)
    # 如果 pkg-config 找不到，尝试手动查找
    set(ZMQ_INCLUDE_DIR "D:/Software/zeromq-4.3.5/include")   # 修改为你的 zmq 头文件路径
    set(ZMQ_LIBRARIES "D:/Software/zeromq-4.3.5/builds/lib/Release") # 修改为你的 zmq 库路径

    if(NOT EXISTS ${ZMQ_INCLUDE_DIR}/zmq.h)
        message(FATAL_ERROR "ZeroMQ headers not found at: ${ZMQ_INCLUDE_DIR}")
    endif()

    if(NOT EXISTS ${ZMQ_LIBRARIES})
        message(FATAL_ERROR "ZeroMQ library not found at: ${ZMQ_LIBRARIES}")
    endif()
else()
    set(ZMQ_INCLUDE_DIR ${ZMQ_INCLUDE_DIRS})
    set(ZMQ_LIBRARIES ${ZMQ_LIBRARIES})
endif()

# -----------------------------
# 公共源文件（所有 target 共用）
# -----------------------------
set(COMMON_SOURCES
    src/camera_capture.cpp
    src/laser_pointcloud.cpp
)

set(COMMON_INCLUDES
    ${PROJECT_SOURCE_DIR}/include
    ${MVSDK_ROOT}/Includes
)

# -----------------------------
# 可执行文件1:INS版本
# -----------------------------
add_executable(ins_Laser_Camera_3D-online
    ins_main.cpp
    src/ins_receiver.cpp
    ${COMMON_SOURCES}
)

target_include_directories(ins_Laser_Camera_3D-online PRIVATE
    ${COMMON_INCLUDES}
)

# -----------------------------
# 链接所有依赖库
# -----------------------------
target_link_libraries(ins_Laser_Camera_3D-online
    ${OpenCV_LIBS}
    ${MV_CAMERA_LIB}
)

# -----------------------------
# 可执行文件2:ZMQ版本
# -----------------------------

# 如果你把 point_publisher 封装成一个库（推荐）
add_library(point_publisher
    src/point_publisher.cpp
    include/point_publisher.h
)

# 给这个库添加 zmq 的包含路径，并导出给使用者
target_include_directories(point_publisher
    PUBLIC ${ZMQ_INCLUDE_DIRS}              # zmq.hpp 被暴露
    PRIVATE ${PROJECT_SOURCE_DIR}/include      # 私有头文件路径
)

# 链接 zmq 库
target_link_libraries(point_publisher PRIVATE zmq)

add_executable(zmq_Laser_Camera_3D-online
    zmq_main.cpp
    src/zmq_receiver.cpp
    src/point_publisher.cpp
    ${COMMON_SOURCES}
)

target_include_directories(zmq_Laser_Camera_3D-online PRIVATE
    ${COMMON_INCLUDES}
    ${ZMQ_INCLUDE_DIR}
)

# -----------------------------
# 链接所有依赖库
# -----------------------------
target_link_libraries(zmq_Laser_Camera_3D-online
    ${OpenCV_LIBS}
    ${MV_CAMERA_LIB}
    D:/Software/zeromq-4.3.5/builds/lib/Release/libzmq-v143-mt-4_3_5.lib
)
# -----------------------------
# 自动拷贝 ZeroMQ DLL 到输出目录（Windows）
# 
#-----------------------------
if(WIN32)
    function(copy_zmq_dlls target_name)
        # 根据构建类型选择正确的 DLL
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(ZMQ_DLL_NAME libzmq-v143-mtd-4_3_5.dll)
            set(ZMQ_DLL_DIR "D:/Software/zeromq-4.3.5/builds/bin/Debug")  # 假设 Debug DLL 在 bin/Debug
        else()
            set(ZMQ_DLL_NAME libzmq-v143-mt-4_3_5.dll)
            set(ZMQ_DLL_DIR "D:/Software/zeromq-4.3.5/builds/bin/Release") # Release DLL 在 bin/Release
        endif()

        # 构建完整路径
        set(ZMQ_DLL_PATH "${ZMQ_DLL_DIR}/${ZMQ_DLL_NAME}")

        # 检查 DLL 是否存在
        if(NOT EXISTS "${ZMQ_DLL_PATH}")
            message(WARNING "ZeroMQ DLL not found: ${ZMQ_DLL_PATH}. Please check ZeroMQ bin path.")
        else()
            # 添加构建后命令：拷贝 DLL 到可执行文件目录
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ZMQ_DLL_PATH}"
                    $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying ZeroMQ DLL: ${ZMQ_DLL_NAME} to output directory..."
            )
        endif()
    endfunction()

    # 为每个需要 ZeroMQ 的可执行文件调用
    copy_zmq_dlls(zmq_Laser_Camera_3D-online)
    # copy_zmq_dlls(ins_Laser_Camera_3D-online)  # 如果 INS 版本也用了 ZMQ
endif()