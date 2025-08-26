cmake_minimum_required(VERSION 3.16)
project(Laser_Camera_3D-online VERSION 1.0)

# 启用 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置默认构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------
# 静态链接 C/C++ 运行时（关键！避免依赖 VC_redist）
# -----------------------------
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")  # Release: /MT

# -----------------------------
# 包含自定义路径
# -----------------------------
include_directories(D:/Software)

# -----------------------------
# 查找并链接 OpenCV
# -----------------------------
set(OpenCV_DIR "D:/Software/OpenCV_4.10/opencv/build")
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR correctly.")
endif()

# -----------------------------
# 查找并链接 Eigen
# -----------------------------
set(EIGEN_INCLUDE_DIR "D:/Software/eigen-3.4.0")
if(NOT EXISTS ${EIGEN_INCLUDE_DIR}/Eigen/Core)
    message(FATAL_ERROR "Eigen not found at: ${EIGEN_INCLUDE_DIR}")
endif()
include_directories(${EIGEN_INCLUDE_DIR})

# -----------------------------
# 手动配置 MVSDK (MVCameraControl)
# -----------------------------
set(MVSDK_ROOT "D:/Software/MVS/Development" CACHE PATH "Path to MVS SDK")
set(MVSDK_LIB_DIR ${MVSDK_ROOT}/Libraries/win64)

# 查找 MvCameraControl.lib（链接时使用）
find_library(MV_CAMERA_LIB
    NAMES MvCameraControl
    PATHS ${MVSDK_LIB_DIR}
    NO_DEFAULT_PATH
)
if(NOT MV_CAMERA_LIB)
    message(FATAL_ERROR "MvCameraControl.lib not found in ${MVSDK_LIB_DIR}")
endif()

# -----------------------------
# 手动配置 ZeroMQ
# -----------------------------
set(ZMQ_INCLUDE_DIR "D:/Software/zeromq-4.3.5/include")
set(ZMQ_LIBRARY_DIR "D:/Software/zeromq-4.3.5/builds/lib/Release")

if(NOT EXISTS ${ZMQ_INCLUDE_DIR}/zmq.h)
    message(FATAL_ERROR "ZeroMQ headers not found at: ${ZMQ_INCLUDE_DIR}")
endif()
if(NOT EXISTS ${ZMQ_LIBRARY_DIR})
    message(FATAL_ERROR "ZeroMQ library dir not found at: ${ZMQ_LIBRARY_DIR}")
endif()

# 链接 ZeroMQ 库文件
set(ZMQ_LIBRARIES 
    "${ZMQ_LIBRARY_DIR}/libzmq-v143-mt-4_3_5.lib"
)

# -----------------------------
# 公共源文件和包含路径
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
# 可执行文件1: INS版本
# -----------------------------
add_executable(ins_Laser_Camera_3D-online
    ins_main.cpp
    src/ins_receiver.cpp
    ${COMMON_SOURCES}
)

target_include_directories(ins_Laser_Camera_3D-online PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${COMMON_INCLUDES}
)

target_link_libraries(ins_Laser_Camera_3D-online
     "D:/Software/OpenCV_4.10/opencv/build/x64/vc16/lib/opencv_world4100.lib"
    ${MV_CAMERA_LIB}
)

# -----------------------------
# 可执行文件2: ZMQ版本
# -----------------------------
add_executable(zmq_Laser_Camera_3D-online
    zmq_main.cpp
    src/zmq_receiver.cpp
    src/point_publisher.cpp      # ✅ 确保这个文件被包含
    ${COMMON_SOURCES}
)

target_include_directories(zmq_Laser_Camera_3D-online PRIVATE
    ${COMMON_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIR}
)

target_link_libraries(zmq_Laser_Camera_3D-online
     "D:/Software/OpenCV_4.10/opencv/build/x64/vc16/lib/opencv_world4100.lib"
    ${MV_CAMERA_LIB}
    ${ZMQ_LIBRARIES}
)

# -----------------------------
# 自动拷贝 DLL
# -----------------------------
function(copy_required_dlls target_name)
    set(DLLS_TO_COPY)

    # 1. OpenCV DLL
    set(OPENCV_DLL "${OpenCV_DIR}/x64/vc16/bin/opencv_world4100.dll")
    if(EXISTS "${OPENCV_DLL}")
        list(APPEND DLLS_TO_COPY "${OPENCV_DLL}")
    else()
        message(WARNING "OpenCV DLL not found: ${OPENCV_DLL}")
    endif()

    # 2. MVSDK DLL
    set(MV_DLL "C:/Program Files (x86)/Common Files/MVS/Runtime/Win64_x64/MvCameraControl.dll")
    if(EXISTS "${MV_DLL}")
        list(APPEND DLLS_TO_COPY "${MV_DLL}")
    else()
        message(WARNING "MvCameraControl.dll not found: ${MV_DLL}")
    endif()

    # 3. ZeroMQ DLL
    set(ZMQ_DLL "D:/Software/zeromq-4.3.5/builds/bin/Release/libzmq-v143-mt-4_3_5.dll")
    if(EXISTS "${ZMQ_DLL}")
        list(APPEND DLLS_TO_COPY "${ZMQ_DLL}")
    else()
        message(WARNING "ZeroMQ DLL not found: ${ZMQ_DLL}")
    endif()

    # 执行拷贝命令
    foreach(dll_path ${DLLS_TO_COPY})
        # ✅ 提前获取文件名（避免 $<basename> 问题）
        get_filename_component(dll_name ${dll_path} NAME)

        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll_path}"
                $<TARGET_FILE_DIR:${target_name}>
            COMMENT "Copying required DLL: ${dll_name}"
            VERBATIM
        )
    endforeach()
endfunction()

copy_required_dlls(ins_Laser_Camera_3D-online)
copy_required_dlls(zmq_Laser_Camera_3D-online)

# -----------------------------
# 生成安装包（只 include 一次）
# -----------------------------
include(CPack)
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_v${PROJECT_VERSION}_Win64")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)