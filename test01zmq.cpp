#include <zmq.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <random>
#include <nlohmann/json.hpp> // 引入 json 库

using json = nlohmann::json;

// 传感器ID
const std::string SENSOR_ID = "lidar_sensor_01";

// 融合节点地址（替换为你自己的 IP 和端口）
const std::string SERVER_ADDR = "tcp://10.101.30.221:5555";

// 发送间隔（毫秒）
const int SEND_INTERVAL_MS = 100;

int main() {
    try {
        zmq::context_t context(1);
        zmq::socket_t socket(context, ZMQ_PUSH);
        socket.connect(SERVER_ADDR);
        std::cout << "传感器 " << SENSOR_ID << " 连接到融合节点: " << SERVER_ADDR << std::endl;

        // 模拟发送多次数据（或改为无限循环）
        for (int i = 0; i < 1000; ++i) {
            // 构造你提供的数据结构
            json data;
            data["header"]["timestamp"] = 1763089100.63957 + i; // 模拟递增时间戳
            data["header"]["frame_id"] = "lidar";
            data["header"]["frame_idx"] = 21 + i;
            data["header"]["pose_idx"] = 119 + i;
            data["header"]["pose"] = {
                {-0.30901700258255005, 0.9510565400123596, 0.0, -11.83535099029541},
                {-0.9510565400123596, -0.30901700258255005, 0.0, -36.42546463012695},
                {0.0, 0.0, 1.0, 5.482308864593506},
                {0.0, 0.0, 0.0, 1.0}
            };

            // 模拟点云数据（可替换为真实数据）
            data["points"] = {
                -6.6287736892700195, -51.8392219543457, -4.937307357788086,
                -10.684879302978516, -40.10737991333008, 4.795661449432373,
                -10.737646102905273, -39.885284423828125, 4.688942909240723,
                -10.798229217529297, -39.603763580322266, 5.102196216583252,
                -10.760740280151367, -39.842533111572266, 5.5765557289123535
                // ...（此处省略更多点，你可以用 vector 动态填充）
            };

            // 转为字符串
            std::string json_str = data.dump(); // 默认压缩格式
            // std::string json_str = data.dump(4); // 美化格式（缩进4空格）

            // 发送
            zmq::message_t msg(json_str.size());
            memcpy(msg.data(), json_str.data(), json_str.size());
            socket.send(msg, zmq::send_flags::none);
            std::cout << "发送第 " << (i+1) << " 条数据\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(SEND_INTERVAL_MS));
        }

        std::cout << "发送完毕。\n";
    }
    catch (const zmq::error_t& e) {
        std::cerr << "ZeroMQ 错误: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}