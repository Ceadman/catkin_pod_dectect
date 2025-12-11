
#include <ros/ros.h>
#include <geographic_msgs/GeoPoseStamped.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 10000  // 本地监听端口

/* XOR 校验和 ----------------------------------------------------------*/
static uint8_t xor_checksum(const uint8_t *buf, size_t len)
{
    uint8_t sum = 0;
    for (size_t i = 0; i < len; ++i) sum ^= buf[i];
    return sum;
}

/* 打印 HEX 行 ---------------------------------------------------------*/
// static void print_hex_line(const uint8_t *buf, size_t len)
// {
//     for (size_t i = 0; i < len; ++i)
//         ROS_INFO("%02X%s", buf[i], (i + 1 == len) ? "\n" : " ");
// }

float bytes_to_float_le(const uint8_t src[4])
{
    union {
        uint32_t u32;
        float    f32;
    } u;

    u.u32 = ((uint32_t)src[0] <<  0)
          | ((uint32_t)src[1] <<  8)
          | ((uint32_t)src[2] << 16)
          | ((uint32_t)src[3] << 24);

    return u.f32;
}

static inline int16_t bytes_to_int16_le(const uint8_t src[2])
{
    return (int16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "uav_control_node");//初始化一个ROS节点
    ROS_INFO("uav_control_init");
    ros::NodeHandle nh;
    ros::Rate rate(20);

    ros::Publisher local_pub = nh.advertise<geographic_msgs::GeoPoseStamped>
                    ("/mavros/setpoint_position/global",10);


    while(ros::ok)
    {
        // 部署“起飞”，切换“GUIDE”模式等
        break;
        ///////////////////////////
    }

    while(ros::ok)
    {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) { ROS_INFO("socket"); close(sock); continue;}

        struct sockaddr_in addr = {
            .sin_family      = AF_INET,
            .sin_port        = htons(PORT),
            .sin_addr   = { .s_addr = htonl(INADDR_ANY) } 
        };

        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            ROS_INFO("bind"); close(sock); continue;
        }
        uint8_t data[64];
        struct sockaddr_in peer;
        socklen_t peer_len = sizeof(peer);

        ssize_t n = recvfrom(sock, data, sizeof(data), 0,
                            (struct sockaddr *)&peer, &peer_len);
        if (n != 64) {
            ROS_INFO("数据长度无法解析: %zd\n", n);
            close(sock); continue;
        }

        /* 打印 HEX 行 */
        // print_hex_line(data, sizeof(data));

        uint8_t calc = xor_checksum(&data[2], 60);   // data[2:62]
        if (calc != data[62]) {
            ROS_INFO("no");
            close(sock); continue;
        }
        geographic_msgs::GeoPoseStamped geopose_pub;
        geopose_pub.header.stamp    = ros::Time::now();
        geopose_pub.header.frame_id = "map";
        float lon, lat;
        lon = bytes_to_float_le(&data[29]);
        lat = bytes_to_float_le(&data[33]);
        int16_t alt = bytes_to_int16_le(&data[37]);
        ROS_INFO("support_az_raw = %f, %f, %d\n", lon, lat, alt);
        geopose_pub.pose.position.latitude  = lat;
        geopose_pub.pose.position.longitude = lon;
        geopose_pub.pose.position.altitude  = alt;

        geopose_pub.pose.orientation.x = 0.0;
        geopose_pub.pose.orientation.y = 0.0;
        geopose_pub.pose.orientation.z = 0.0;
        geopose_pub.pose.orientation.w = 1.0;
        /* 4. 发布 */
        local_pub.publish(geopose_pub);

        ros::spinOnce();
        rate.sleep();
    }
}