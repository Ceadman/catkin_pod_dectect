
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/Altitude.h>
#include <sensor_msgs/NavSatFix.h> 
#include <std_msgs/Float32.h>
#include <mavros_msgs/WaypointPush.h>
#include <mavros_msgs/Waypoint.h>
#include <mavros_msgs/CommandCode.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/core.hpp>
#include <tf/transform_datatypes.h>

#include <iostream>

#define ALTITUDE_TAKEOFF 30
#define FRAME_VY 0.5
#define FRAME_VZ 0
#define CRUISE_SPEED 15
#define WP_RADIUS 25
#define LINE_SPACING 100
#define HORIZONTAL_FOV 1.047

float K[] = {277.191356, 0.0, 2560, 0.0, 277.191356, 1088, 0.0, 0.0, 1.0};
mavros_msgs::State current_mode;
mavros_msgs::PositionTarget vel;
ros::Publisher local_pub;
bool land_flag = false;
bool task = false;
int approach_step = 0;
bool flag_init_position = false;
ros::Time last_detect_frame;
geometry_msgs::PoseStamped pose_mav_info;
double pose_mav_info_roll  = 0.0;
double pose_mav_info_pitch = 0.0;
double pose_mav_info_yaw   = 0.0;
geometry_msgs::PoseStamped pose_start_to_through;
geometry_msgs::PoseStamped init_position_take_off;
sensor_msgs::NavSatFix current_global_pos;
float target_area;
int direction_y = 0;
int direction_z = 0;
const double hover_R = 80.0;        // 盘旋半径 30 m
mavros_msgs::PositionTarget center_init;   // 初始圆心
mavros_msgs::PositionTarget center_takeup;
mavros_msgs::PositionTarget center_target;
bool circle_init = false;     // 已把圆心记下
geometry_msgs::TwistStamped command_vel;

void uav_control_vel(mavros_msgs::PositionTarget& vel_set)
{
    vel_set.coordinate_frame = mavros_msgs::PositionTarget::FRAME_BODY_NED;
    vel_set.type_mask = 
            mavros_msgs::PositionTarget::IGNORE_AFX |
            mavros_msgs::PositionTarget::IGNORE_AFY |
            mavros_msgs::PositionTarget::IGNORE_AFZ |
            mavros_msgs::PositionTarget::IGNORE_PX |
            mavros_msgs::PositionTarget::IGNORE_PY |
            mavros_msgs::PositionTarget::IGNORE_PZ |
            mavros_msgs::PositionTarget::IGNORE_YAW;
    local_pub.publish(vel_set);
}

void uav_control_vel_local(mavros_msgs::PositionTarget& vel_set)
{
    vel_set.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    vel_set.type_mask = 
            mavros_msgs::PositionTarget::IGNORE_AFX |
            mavros_msgs::PositionTarget::IGNORE_AFY |
            mavros_msgs::PositionTarget::IGNORE_AFZ |
            mavros_msgs::PositionTarget::IGNORE_PX |
            mavros_msgs::PositionTarget::IGNORE_PY |
            mavros_msgs::PositionTarget::IGNORE_PZ |
            mavros_msgs::PositionTarget::IGNORE_YAW;
    local_pub.publish(vel_set);
}

void uav_control_pos(mavros_msgs::PositionTarget& pos_set)
{
    pos_set.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    pos_set.type_mask = 
            mavros_msgs::PositionTarget::IGNORE_AFX |
            mavros_msgs::PositionTarget::IGNORE_AFY |
            mavros_msgs::PositionTarget::IGNORE_AFZ |
            mavros_msgs::PositionTarget::IGNORE_VX |
            mavros_msgs::PositionTarget::IGNORE_VY |
            mavros_msgs::PositionTarget::IGNORE_VZ |
            mavros_msgs::PositionTarget::IGNORE_YAW;
    local_pub.publish(pos_set);
}

void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    if (flag_init_position ==false && (msg->pose.position.z!=0)) //如果还未初始化 且 z坐标不为0>>>
    {
		init_position_take_off.pose.position.x = msg->pose.position.x;
	    init_position_take_off.pose.position.y = msg->pose.position.y;
	    init_position_take_off.pose.position.z = msg->pose.position.z;
        flag_init_position = true;  //>>>将“当地位置坐标”设为“初始坐标”并且将“初始化flag”设为true
    }
    pose_mav_info.pose.position.x = msg->pose.position.x;
    pose_mav_info.pose.position.y = msg->pose.position.y;
    pose_mav_info.pose.position.z = msg->pose.position.z;
    tf::Quaternion q(
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z,
        msg->pose.orientation.w);
    tf::Matrix3x3 m(q);
    m.getRPY(pose_mav_info_roll, pose_mav_info_pitch, pose_mav_info_yaw); // 单位 rad
}
    
void global_pose_cb(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    current_global_pos = *msg;
}

void state_cb(const mavros_msgs::State::ConstPtr& msg)
{
    current_mode = *msg;
}

void command_vel_cb(const geometry_msgs::TwistStamped::ConstPtr& msg)
{
    command_vel = *msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "plane_track_node");//初始化一个ROS节点
    ros::NodeHandle nh;
    ros::Rate rate(20);

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";
    mavros_msgs::SetMode takeoff_mode;
    takeoff_mode.request.custom_mode = "AUTO.TAKEOFF";
    ros::Time last_request = ros::Time::now();//更新请求时间
    ros::Time time_lost_track = ros::Time::now();
    ros::Time the_last_unattained_height = ros::Time::now();
    mavros_msgs::CommandBool arm_cmd;//mavros_msgs::CommandBool：ROS消息类型，用于解锁或锁定无人机
    arm_cmd.request.value = true;//将arm_cmd.request.value设置为true，表示解锁无人机

    bool wp_uploaded = false; 
    mavros_msgs::WaypointPush wp_push;
    mavros_msgs::Waypoint wp;
    const double cruise_alt  = ALTITUDE_TAKEOFF;   // 巡航高度
    const double cruise_speed = CRUISE_SPEED;   // 水平巡航速度
    const double wp_radius    = WP_RADIUS;    // 认为到达该半径即可换向
    const double line_spacing = LINE_SPACING;   // 每行间距
    const double freq         = 20.0;   // 控制频率
    int mode_num = 0; 
    double min_x, max_x, min_y, max_y;
    double current_y;
    bool cruise_done;
    enum CruiseState {GO_EAST, GO_WEST, TURN_NORTH} move_state;
    bool last_was_east = true; 

    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>
                    ("/tailsitter_0/mavros/local_position/pose", 10, pose_cb);
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
                    ("/tailsitter_0/mavros/state", 10, state_cb);
    ros::Subscriber global_pos_sub = nh.subscribe<sensor_msgs::NavSatFix>
                    ("/tailsitter_0/mavros/global_position/global", 10, global_pose_cb);
    ros::Subscriber target_point_sub = nh.subscribe<geometry_msgs::TwistStamped>
                    ("/navigation/command_velocity_local", 5, command_vel_cb);          
    local_pub = nh.advertise<mavros_msgs::PositionTarget>
                    ("/tailsitter_0/mavros/setpoint_raw/local",10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
                    ("/tailsitter_0/mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
                    ("/tailsitter_0/mavros/set_mode");

    std::cout << "请输入矩形对角坐标 (min_x max_x min_y max_y): ";
    std::string line;
    std::getline(std::cin, line);
    std::stringstream ss(line);
    if(!(ss >> min_x >> max_x >> min_y >> max_y)){
        ROS_ERROR("输入格式错误");
        return -1;
    }
    if(min_x > max_x) std::swap(min_x, max_x);
    if(min_y > max_y) std::swap(min_y, max_y);
    std::cout << "矩形范围: x[" << min_x << "," << max_x
    << "]  y[" << min_y << "," << max_y << "]"
    << std::endl;
    move_state = GO_EAST;
    current_y = min_y;
    cruise_done = false;

    while(ros::ok())
    {
        if (!circle_init)
        {
            // 以当前水平位置为圆心
            center_init.position.x = pose_mav_info.pose.position.x;
            center_init.position.y = pose_mav_info.pose.position.y;
            center_init.position.z = pose_mav_info.pose.position.z;  
            circle_init = true;
        }
        /* -------- 起飞阶段：先把高度拉到 10 m ---------- */

        while (current_global_pos.latitude == 0.0)
        {
            ROS_WARN("Waiting for global position...");
            ros::Duration(1.0).sleep();
            ros::spinOnce();
        }

        if( !current_mode.armed && (ros::Time::now() - last_request > ros::Duration(2.0))) //内层条件：检查是否解锁无人机并尝试解锁
        {
            if( arming_client.call(arm_cmd) && arm_cmd.response.success)
            {
                ROS_INFO("Vehicle armed");
                flag_init_position = false;
            }
            last_request = ros::Time::now();
        }
        else
        {
            if( current_mode.mode != "AUTO.TAKEOFF" && (ros::Time::now() - last_request > ros::Duration(2.0)))
            {
                if (set_mode_client.call(takeoff_mode) && takeoff_mode.response.mode_sent)
                {
                    ROS_INFO("AUTO.TAKEOFF enabled");
                    flag_init_position = false;
                    break;
                }
                else
                {
                    ROS_WARN("Failed to set AUTO.TAKEOFF, retry in 2 s ...");
                }
                last_request = ros::Time::now();
            }
        }
        rate.sleep();
        ros::spinOnce();
    }

    while(ros::ok())
    {
        if(pose_mav_info.pose.position.z > 9)
        {
            if (ros::Time::now() - the_last_unattained_height > ros::Duration(2.0))
            {
                the_last_unattained_height = ros::Time::now();
                ROS_INFO("Takes up");
                mode_num = 1;
                break; 
            }  
        }
        else
        {
            the_last_unattained_height = ros::Time::now();
        }
        rate.sleep();
        ros::spinOnce();
    }

    while(ros::ok())
    {
        if(current_mode.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(2.0)))
        {   //内层条件：检查是否为起飞模式并尝试切换
            if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
            {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        }
        center_takeup.header.frame_id = "takeup";
        center_takeup.position.x = center_init.position.x;
        center_takeup.position.y = center_init.position.y;
        center_takeup.position.z = center_init.position.z + ALTITUDE_TAKEOFF;
        uav_control_pos(center_takeup); 
        if(abs(pose_mav_info.pose.position.z - center_takeup.position.z) < 0.5)
        {
            if (ros::Time::now() - the_last_unattained_height > ros::Duration(2.0))
            {
                ROS_INFO("Plane Stabled");
                mode_num = 1;
                break; 
            } 
        }
        else
        {
            the_last_unattained_height = ros::Time::now();
        }
        rate.sleep();
        ros::spinOnce();
    }

    while(ros::ok)
    {
        // printf("mode_num = %d\r\n",mode_num);
        switch(mode_num)
        {
            case 1:
            {
                if(current_mode.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(2.0)))
                {   //内层条件：检查是否为起飞模式并尝试切换
                    if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
                    {
                        ROS_INFO("Offboard enabled");
                    }
                    last_request = ros::Time::now();
                }
                cruise_done = false;
                center_target.position.x = center_init.position.x + min_x;
                center_target.position.y = center_init.position.y + min_y;
                center_target.position.z = cruise_alt;
                double dx = pose_mav_info.pose.position.x - center_target.position.x;
                double dy = pose_mav_info.pose.position.y - center_target.position.y;
                uav_control_pos(center_target); 
                if (sqrt(dx*dx + dy*dy) < hover_R + 0.5 && abs(center_target.position.z - cruise_alt) < 2) // 30 m 圆域
                {   
                    if(ros::Time::now() - last_request > ros::Duration(2.0))
                    {
                        ROS_INFO("Cruise Ready");
                        vel.header.frame_id = "map";
                        mode_num = 2;
                        break;
                    }
                }
                break;
            }
            case 2:
            {
                if(cruise_done == true)
                {
                    cruise_done == false;
                    mode_num = 1;
                }
                else
                {
                    ROS_INFO("Cruise Control: %d", move_state);
                    switch(move_state)
                    {
                        case GO_EAST:
                        {
                            vel.velocity.x = cruise_speed;
                            vel.velocity.y = 0;
                            if(pose_mav_info.pose.position.x >= max_x - wp_radius)
                            {
                                last_was_east = true;
                                move_state = TURN_NORTH;
                                current_y += line_spacing;
                                if(current_y >= max_y) 
                                {
                                    cruise_done = true;
                                }
                            }
                            break;
                        }
                        case GO_WEST:
                        {
                            vel.velocity.x = -cruise_speed;
                            vel.velocity.y = 0;
                            // if(pose_mav_info.pose.position.x <= min_x + wp_radius)
                            ROS_INFO("%.2f", pose_mav_info.pose.position.x);
                            if(pose_mav_info.pose.position.x <= (min_x + max_x) / 2)
                            {
                                
                                vel.velocity.z = 0.5;
                                ROS_INFO("VZ = 0.5");
                                last_was_east = false;
                                // move_state = TURN_NORTH;
                                // current_y += line_spacing;
                                // if(current_y >= max_y) 
                                // {
                                //     cruise_done = true;
                                // }
                            }
                            break;
                        }
                        case TURN_NORTH:
                        {
                            /* 向北飞一行间距 */
                            vel.velocity.x = 0;
                            vel.velocity.y = cruise_speed;
                            if(pose_mav_info.pose.position.y >= current_y - wp_radius)
                            {
                                move_state = (last_was_east ? GO_WEST : GO_EAST);
                            }
                            break;
                        }
                    }
                    uav_control_vel_local(vel);
                }
                break;
            }
        }
        ros::spinOnce();
        rate.sleep();
    }
}