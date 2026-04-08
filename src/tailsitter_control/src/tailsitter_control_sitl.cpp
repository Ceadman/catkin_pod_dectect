
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/GlobalPositionTarget.h>
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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <mavros_msgs/CommandLong.h>
#include <mavros_msgs/CommandInt.h>


#include <iostream>

#define ALTITUDE_TAKEOFF 50
#define FRAME_VY 0.5
#define FRAME_VZ 0
#define CRUISE_SPEED 20
#define WP_RADIUS 25
#define LINE_SPACING 50
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
const double hover_R = 60.0;        // 盘旋半径 30 m
// mavros_msgs::PositionTarget center_init;   // 初始圆心
// mavros_msgs::PositionTarget center_takeup;
// mavros_msgs::PositionTarget center_target;

mavros_msgs::GlobalPositionTarget global_target;
mavros_msgs::GlobalPositionTarget center_init;   // 初始圆心
mavros_msgs::GlobalPositionTarget center_takeup;
mavros_msgs::GlobalPositionTarget center_target;
float current_rel_position_alt = 0;
static const double DEG_PER_M = 1.0 / 111320.0;


bool circle_init = false;     // 已把圆心记下
geometry_msgs::TwistStamped command_vel;

ros::ServiceClient cmd_int_client;

// void uav_control_vel(mavros_msgs::PositionTarget& vel_set)
// {
//     vel_set.coordinate_frame = mavros_msgs::PositionTarget::FRAME_BODY_NED;
//     vel_set.type_mask = 
//             mavros_msgs::PositionTarget::IGNORE_AFX |
//             mavros_msgs::PositionTarget::IGNORE_AFY |
//             mavros_msgs::PositionTarget::IGNORE_AFZ |
//             mavros_msgs::PositionTarget::IGNORE_PX |
//             mavros_msgs::PositionTarget::IGNORE_PY |
//             mavros_msgs::PositionTarget::IGNORE_PZ |
//             mavros_msgs::PositionTarget::IGNORE_YAW;
//     local_pub.publish(vel_set);
// }

void uav_control_vel_local(mavros_msgs::PositionTarget& vel_set)
{

     // 静态变量用于存储上一次调用时间
    static ros::Time last_time = ros::Time::now();
    
    // 获取当前时间
    ros::Time current_time = ros::Time::now();
    
    // 计算时间间隔（秒）
    float dt = (current_time - last_time).toSec();
    
    // 更新时间戳，为下一次调用做准备
    last_time = current_time;
    
    // 确保dt不会太大（例如程序启动时的第一次调用）
    if (dt <= 0 || dt > 1.0) {
        dt = 0.1;
    }

  

    float airSpeed = sqrt(vel_set.velocity.x * vel_set.velocity.x +vel_set.velocity.y * vel_set.velocity.y);   
    double yaw_rad = atan2(vel_set.velocity.x, vel_set.velocity.y);  // x = 东向，y = 北向
    // 转换为度
    double yaw_deg = yaw_rad * 180 / M_PI;
    // 确保在0-360度范围内
    if (yaw_deg < 0) {
        yaw_deg += 360.0;
    }

    float current_altitude = pose_mav_info.pose.position.z;  // 当前高度（相对起飞点）
    // ros::Time current_time = ros::Time::now();
    // 计算目标高度（基于当前高度和垂直速度）
    float target_altitude = current_altitude + vel_set.velocity.z * dt;  // d
    
   
    mavros_msgs::CommandInt cmd;
    cmd.request.frame = 0;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd.request.command = 43000;   // MAV_CMD_DO_REPOSITION
    cmd.request.current = 0;     // 在GUIDED模式下使用
    cmd.request.autocontinue = 0;
    
    cmd.request.param1 = 0; 
    cmd.request.param2 = airSpeed;  
    cmd.request.param3 = 2.0;  
    cmd.request.param4 = 0;   
    cmd.request.x = 0;  
    cmd.request.y = 0;  
    cmd.request.z = 0;   
    
    
    if(cmd_int_client.call(cmd)){
      ///   ROS_INFO("CommandInt VEL sent, result: %d", cmd.response.success);
         ROS_INFO("CommandInt dt sent, result: %f", dt);

    }



     mavros_msgs::CommandInt cmd1;
    cmd1.request.frame = 0;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd1.request.command = 43002;   // MAV_CMD_DO_REPOSITION
    cmd1.request.current = 0;     // 在GUIDED模式下使用
    cmd1.request.autocontinue = 0;
    
    cmd1.request.param1 = 1; 
    cmd1.request.param2 = yaw_deg;  
    cmd1.request.param3 = 10.0;  
    cmd1.request.param4 = 0;   
    cmd1.request.x = 0;  
    cmd1.request.y = 0;  
    cmd1.request.z = 0;   
    
    
    if(cmd_int_client.call(cmd1)){
    ///  ROS_INFO("CommandInt Yaw sent, result: %f", yaw_deg);
    }




    mavros_msgs::CommandInt cmd2;
    cmd2.request.frame = 6;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd2.request.command = 43003;   // MAV_CMD_DO_REPOSITION
    cmd2.request.current = 0;     // 在GUIDED模式下使用
    cmd2.request.autocontinue = 0;
    
    cmd2.request.param1 = 0; 
    cmd2.request.param2 = 0;  
    cmd2.request.param3 = vel_set.velocity.z;  
    cmd2.request.param4 = 0;   
    cmd2.request.x = 0;  
    cmd2.request.y = 0;  
    cmd2.request.z = target_altitude;   
    
    
    if(cmd_int_client.call(cmd2)){
        // ROS_INFO("CommandInt dt sent, result: %f", dt);
       ///  ROS_INFO("CommandInt Yaw sent, result: %f", target_altitude);
    }

}

// void uav_control_pos(mavros_msgs::PositionTarget& pos_set)
// {
//     pos_set.header.stamp = ros::Time::now();
//     pos_set.coordinate_frame = mavros_msgs::PositionTarget::FRAME_BODY_NED;///mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
//     pos_set.type_mask = 
//             mavros_msgs::PositionTarget::IGNORE_AFX |
//             mavros_msgs::PositionTarget::IGNORE_AFY |
//             mavros_msgs::PositionTarget::IGNORE_AFZ |
//             mavros_msgs::PositionTarget::IGNORE_VX |
//             mavros_msgs::PositionTarget::IGNORE_VY |
//             mavros_msgs::PositionTarget::IGNORE_VZ |
//             mavros_msgs::PositionTarget::IGNORE_YAW;

//         // ROS_INFO("GUIDED enabledrrrr - Position: [X: %f, Y: %f, Z: %f]", 
//         // pos_set.position.x,
//         // pos_set.position.y,
//         // pos_set.position.z);
//     local_pub.publish(pos_set);
// }

void uav_control_pos(mavros_msgs::GlobalPositionTarget& pos_set)
{

    mavros_msgs::CommandInt cmd;
    cmd.request.frame = 6;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd.request.command = 192;   // MAV_CMD_DO_REPOSITION
    cmd.request.current = 0;     // 在GUIDED模式下使用
    cmd.request.autocontinue = 0;
    
    // 将经纬度转换为整数格式（度 * 1e7）
    cmd.request.param1 = -1;  // 速度
    cmd.request.param2 = 0;   // 位掩码
    cmd.request.param3 = 0;   // 保留
    cmd.request.param4 = 0;   // 偏航角
    
    // 注意：CommandInt使用整数格式的经纬度
    cmd.request.x = static_cast<int32_t>(pos_set.latitude * 1e7); ///353632622; ///static_cast<int32_t>(pos_set.latitude * 1e7);;///321265489;   ///static_cast<int32_t>(pos_set.latitude * 1e7);
    cmd.request.y = static_cast<int32_t>(pos_set.longitude * 1e7);///1491542376;////static_cast<int32_t>(pos_set.longitude * 1e7);///1182365894;  ///static_cast<int32_t>(pos_set.longitude * 1e7);
    cmd.request.z = ALTITUDE_TAKEOFF;                             ////50.0;///pos_set.altitude;                          
    
    // ROS_INFO("GUIDED enabledrrrr - Position: [X: %d, Y: %d, Z: %f]", 
    // cmd.request.x,
    // cmd.request.y,
    // cmd.request.z);
    
    if(cmd_int_client.call(cmd)){
         ROS_INFO("CommandInt sent, result: %d", cmd.response.success);
        ///return cmd.response.success;
    }
 
}

void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{

    current_rel_position_alt =  msg->pose.position.z;
    // if (flag_init_position ==false && (msg->pose.position.z!=0)) //如果还未初始化 且 z坐标不为0>>>
    // {
	// 	init_position_take_off.pose.position.x = msg->pose.position.x;
	//     init_position_take_off.pose.position.y = msg->pose.position.y;
	//     init_position_take_off.pose.position.z = msg->pose.position.z;
    //     flag_init_position = true;  //>>>将“当地位置坐标”设为“初始坐标”并且将“初始化flag”设为true
    // }
    pose_mav_info.pose.position.x = msg->pose.position.x;
    pose_mav_info.pose.position.y = msg->pose.position.y;
    pose_mav_info.pose.position.z = msg->pose.position.z;
    // ROS_INFO("GUIDED enabled3333 - Position: [X: %f, Y: %f, Z: %f]", 
    //      pose_mav_info.pose.position.x,
    //      pose_mav_info.pose.position.y,
    //      pose_mav_info.pose.position.z);
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

    // ROS_INFO("GUIDED enabled3333 - Position: [X: %f, Y: %f, Z: %f]", 
    // current_global_pos.latitude,
    // current_global_pos.longitude,
    // current_global_pos.altitude);
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
    offb_set_mode.request.custom_mode = "GUIDED";//  GUIDED  OFFBOARD
    mavros_msgs::SetMode takeoff_mode;
    takeoff_mode.request.custom_mode = "AUTO";
    ros::Time last_request = ros::Time::now();//更新请求时间
    ros::Time time_lost_track = ros::Time::now();
    ros::Time the_last_unattained_height = ros::Time::now();
    mavros_msgs::CommandBool arm_cmd;    //mavros_msgs::CommandBool：ROS消息类型，用于解锁或锁定无人机
    arm_cmd.request.value = true;        //将arm_cmd.request.value设置为true，表示解锁无人机

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



    // 在循环外定义计时变量
    ros::Time last_control_time = ros::Time::now();
    const double CONTROL_RATE = 20.0;  // 10 Hz
    const ros::Duration CONTROL_INTERVAL(1.0 / CONTROL_RATE);///1.0 / CONTROL_RATE


    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>
                    ("/mavros/local_position/pose", 10, pose_cb);
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
                    ("/mavros/state", 10, state_cb);
    ros::Subscriber global_pos_sub = nh.subscribe<sensor_msgs::NavSatFix>
                    ("/mavros/global_position/global", 10, global_pose_cb);
    ros::Subscriber target_point_sub = nh.subscribe<geometry_msgs::TwistStamped>
                    ("/navigation/command_velocity_local", 5, command_vel_cb);          
    // local_pub = nh.advertise<mavros_msgs::PositionTarget>
    //                 ("/mavros/setpoint_raw/local",10);  
    // local_pub = nh.advertise<mavros_msgs::GlobalPositionTarget>
    //                 ("/mavros/setpoint_raw/global",10);  ////
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
                    ("/mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
                    ("/mavros/set_mode");
    cmd_int_client = nh.serviceClient<mavros_msgs::CommandInt>("/mavros/cmd/command_int");

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
        if (!circle_init && current_mode.armed )
        {
            // 以当前水平位置为圆心
            // center_init.position.x = pose_mav_info.pose.position.x;
            // center_init.position.y = pose_mav_info.pose.position.y;
            // center_init.position.z = cruise_alt;///pose_mav_info.pose.position.z;  

         center_init.latitude  = current_global_pos.latitude;
         center_init.longitude = current_global_pos.longitude;
         center_init.altitude  = current_rel_position_alt+ALTITUDE_TAKEOFF;  
         circle_init = true;
         ROS_INFO("GUIDED 444433 - Position: [X: %f, Y: %f, Z: %f]", 
         center_init.latitude,
         center_init.longitude,
         center_init.altitude);
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
           // if( arming_client.call(arm_cmd) && arm_cmd.response.success)
          //  {
                ROS_INFO("Vehicle armed");
                flag_init_position = false;
           // }
            last_request = ros::Time::now();
        }
        else
        {
            if( current_mode.mode == "AUTO")  //// && (ros::Time::now() - last_request > ros::Duration(2.0))
            {
                // if (set_mode_client.call(takeoff_mode) && takeoff_mode.response.mode_sent)
                // {
                     ROS_INFO("AUTO enabled");
                     flag_init_position = false;
                     break;
                // }
                // else
                // {
                //     ROS_WARN("Failed to set AUTO, retry in 2 s ...");
                // }

                ROS_INFO("AUTO altitude:%f",current_rel_position_alt);
                last_request = ros::Time::now();
            }
        }
        rate.sleep();
        ros::spinOnce();
    }

    while(ros::ok())
    {
         ROS_INFO("AUTO altitutde:%f",current_rel_position_alt);
        if(current_rel_position_alt > 20)// if(pose_mav_info.pose.position.z > 9)
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
        if(current_mode.mode != "GUIDED" && (ros::Time::now() - last_request > ros::Duration(2.0)))
        {   //内层条件：检查是否为起飞模式并尝试切换
            if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
            {
                ROS_INFO("GUIDED eeeenabled");

                center_takeup.header.frame_id = "";
                center_takeup.latitude = center_init.latitude;
                center_takeup.longitude = center_init.longitude;
                center_takeup.altitude = center_init.altitude;     ///+ ALTITUDE_TAKEOFF;
                uav_control_pos(center_takeup); 
            }
            last_request = ros::Time::now();
        }
       
        // center_takeup.position.x = center_init.position.x;
        // center_takeup.position.y = center_init.position.y;
        // center_takeup.position.z = 20;//center_init.position.z; ///+ ALTITUDE_TAKEOFF;
     

        // ROS_INFO("GUIDED enabled222111 - Position: [X: %f, Y: %f, Z: %f]", 
        //  pose_mav_info.pose.position.z ,
        //  center_takeup.position.z,
        //  abs(pose_mav_info.pose.position.z - center_takeup.position.z));
        //    // 以10Hz的频率执行控制
 
        // if(ros::Time::now() - last_control_time >= CONTROL_INTERVAL )
        // {
        // uav_control_pos(center_takeup);
        // last_control_time = ros::Time::now();
        // }
       ////  uav_control_pos(center_takeup); 
        if(abs(current_rel_position_alt - center_takeup.altitude)< 0.5)   ///if(abs(pose_mav_info.pose.position.z - center_takeup.position.z) < 0.5)
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
                if(current_mode.mode != "GUIDED" && (ros::Time::now() - last_request > ros::Duration(2.0)))
                {   //内层条件：检查是否为起飞模式并尝试切换
                    if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
                    {
                        ROS_INFO("GUIDED enabled");
                    }
                    last_request = ros::Time::now();
                }
                cruise_done = false;
                // center_target.position.x = center_init.position.x + min_x;
                // center_target.position.y = center_init.position.y + min_y;
                // center_target.position.z = cruise_alt;
                // double dx = pose_mav_info.pose.position.x - center_target.position.x;
                // double dy = pose_mav_info.pose.position.y - center_target.position.y;
                // uav_control_pos(center_target); 

                global_target.latitude  = center_init.latitude  + min_y * DEG_PER_M;
                global_target.longitude = center_init.longitude + min_x * DEG_PER_M;
                global_target.altitude  = ALTITUDE_TAKEOFF;
                double dLat = (current_global_pos.latitude  - global_target.latitude)  * 111320.0;
                double dLon = (current_global_pos.longitude - global_target.longitude) * 111320.0 * cos(current_global_pos.latitude*M_PI/180);
               
                if(ros::Time::now() - last_control_time >=  ros::Duration(2.0))
                {
                 uav_control_pos(global_target);
                 last_control_time = ros::Time::now();
                }


                // ROS_INFO("current - Position: [X: %f, Y: %f]", 
                // hypot(dLat,dLon),
                // abs(current_rel_position_alt  - global_target.altitude)
                // );

                if (hypot(dLat,dLon) < hover_R + 0.5 && abs(current_rel_position_alt  - global_target.altitude) < 2) /// && abs(current_rel_position_alt  - global_target.altitude) < 5  if (sqrt(dx*dx + dy*dy) < hover_R + 0.5 && abs(center_target.position.z - cruise_alt) < 2) // 30 m 圆域
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
                if(command_vel.header.frame_id == "PN_Control")
                {
                    if(ros::Time::now() - time_lost_track > ros::Duration(0.5))
                    {
                        ROS_INFO("PN Control");
                        time_lost_track = ros::Time::now();
                        mode_num = 3;
                        break;
                    }
                }
                else
                {
                    time_lost_track = ros::Time::now();
                }

                if(cruise_done == true)
                {
                    cruise_done == false;
                    move_state = GO_EAST;
                    current_y = min_y;
                    mode_num = 1;

                if (set_mode_client.call(takeoff_mode) && takeoff_mode.response.mode_sent)
                {
                    ROS_INFO("AUTO enabled");
                    flag_init_position = false;
                    break;
                }
                else
                {
                    ROS_WARN("Failed to set AUTO, retry in 2 s ...");
                }
                // last_request = ros::Time::now();


                }
                else
                {
                    // ROS_INFO("Cruise Control :%d", move_state);
                    // printf("mode_num = %d\r\n",mode_num);

                    ROS_INFO("current - Position: [X: %f, Y: %f,current_y:%f]", 
                    pose_mav_info.pose.position.x,
                    pose_mav_info.pose.position.y ,
                    current_y
                    );
                    switch(move_state)
                    {
                        case GO_EAST:
                        {
                            vel.velocity.x = cruise_speed;
                            vel.velocity.y = 0;
                            vel.velocity.z = 0;

                            
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
                            vel.velocity.z = 0;

                           if(pose_mav_info.pose.position.x <= min_x + wp_radius)
                            {
                                last_was_east = false;
                                move_state = TURN_NORTH;
                                current_y += line_spacing;
                                if(current_y >= max_y) 
                                {
                                    cruise_done = true;
                                }
                            }
                            break;
                        }
                        case TURN_NORTH:
                        {
                            /* 向北飞一行间距 */
                            vel.velocity.x = 0;
                            vel.velocity.y = cruise_speed;
                            vel.velocity.z = 0;

                            if(pose_mav_info.pose.position.y >= current_y - wp_radius)
                            {
                                move_state = (last_was_east ? GO_WEST : GO_EAST);
                            }
                            break;
                        }
                    }
                if(ros::Time::now() - last_control_time >= CONTROL_INTERVAL)
                {
                  uav_control_vel_local(vel);
                 last_control_time = ros::Time::now();
                }
                   
                }
                break;
            }
            case 3:
            {
                move_state = GO_EAST;
                current_y = min_y;
                cruise_done = false;
                if(command_vel.header.frame_id == "PN_Control")
                {
                    vel.velocity.x = command_vel.twist.linear.x;
                    vel.velocity.y = command_vel.twist.linear.y;
                    vel.velocity.z = command_vel.twist.linear.z;
                    time_lost_track = ros::Time::now();
                    // uav_control_vel_local(vel);
                    if(ros::Time::now() - last_control_time >= CONTROL_INTERVAL)
                    {
                     uav_control_vel_local(vel);
                     last_control_time = ros::Time::now();
                    }
                }
                else
                {
                    // uav_control_vel_local(vel);
                    if(ros::Time::now() - last_control_time >= CONTROL_INTERVAL)
                    {
                     uav_control_vel_local(vel);
                     last_control_time = ros::Time::now();
                   }
                    if(ros::Time::now() - time_lost_track > ros::Duration(3.0))
                    {
                        ROS_INFO("Target lost, resuming cruise");
                        vel.header.frame_id = "none";
                        last_request = ros::Time::now();
                        mode_num = 1;
                    }
                }
                break;
            }
        }
        ros::spinOnce();
        rate.sleep();
    }
}