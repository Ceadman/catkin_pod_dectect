
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
#include <mavros_msgs/AttitudeTarget.h>
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
#include <mavros_msgs/AttitudeTarget.h> 


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

ros::Publisher local_pub, att_pub_;
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
const double hover_R = 60.0;        // �����뾶 30 m
// mavros_msgs::PositionTarget center_init;   // ��ʼԲ��
// mavros_msgs::PositionTarget center_takeup;
// mavros_msgs::PositionTarget center_target;

mavros_msgs::GlobalPositionTarget global_target;
mavros_msgs::GlobalPositionTarget center_init;   // ��ʼԲ��
mavros_msgs::GlobalPositionTarget center_takeup;
mavros_msgs::GlobalPositionTarget center_target;
float current_rel_position_alt = 0;
static const double DEG_PER_M = 1.0 / 111320.0;

mavros_msgs::AttitudeTarget att_cmd;



bool circle_init = false;     // �Ѱ�Բ�ļ���
geometry_msgs::TwistStamped command_vel;

ros::ServiceClient cmd_int_client;

void uav_control_vel_local(mavros_msgs::PositionTarget& vel_set)
{
    static ros::Time last_time = ros::Time::now();
    
    ros::Time current_time = ros::Time::now();
    
    float dt = (current_time - last_time).toSec();
    
    last_time = current_time;
    
    if (dt <= 0 || dt > 1.0) {
        dt = 0.1;
    }


    float airSpeed = sqrt(vel_set.velocity.x * vel_set.velocity.x +vel_set.velocity.y * vel_set.velocity.y);   
    double yaw_rad = atan2(vel_set.velocity.x, vel_set.velocity.y);
    // 在 x=1, y=0 时得到 π/2, x=0, y=1 得到 0
    double yaw_deg = yaw_rad * 180 / M_PI;

    if (yaw_deg < 0) {
        yaw_deg += 360.0;
    }

    if(airSpeed <18){
        airSpeed =18;
    }else if(airSpeed >24){
        airSpeed =24;
    }

    float current_altitude = pose_mav_info.pose.position.z;  // ��ǰ�߶ȣ������ɵ㣩
    // ros::Time current_time = ros::Time::now();
    if(vel_set.velocity.z > 8)
    {
        vel_set.velocity.z = 8;
    }
    else if(vel_set.velocity.z < -8)
    {
        vel_set.velocity.z = -8;
    }
    float target_altitude = current_altitude + vel_set.velocity.z * dt;  // d
   
    mavros_msgs::CommandInt cmd;
    cmd.request.frame = 0;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd.request.command = 43000;   // MAV_CMD_DO_REPOSITION
    cmd.request.current = 0;     // GUIDED
    cmd.request.autocontinue = 0;
    
    cmd.request.param1 = 0; 
    cmd.request.param2 = 18;  
    cmd.request.param3 = 2.0;  
    cmd.request.param4 = 0;   
    cmd.request.x = 0;  
    cmd.request.y = 0;  
    cmd.request.z = 0;   
    
    
    if(cmd_int_client.call(cmd)){
         ROS_INFO("CommandInt VEL sent, result: %d", cmd.response.success);
        /// ROS_INFO("CommandInt dt sent, result: %f", dt);

    }


    mavros_msgs::CommandInt cmd1;
    cmd1.request.frame = 0;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd1.request.command = 43002;   // MAV_CMD_DO_REPOSITION
    cmd1.request.current = 0;     // GUIDED
    cmd1.request.autocontinue = 0;
    
    cmd1.request.param1 = 1; 
    cmd1.request.param2 = yaw_deg;  
    cmd1.request.param3 = 10.0;  
    cmd1.request.param4 = 0;   
    cmd1.request.x = 0;  
    cmd1.request.y = 0;  
    cmd1.request.z = 0;   
    

    double current_yaw = fmod(450 - pose_mav_info_yaw * 180.0 / M_PI, 360.0);
    if(current_yaw <0){
        current_yaw += 360.0;
    }
    
    if(cmd_int_client.call(cmd1)){
      // ROS_INFO("CommandInt Yaw sent, result: %f,%f", yaw_deg, current_yaw);
    }

    mavros_msgs::CommandInt cmd2;
    cmd2.request.frame = 3;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd2.request.command = 43001;   // MAV_CMD_DO_REPOSITION
    cmd2.request.current = 0;     // GUIDED
    cmd2.request.autocontinue = 0;
    
    cmd2.request.param1 = 0; 
    cmd2.request.param2 = 0;  
    cmd2.request.param3 = 0; 
    cmd2.request.param4 = 0;   
    cmd2.request.x = 0;  
    cmd2.request.y = 0;  
    cmd2.request.z = target_altitude;   
    
    
    if(cmd_int_client.call(cmd2)){
        // ROS_INFO("CommandInt dt sent, result: %f", dt);
   //    ROS_INFO("CommandInt altitude sent, result: %f,%f,%f", target_altitude, current_altitude, vel_set.velocity.z);
     }

}


void uav_control_pos(mavros_msgs::GlobalPositionTarget& pos_set)
{

    mavros_msgs::CommandInt cmd;
    cmd.request.frame = 6;       // MAV_FRAME_GLOBAL_RELATIVE_ALT
    cmd.request.command = 192;   // MAV_CMD_DO_REPOSITION
    cmd.request.current = 0;     // GUIDED
    cmd.request.autocontinue = 0;
    
    cmd.request.param1 = -1;  
    cmd.request.param2 = 0;  
    cmd.request.param3 = 0;  
    cmd.request.param4 = 0;   
    
    // CommandInt
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
    // if (flag_init_position ==false && (msg->pose.position.z!=0)) 
    // {
	// 	init_position_take_off.pose.position.x = msg->pose.position.x;
	//     init_position_take_off.pose.position.y = msg->pose.position.y;
	//     init_position_take_off.pose.position.z = msg->pose.position.z;
    //     flag_init_position = true;  
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
    m.getRPY(pose_mav_info_roll, pose_mav_info_pitch, pose_mav_info_yaw); // ��λ rad
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


void pose_command_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    att_cmd.header = msg->header;
    att_cmd.header.frame_id = "base_link"; 

    att_cmd.orientation = msg->pose.orientation;
    att_cmd.type_mask = 7;

    att_cmd.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ROLL_RATE | 
                    mavros_msgs::AttitudeTarget::IGNORE_PITCH_RATE |
                    mavros_msgs::AttitudeTarget::IGNORE_YAW_RATE;
                   /// mavros_msgs::AttitudeTarget::IGNORE_THRUST;
                 
    
    att_cmd.thrust = 0.7;///msg->pose.position.x;
    if (att_cmd.thrust < 0.0) att_cmd.thrust = 0.0;
    if (att_cmd.thrust > 1.0) att_cmd.thrust = 1.0;
   // ROS_INFO("CommandIntsssssss sent, resultrrrrrrrrrrrrrrrrr");
    
    
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "plane_track_node");//��ʼ��һ��ROS�ڵ�
    ros::NodeHandle nh;
    ros::Rate rate(20);

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "GUIDED";//  GUIDED  OFFBOARD
    mavros_msgs::SetMode takeoff_mode;
    takeoff_mode.request.custom_mode = "AUTO";
    ros::Time last_request = ros::Time::now();//��������ʱ��
    ros::Time time_lost_track = ros::Time::now();
    ros::Time the_last_unattained_height = ros::Time::now();
    mavros_msgs::CommandBool arm_cmd;    //mavros_msgs::CommandBool��ROS��Ϣ���ͣ����ڽ������������˻�
    arm_cmd.request.value = true;        //��arm_cmd.request.value����Ϊtrue����ʾ�������˻�

    bool wp_uploaded = false; 
    mavros_msgs::WaypointPush wp_push;
    mavros_msgs::Waypoint wp;
    const double cruise_alt  = ALTITUDE_TAKEOFF;   // Ѳ���߶�
    const double cruise_speed = CRUISE_SPEED;   // ˮƽѲ���ٶ�
    const double wp_radius    = WP_RADIUS;    // ��Ϊ����ð뾶���ɻ���
    const double line_spacing = LINE_SPACING;   // ÿ�м��
    const double freq         = 20.0;   // ����Ƶ��
    int mode_num = 0; 
    double min_x, max_x, min_y, max_y;
    double current_y;
    bool cruise_done = false;
    enum CruiseState {GO_EAST, GO_WEST, TURN_NORTH} move_state;
    bool last_was_east = true;
    bool get_target_flag = false;



    // ��ѭ���ⶨ���ʱ����
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
    ros::Subscriber pose_command_sub = nh.subscribe<geometry_msgs::PoseStamped>
                    ("/navigation/attitude_cmd", 5, pose_command_cb);
    att_pub_ = nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 10);
     
    

    

    while(ros::ok())
    {
        if (!circle_init && current_mode.armed && current_mode.mode == "GUIDED")  ///��¼��һ����guidedģʽ��λ��
        {
            center_init.latitude  = current_global_pos.latitude;
            center_init.longitude = current_global_pos.longitude;
             center_init.altitude  = current_rel_position_alt+ALTITUDE_TAKEOFF;  
             circle_init = true;
             ROS_INFO("GUIDED 444433 - Position: [X: %f, Y: %f, Z: %f]", 
            center_init.latitude,
            center_init.longitude,
            center_init.altitude);
        }

        if(command_vel.header.frame_id == "PN_Control"){
            // ROS_INFO_DELAYED_THROTTLE(0.5, "TARGET IN");
        }
        
        if(current_mode.mode == "GUIDED" && command_vel.header.frame_id == "PN_Control")                   ///����guidedģʽ����ʶ��Ŀ��
        {
            mode_num = 2;
        } 
        else if(current_mode.mode == "GUIDED" && command_vel.header.frame_id != "PN_Control")            ///����guidedģʽ��û��Ŀ�꣬���ص���ʼ������
        { 
               
            if (set_mode_client.call(takeoff_mode) && takeoff_mode.response.mode_sent )
            {
                /// att_pub_.publish(att_cmd);
                ROS_INFO("AUTO enabled");
            }
          
            mode_num = 1;
        }else{
           
            mode_num = 0;                                                                                    ///��������ģʽ
        }



        // printf("mode_num = %d\r\n",mode_num);
        switch(mode_num)
        {
            case 1:
            {
                break;
            }
            case 2:
            {
                if(current_mode.mode == "GUIDED" && command_vel.header.frame_id == "PN_Control")
                {
                  vel.velocity.x = command_vel.twist.linear.x;
                  vel.velocity.y = command_vel.twist.linear.y;
                  vel.velocity.z = command_vel.twist.linear.z;
                  time_lost_track = ros::Time::now();
                  if(ros::Time::now() - last_control_time >= CONTROL_INTERVAL)
                  {
                    ///uav_control_vel_local(vel);
                    att_pub_.publish(att_cmd);
                    ///ROS_INFO_DELAYED_THROTTLE(0.5, "TARGET IN");
                    last_control_time = ros::Time::now();
                  }
                }else{
                    mode_num = 1; 
                }
                break;
            }

          default:
           {
            break;
           }

        }
        ros::spinOnce();
        rate.sleep();
    }
}