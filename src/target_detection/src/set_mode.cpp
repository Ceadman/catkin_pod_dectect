#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>  

// basic modes from MAV_MODE
// uint8 MAV_MODE_PREFLIGHT = 0
// uint8 MAV_MODE_STABILIZE_DISARMED = 80
// uint8 MAV_MODE_STABILIZE_ARMED = 208
// uint8 MAV_MODE_MANUAL_DISARMED = 64
// uint8 MAV_MODE_MANUAL_ARMED = 192
// uint8 MAV_MODE_GUIDED_DISARMED = 88
// uint8 MAV_MODE_GUIDED_ARMED = 216
// uint8 MAV_MODE_AUTO_DISARMED = 92
// uint8 MAV_MODE_AUTO_ARMED = 220
// uint8 MAV_MODE_TEST_DISARMED = 66
// uint8 MAV_MODE_TEST_ARMED = 194

// uint8 base_mode # filled by MAV_MODE enum value or 0 if custom_mode != ''
// string custom_mode // string mode representation or integer
bool success = false;
bool aim_flag = false;


//建立一个订阅消息体类型的变量，用于存储订阅的信息
//订阅的消息体的头文件，该消息体的类型为mavros_msgs::State
//查看无人机的状态

// 解析如下：
// header：消息头，包含时间戳和框架信息；
//connected：表示是否连接到了 mavros 节点；
// armed：表示无人机当前是否上锁；
// guided：表示无人机当前是否处于 GUIDED 模式；
// mode：表示当前无人机所处的模式，包括以下几种：
// MANUAL，ACRO，ALTCTL，POSCTL，OFFBOARD，STABILIZE，RATTITUDE，AUTO.MISSION
// AUTO.LOITER，AUTO.RTL，AUTO.LAND，AUTO.RTGS，AUTO.READY，AUTO.TAKEOFF

mavros_msgs::State current_state;
 
//订阅时的回调函数，接受到该消息体的内容时执行里面的内容，内容是储存飞控当前的状态
void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
    ROS_INFO("STATUS:%d,%d",current_state.connected,current_state.armed);
}

/*
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
geometry_msgs/Pose pose
  geometry_msgs/Point position
    float64 x
    float64 y
    float64 z
  geometry_msgs/Quaternion orientation
    float64 x
    float64 y
    float64 z
    float64 w
*/
geometry_msgs::PoseStamped current_pos;
void pos_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
    current_pos = *msg;

    ROS_INFO("STATUS:");
}
/*
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
geometry_msgs/Quaternion orientation
  float64 x
  float64 y
  float64 z
  float64 w
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
  float64 x
  float64 y
  float64 z
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
  float64 x
  float64 y
  float64 z
float64[9] linear_acceleration_covariance
*/
sensor_msgs::Imu current_imu;
void imu_cb(const sensor_msgs::Imu::ConstPtr& msg){
    current_imu = *msg;
    // ROS_INFO("STATUSa33aa:%f,%f,%f",current_imu.angular_velocity.x,current_imu.angular_velocity.y,current_imu.angular_velocity.z);
}

/*
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
geometry_msgs/Twist twist
  geometry_msgs/Vector3 linear
    float64 x
    float64 y
    float64 z
  geometry_msgs/Vector3 angular
    float64 x
    float64 y
    float64 z

*/

geometry_msgs::TwistStamped current_vel;
void vel_cb(const geometry_msgs::TwistStamped::ConstPtr& msg){
    current_vel = *msg;
    // ROS_INFO("STATUSa33aawww:");
}

void target_callback(const geometry_msgs::PointStamped::ConstPtr& msg){
        if(msg->header.frame_id == "Image_Frame_AIM")
        {
            aim_flag = true;
        }
        else{
            aim_flag = false;
        }
            
}

int main(int argc, char **argv)
{
    //ros系统的初始化，argc和argv在后期节点传值会使用，最后一个参数offb_node为节点名称
    ros::init(argc, argv, "setmode_node"); 

    //实例化ROS句柄，这个ros::NodeHandle类封装了ROS中的一些常用功能
    ros::NodeHandle nh;

    //这是一个订阅者对象，可以订阅无人机的状态信息（状态信息来源为MAVROS发布），用来储存无人机的状态，在回调函数中会不断更新这个状态变量
    //<>里面为模板参数，传入的是订阅的消息体类型，（）里面传入三个参数，分别是该消息体的位置、缓存大小（通常为1000）、回调函数
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
    ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 10, pos_cb);
    ros::Subscriber VEL_sub = nh.subscribe<geometry_msgs::TwistStamped>("mavros/local_position/velocity", 10, vel_cb);
    ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>("mavros/imu/data", 10, imu_cb);
 
    ros::Subscriber tar_sub = nh.subscribe<geometry_msgs::PointStamped>("/detection/ultralytics/img_link_target_pub", 10, target_callback);
    //这是一个发布者对象，用来在本地坐标系下发布目标点，后面会以20Hz频率发布目标点
    //<>里面为模板参数，传入的是发布的消息体类型，（）里面传入两个参数，分别是该消息体的位置、缓存大小（通常为1000）

    //一个客户端，用来解锁无人机，这是因为无人机如果降落后一段时间没有收到信号输入，会自动上锁来保障安全
    //启动服务的函数为nh下的serviceClient<>()函数，<>里面是该服务的类型，（）里面是该服务的路径
    // ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");

    //一个客户端，用来切换飞行模式
    //启动服务的函数为nh下的serviceClient<>()函数，<>里面是该服务的类型，（）里面是该服务的路径
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

    //PX4在两个Offboard命令之间有一个500ms的延时，如果超过此延时，系统会将回到无人机进入Offboard模式之前的最后一个模式。
    ros::Rate rate(50.0);

    // 等待飞控和MAVROS建立连接，current_state是我订阅的MAVROS的状态，在收到心跳包之后连接成功跳出循环
    while(ros::ok() && !current_state.connected){
        ROS_INFO("Wait for MavInfo");

        ros::spinOnce();
        rate.sleep();
    }

    //实例化一个geometry_msgs::PoseStamped类型的对象，并对其赋值，最后将其发布出去
    //尽管PX4在航空航天常用的NED坐标系下操控飞机，但MAVROS将自动将该坐标系切换至常规的ENU坐标系下
    // geometry_msgs::PoseStamped pose;
    // pose.pose.position.x = 0;
    // pose.pose.position.y = 0;
    // pose.pose.position.z = 2;

    //建立一个类型为SetMode的服务端offb_set_mode，并将其中的模式mode设为"OFFBOARD"，作用便是用于后面的客户端与服务端之间的通信（服务）
    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "STABILIZE";
    mavros_msgs::SetMode offb_set_mode_end;
    offb_set_mode_end.request.custom_mode = "MANUAL";

    //建立一个类型为CommandBool的服务端arm_cmd，并将其中的是否解锁设为"true"，作用便是用于后面的客户端与服务端之间的通信（服务）
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    //更新时间
    ros::Time last_request = ros::Time::now();

    //大循环，只要节点还在ros::ok()的值就为正
    while(ros::ok())
    {
      ROS_INFO("aim_flag:%d", aim_flag);
       
        //首先判断当前模式是否为GUIDED模式，如果不是，则进入if语句内部
        //这里是3秒钟进行一次判断，避免飞控被大量的请求阻塞
        if( current_state.mode != "STABILIZE" && (ros::Time::now() - last_request > ros::Duration(3.0)))
        {
            if(aim_flag == true)
            {
                if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
                {
                    //打开GUIDED模式后在终端打印信息
                     ROS_INFO("STABILIZE enabled");
                }
                //更新时间
                last_request = ros::Time::now();
            } 
        }
        else if( current_state.mode != "MANUAL" && (ros::Time::now() - last_request > ros::Duration(3.0)))
        {
            if(aim_flag == false)
            {
                if( set_mode_client.call(offb_set_mode_end) && offb_set_mode_end.response.mode_sent)
                {
                    //打开GUIDED模式后在终端打印信息
                     ROS_INFO("MANUAL enabled");
                }
                //更新时间
                last_request = ros::Time::now();
            } 
        }
        

        else 
        {
            //判断当前状态是否解锁，如果没有解锁，则进入if语句内部
            //这里是5秒钟进行一次判断，避免飞控被大量的请求阻塞
            // if( !current_state.armed && (ros::Time::now() - last_request > ros::Duration(3.0)))
            // {
            //     //客户端arming_client向服务端arm_cmd发起请求call，然后服务端回应response成功解锁，则解锁成功
            //     if( arming_client.call(arm_cmd) && arm_cmd.response.success)
            //     {
            //         //解锁后在终端打印信息
            //         ROS_INFO("Vehicle armed");
            //     }
            //     //更新时间
            //     last_request = ros::Time::now();
            // }
        }
        
        //当spinOnce函数被调用时，会调用回调函数队列中第一个回调函数，这里回调函数是state_cb函数
        ros::spinOnce();
        //根据前面ros::Rate rate(20.0);制定的发送频率自动休眠 休眠时间 = 1/频率
        rate.sleep();
    }

    return 0;
}

