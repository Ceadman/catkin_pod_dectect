//
// Created by xyl on 24-5-15.
//

#ifndef SRC_CONTROLLER_H
#define SRC_CONTROLLER_H

#include "PID.hpp"
#include "ros/ros.h"

#include "Eigen/Dense"

#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/PositionTarget.h>

class PID_Controller {
private:
    PID<double> * x_pid_controller;
    PID<double> * y_pid_controller;
    PID<double> * z_pid_controller;
    PID<double> * yaw_pid_controller;
    double offset_docking_x = 0;
    double offset_docking_y = 0;
    double offset_docking_z = 0;

public:
    PID_Controller();//构造函数赋初值
    ~PID_Controller();//析构函数重置
    mavros_msgs::PositionTarget control(double error_x,double error_y,double error_z);
};


#endif //SRC_CONTROLLER_H
