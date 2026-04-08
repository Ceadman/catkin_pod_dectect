//
// Created by xyl on 24-3-12.
//

#ifndef SRC_USER_ALGROTHM_H
#define SRC_USER_ALGROTHM_H

#include <cmath>
#include "Eigen/Dense"

template<typename T>
class PID
{
private:
    T error_p;//当前误差
    T error_i;//累计误差
    T error_d;//误差变化率
    T error_last;//上一次计算的误差值

    T separate_integral;//积分分离阈值

    T output_limit;//输出限幅

    float kp;//比例系数
    float ki;//积分系数
    float kd;//微分系数
public:
    PID(float p=0,float i=0,float d=0,float ol=0 ,float sp=0):kp(p) ,ki(i) ,kd(d) ,output_limit(ol) ,separate_integral(sp){

    }
    ~PID() = default;
    T PID_caculate(const T& target,const T& current);
    void reset();
    void set_output_limit(const T& _);
    void set_kp(const T& _);
    void set_ki(const T& _);
    void set_kd(const T& _);
    void set_ol(const T& _);
    void set_sp(const T& _);
};

template<typename T>
void PID<T>::reset() {//重置所有误差值，用于控制器重启
    error_p = 0;
    error_i = 0;
    error_d = 0;
    error_last = 0;
}

template<typename T>
void PID<T>::set_kp(const T &_) {//设置比例系数
    kp = _;
}
template<typename T>
void PID<T>::set_ki(const T &_) {//设置积分系数
    ki = _;
}
template<typename T>
void PID<T>::set_kd(const T &_) {//设置微分系数
    kd = _;
}
template<typename T>
void PID<T>::set_ol(const T &_) {//设置输出限幅（控制量的最大绝对值限制）
    output_limit = _;
}
template<typename T>
void PID<T>::set_sp(const T &_) {//设置积分分离阈值（当误差绝对值小于此值时启用积分项）
    separate_integral = _;
}
template<typename T>
T PID<T>::PID_caculate(const T& target,const T& current)
{
    error_p = target - current;

    if(fabs(error_p) < separate_integral)
        error_i += error_p;//误差小于积分阈值时累积误差
    else
        error_i =0;//超过积分阈值时直接置为0

    error_d =  error_p - error_last;//计算误差的变化量（微分值）

    T output = kp * error_p + ki * error_i + kd * error_d;//PID的输出值

    error_last = error_p;//记录上次的误差值


    if(output > output_limit)
        return output_limit;//如果输出值大于输出限幅则按输出限幅输出
    else if(output < -output_limit)
        return (-output_limit);//如果输出值大于输出限幅则按输出限幅输出
    else
        return output;//返回输出量
}
template <typename T>
void  PID<T>::set_output_limit(const T& _){//设置输出限幅
    output_limit = _;
}

#endif //SRC_USER_ALGROTHM_H
