#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
import tf
import numpy as np
from math import sin, cos
import sys
import os
from filter3d import MedianThenLowPass3D
from sensor_msgs.msg import CameraInfo

# 低通滤波
FS = 50 # 采样频率
FC = 8  # 截止频率

MEMORY_GUIDANCE = True          # 是否启用记忆制导
MEMORY_TIMEOUT  = 5          # 记忆制导最长持续时间（s）

RMAG_TO_STRIKE = 1.5
V_MIN_XY = 0.05   # 水平面速度死区
V_MIN_Z  = 0.05   # 垂向速度死区
MAX_QDOT_PITCH =  30.0 * np.pi / 180.0   # 俯仰方向最大角速度，rad/s
MAX_QDOT_YAW   =  90.0 * np.pi / 180.0   # 偏航方向最大角速度，rad/s

DT = 0.15  # 导引率的采样周期
VZ_MAX = 2.5
VXY_MAX = 20

class Proportional_Navigation_Law:
    def __init__(self):
        self.filter = MedianThenLowPass3D(fs=FS, fc=FC)
        self.drone_position_cb_flag = False
        self.target_receive_flag = False
        self.drone_velocity_cb_flag = False

        self.last_target_t  = None
        self.last_target_pos  = None
        self.target_state_rec = np.zeros(6)
        self.target_in_img = False

        self.memory_active = False   # 是否处于记忆制导状态
        self.memory_start_t = None   # 进入记忆制导的时刻
        self.last_valid_seeker = None   # 丢失瞬间的相对状态（用于记忆）
        self.last_vm = None     
        self._t_last = None
        self.uav_stable_flag = False
        self.ready_to_memory = False
        self.stable_target_flag = False
        self.target_seen_t0 = None
        self.int_x = 0
        self.int_y = 0
        self.int_z = 0
        self.int_r = np.zeros(3)

    def left_cam_info_callback(self, msg: CameraInfo):
        self.left_K = np.array(msg.K).reshape(3, 3)
        self.left_D = msg.D
        self.left_f_x = self.left_K[0, 0]
        self.left_f_y = self.left_K[1, 1]
        self.left_width = msg.width
        self.left_height = msg.height

    def is_it_lost(self, msg: PointStamped):
        self.target_in_img = False
        margin_x = self.left_width / 10
        margin_y = self.left_height / 10
        x_min = margin_x
        x_max = self.left_width - margin_x
        y_min = margin_y
        y_max = self.left_height - margin_y
        px = msg.point.x
        py = msg.point.y
        if msg.header.frame_id == "Left_Camera_Optical_Frame":
            now = rospy.Time.now()
            self.target_in_img = True

            if self.stable_target_flag == False:
                if (px < x_min) or (px > x_max) or (py < y_min) or (py > y_max):
                    # 在边缘 1/8 区域
                    self.target_in_img = False
            
            if self.target_in_img == True:
                if self.target_seen_t0 is None:
                    self.target_seen_t0 = now
                if now - self.target_seen_t0 < rospy.Duration(0.5):
                    self.stable_target_flag == True
            else:
                self.target_seen_t0 = None
        else:
            self.target_in_img = False

    def is_uav_takeup(self, msg: PositionTarget):
        if msg.header.frame_id == "map":
            self.uav_stable_flag = True


    def position_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position
        self.orientation = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        self.euler_angles = tf.transformations.euler_from_quaternion(self.orientation, 'sxyz')
        self.drone_position_cb_flag = True

    def velocity_callback(self, msg: TwistStamped):
        self.linear_velocity = msg.twist.linear
        self.angular_velocity = msg.twist.angular
        self.drone_velocity_cb_flag = True

    def uav_state_to_missile_state(self, position, linear_velocity):
        x, y, z = position[0], position[1], position[2]
        vx, vy, vz = linear_velocity[0], linear_velocity[1], linear_velocity[2]
        V = np.sqrt(vx**2 + vy**2 + vz**2)
        V       = np.sqrt(vx**2 + vy**2 + vz**2)
        V_hor   = np.sqrt(vx**2 + vy**2)          # 水平合速度

        psi = np.arctan2(vy, vx)
        gamma = np.arctan2(vz, V_hor)
        # return np.array([x, y, z, V, psi, gamma])
        return np.array([x, y, z, vx, vy, vz])

    def target_pos_cb(self, msg: PointStamped):
        t = msg.header.stamp.to_sec()
        xyz_new = np.array([msg.point.x, msg.point.y, msg.point.z])
        p_flt = self.filter.update(xyz_new) 

        if self.last_target_t is None:
            # 只填位置
            self.target_state_rec[:3] = p_flt
            self.target_state_rec[3:] = 0.
            self.target_receive_flag = True
            self.last_target_t, self.last_target_pos = t, p_flt
            return
        else:
            # 估计速度
            dt = max(t - self.last_target_t, 1e-6)
            dp = p_flt - self.last_target_pos
            v_x = dp[0] / dt
            v_y = dp[1] / dt
            v_xy = np.hypot(dp[0], dp[1]) / dt
            v_z  = abs(dp[2]) / dt

            target_psi = np.arctan2(dp[1], dp[0])
            target_gamma = np.arctan2(dp[2], np.hypot(dp[0], dp[1]))

            # 总速度
            V = np.linalg.norm(dp) / dt 
            # 写回状态向量
            # self.target_state_rec[:] = [*p_flt, V, target_psi, target_gamma]
            self.target_state_rec[:] = [*p_flt, v_x, v_y, v_z]

        # 缓存这一帧
        self.last_target_t, self.last_target_pos = t, p_flt
        self.target_receive_flag = True

    def get_seeker_state(self, target_state, missile_state):
        """
        计算“导引头状态”，即在惯性坐标系下的相对位置与速度
        :param target_state: 目标状态
        :param missile_state: 导弹状态
        :return: 惯性坐标系下的相对位置与速度
        """

        R = np.asarray(target_state[:3]) - np.asarray(missile_state[:3])
        V = np.asarray(target_state[3:6]) - np.asarray(missile_state[3:6])
        return np.concatenate((R, V))

    def pn_guidance_predict(self, seeker_state_6d):
        """
        用丢失瞬间的相对运动状态，外推当前 LOS 并执行 PN
        相当于‘零效脱靶量’预测，再 PN
        """
        # 外推时间
        t_go = max((rospy.Time.now() - self.memory_start_t).to_sec(), 0.0)

        # 丢失瞬间状态
        R0 = seeker_state_6d[:3]
        V0 = seeker_state_6d[3:]
        # 匀速外推（目标直线假设）
        R_pred = R0 + V0 * t_go
        V_pred = V0
        # 组合成预测 seeker 状态
        seeker_pred = np.concatenate((R_pred, V_pred))
        # 标准 PN
        return self.pn_guidance(seeker_pred)

    def pn_guidance(self, seeker_state):
        """
        使用比例导引律提供指令加速度
        :param seeker_state: 当前导引头状态
        :return: 惯性坐标系下的指令加速度
        """
        NXY = 6
        NZ = 4
        r_dead = RMAG_TO_STRIKE
        Vc_max = 800.0
        R = np.asarray(seeker_state[:3], dtype=float)
        V = np.asarray(seeker_state[3:], dtype=float)
        r_mag = np.linalg.norm(R)

        # 极近区：纯追击，直接指向目标，加速度 = 0
        if r_mag < r_dead:
            K_align = 2.0  # 平行增益
            return -K_align * V 
        LOS_unit = R / r_mag
        Vc = -np.dot(R, V) / r_mag

        # Vc 符号翻转或过大 → 限幅
        Vc = np.clip(Vc, -Vc_max, Vc_max)
        Omega = np.cross(R, V) / (r_mag * r_mag)

        # 距离加权：self.r_mag 越小，N 越小，即增益退避方法
        gain = np.tanh(np.maximum(r_mag, 1e-3) / (20 * r_dead))
        Nxy_eff = NXY * np.clip(gain, 0.2, 1)
        Nz_eff = NZ * np.clip(gain, 0.4, 1)

        a_base = Vc * np.cross(Omega, LOS_unit)
        # 水平与垂直分量
        a_xy = Nxy_eff * a_base[:2]
        a_z  = Nz_eff * a_base[2]
        # a_xy = a_base[:2]
        # a_z  = a_base[2]
        
        a_cmd = np.array([a_xy[0], a_xy[1], a_z])

        print("com_a", a_cmd)
        return a_cmd

    def body_to_world_matrix(self, psi, gma, phi):
        """构建 Z-Y-X 旋转顺序的弹体->世界旋转矩阵"""
        cpsi, cgm, cph = np.cos(psi), np.cos(gma), np.cos(phi)
        spsi, sgm, sph = np.sin(psi), np.sin(gma), np.sin(phi)
        
        R = np.array([
            [cpsi*cgm, cpsi*sgm*sph - spsi*cph, cpsi*sgm*cph + spsi*sph],
            [spsi*cgm, spsi*sgm*sph + cpsi*cph, spsi*sgm*cph - cpsi*sph],
            [   -sgm,               cgm*sph,               cgm*cph]
        ])
        return R

    def get_commanded_accel(self, seeker_now, missile_state):
        """
        统一封装：丢目标判别 + 记忆制导 / 正常 PN 选择
        seeker_now : 当前量测（EKF 分支用 new_seeker_est，其余分支用 seeker_state）
        返回 -> commanded_accel, 是否强制 break
        """
        R = np.asarray(seeker_now[:3], dtype=float)
        # ---- 丢目标判别 ----
        target_lost = False
        if MEMORY_GUIDANCE:
            if self.target_in_img:
                self.ready_to_memory = True

            if not self.target_in_img and self.ready_to_memory == True:
                # rospy.loginfo(np.linalg.norm(seeker_now[3:5]))
                target_lost = True
            else:
                target_lost = False
                self.memory_active = False

        # ---- 首次丢失 → 启动记忆制导 ----
        if MEMORY_GUIDANCE and target_lost and (not self.memory_active):
            self.memory_active = True
            self.memory_start_t  = rospy.Time.now()
            self.last_valid_seeker = seeker_now.copy()

            self.last_vm = np.linalg.norm(missile_state[3:6]) 
            print(f"[{rospy.Time.now().to_sec():.2f}s] 目标丢失 → 进入记忆制导")
            return self.pn_guidance_predict(self.last_valid_seeker), False
        
        if MEMORY_GUIDANCE and self.memory_active and target_lost:
            if rospy.Time.now() - self.memory_start_t < rospy.Duration(MEMORY_TIMEOUT):
                print("记忆制导中", (rospy.Time.now() - self.memory_start_t).to_sec())
                return self.pn_guidance_predict(self.last_valid_seeker), False
            else:
                # 超时
                self.memory_active = False
                print(f"[{rospy.Time.now().to_sec():.2f}s] 记忆制导超时，放弃捕获")
                # print("--seeker_state2--", seeker_state)
                self.ready_to_memory = False
                return np.zeros(3), True
        
        return self.pn_guidance(seeker_now), False
    
def print_missile_state(state: np.ndarray):
        x, y, z, V, psi, gamma = state
        lines = [
            "-------- Missile State --------",
            f"  x : {x:8.2f} m",
            f"  y : {y:8.2f} m",
            f"  z : {z:8.2f} m",
            f"  V : {V:8.2f} m/s",
            f"psi : {np.degrees(psi):7.2f} °",
            f"gma : {np.degrees(gamma):7.2f} °",
            "-------------------------------"
        ]
        up7_home = "\033[7A\033[G"
        print(up7_home + "\n".join(lines), end='')
        # 立即刷新，防止缓冲
        sys.stdout.flush()

def print_target_state(state: np.ndarray):
        # x, y, z, V, psi, gamma = state
        # lines = [
        #     "-------- Target State --------",
        #     f"  x : {x:8.2f} m",
        #     f"  y : {y:8.2f} m",
        #     f"  z : {z:8.2f} m",
        #     f"  V : {V:8.2f} m/s",
        #     f"psi : {np.degrees(psi):7.2f} °",
        #     f"gma : {np.degrees(gamma):7.2f} °",
        #     "-------------------------------"
        # ]
        # up7_home = "\033[7A\033[G"
        # print(up7_home + "\n".join(lines), end='')
        # # 立即刷新，防止缓冲
        # sys.stdout.flush()

        # 普通print
        x, y, z, V, psi, gamma = state
        print("-------- Target State --------")
        print(f"  x : {x:8.2f} m")
        print(f"  y : {y:8.2f} m")
        print(f"  z : {z:8.2f} m")
        print(f"  V : {V:8.2f} m/s")
        print(f"psi : {np.degrees(psi):7.2f} °")
        print(f"gma : {np.degrees(gamma):7.2f} °")
        print("-------------------------------")

if __name__ == "__main__":
    rospy.init_node("pn_guidance")
    rospy.loginfo("PN init")
    pn_law = Proportional_Navigation_Law()
    
    rospy.Subscriber("/tailsitter_0/mavros/local_position/odom", Odometry, pn_law.position_callback)
    rospy.Subscriber("/tailsitter_0/mavros/local_position/velocity_local", TwistStamped, pn_law.velocity_callback)
    rospy.Subscriber("/detection/odom/target_point", PointStamped, pn_law.target_pos_cb)
    rospy.Subscriber("/detection/ultralytics/left_target_point_pub", PointStamped, pn_law.is_it_lost)
    rospy.Subscriber("/tailsitter_0/mavros/setpoint_raw/local", PositionTarget, pn_law.is_uav_takeup)
    command_vel_pub = rospy.Publisher("/navigation/command_velocity_local", TwistStamped, queue_size=5)
    rospy.Subscriber("/stereo/left/camera_info", CameraInfo, pn_law.left_cam_info_callback)
       
    rate = rospy.Rate(1/DT)
    # os.system('clear')
    while not rospy.is_shutdown():
        if pn_law.target_receive_flag == True and pn_law.drone_position_cb_flag == True and pn_law.drone_velocity_cb_flag == True:
            command_t0 = rospy.Time.now().to_sec()
            commanded_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            tau_xy = 1.19  # 无人机水平速度响应时间常数
            tau_z = 0.34 * 1.1  # 无人机垂直速度响应时间常数
            max_acc_xy = 20.0
            max_acc_z  = 15.0
            dt = DT
            current_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            current_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            simulation_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            simulation_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            last_com_accel = None
            break

    while not rospy.is_shutdown():
            if pn_law.uav_stable_flag != True:
                continue
            command_vel = TwistStamped()
            if pn_law.target_in_img == False and pn_law.memory_active == False:
                # print("------------------reset----------------------")
                command_vel.header.frame_id = "Cruise_Control"
                command_t0 = rospy.Time.now().to_sec()
                commanded_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
                dt = DT
                current_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
                current_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
                simulation_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
                simulation_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            else:
                command_vel.header.frame_id = "PN_Control"

            # 提前估计无人机状态
            simulation_position += simulation_velocity * dt
            missile_state = pn_law.uav_state_to_missile_state(simulation_position, simulation_velocity)
            target_state = pn_law.target_state_rec
            # print_missile_state(missile_state)
            # print_target_state(target_state) # 调试时两者只开其一
            distance = np.linalg.norm(missile_state[:3]-target_state[:3])
            seeker_state = pn_law.get_seeker_state(target_state, missile_state)
           
            commanded_accel, force_break = pn_law.get_commanded_accel(seeker_state, missile_state)
            current_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            
            commanded_velocity += commanded_accel * dt

            # print("com_acc",commanded_accel)
            # print("target_pos:", np.array([target_state[0], target_state[1], target_state[2]]))
            print("SEEK_pos:", np.array([seeker_state[0], seeker_state[1], seeker_state[2]]))
            # print("SEEK_V:", np.array([seeker_state[3], seeker_state[4], seeker_state[5]]))
            print("current",np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z]))
            # print("simulation",simulation_velocity)
            print("——————————————————")

            # 限速
            dv_xy = commanded_velocity[:2] - current_velocity[:2]
            dv_z  = commanded_velocity[2]  - current_velocity[2]

            dv_xy = np.clip(dv_xy, -max_acc_xy * dt, max_acc_xy * dt)
            dv_z  = np.clip(dv_z,  -max_acc_z * dt,  max_acc_z * dt)    

            commanded_velocity[:2] = current_velocity[:2] + dv_xy
            commanded_velocity[2]  = current_velocity[2]  + dv_z
            commanded_velocity[2]  = np.clip(commanded_velocity[2], -VZ_MAX, VZ_MAX) 
            commanded_velocity[:2] = np.clip(commanded_velocity[:2], -VXY_MAX, VXY_MAX)   

            simulation_velocity[:2] += (commanded_velocity[:2] - simulation_velocity[:2]) * dt / tau_xy
            simulation_velocity[2]  += (commanded_velocity[2] - simulation_velocity[2]) * dt / tau_z

            command_vel.twist.linear.x = commanded_velocity[0]
            command_vel.twist.linear.y = commanded_velocity[1]
            # “无人机指令坐标系“z轴向下
            command_vel.twist.linear.z = -commanded_velocity[2]
            print("command:",np.array([commanded_velocity[0], commanded_velocity[1], commanded_velocity[2]]))
            
            command_vel_pub.publish(command_vel)
            
            if force_break:
                continue

            rate.sleep()