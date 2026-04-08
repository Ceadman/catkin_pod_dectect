#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PointStamped, TwistStamped, AccelStamped, Vector3Stamped, PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
import tf
import numpy as np
from math import sin, cos
import sys
import os
from filter3d import MedianThenLowPass3D
# from filter3d import KalmanFilter3D
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32 

PREFORE_GUIDANCE = False 

IMAGE_HEIGHT = 1080
IMAGE_WIGTH = 1920

# 低通滤波
FS = 50 # 采样频率
FC = 8  # 截止频率

MEMORY_GUIDANCE = False         # 是否启用记忆制导
MEMORY_TIMEOUT  = 1.5          # 记忆制导最长持续时间（s）

RMAG_TO_STRIKE = 1
V_MIN_XY = 0.05   # 水平面速度死区
V_MIN_Z  = 0.05   # 垂向速度死区
MAX_QDOT_PITCH =  30.0 * np.pi / 180.0   # 俯仰方向最大角速度，rad/s
MAX_QDOT_YAW   =  90.0 * np.pi / 180.0   # 偏航方向最大角速度，rad

DT = 0.1         # 导引率的采样周期
VZ_MAX = 5
VXY_MAX = 30

K_FF = 0

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

class Proportional_Navigation_Law:
    def __init__(self):
        self.pod_width = IMAGE_WIGTH
        self.pod_height = IMAGE_HEIGHT
        # self.filter = KalmanFilter3D(dt=0.15, process_noise=0.1, measurement_noise=0.5)
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
        self.ready_to_memory = False
        self.stable_target_flag = False
        self.target_seen_t0 = None
        self.int_x = 0
        self.int_y = 0
        self.int_z = 0
        self.int_r = np.zeros(3)
        self.omega_los_map_future = np.array([0, 0])
        self.pod_angle_in_map = np.array([0, 0, 0])
        self.pod_angle_in_map_header = 1
        self.pod_mode = 0
        self.target_lost = False

        self.last_dist_t = None    # 上次时间戳
        self.last_dist = None      # 上次距离值
        self.est_Vc = 0.0  
        self.distance = 0

        self.last_distance = None 

        self.commanded_accel_last = np.array([0.0, 0.0, 0.0]) 
        self.g = 9.81
        
    def camera_focal_cb(self, msg: Float32):
        self.left_f_x = msg.data
        self.left_f_y = msg.data
    
    def pod_mode_cb(self, msg: Float32):
        self.pod_mode = msg.data

    def position_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position
        # rospy.loginfo_throttle(2.0, f"Altitude:{msg.pose.pose.position:.2f}")
        
        quat = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quat, 'sxyz')
        self.euler_angles = np.array([roll, pitch, yaw])
        self.orientation = tf.transformations.quaternion_from_euler(roll, -pitch, yaw, 'sxyz')
        self.drone_position_cb_flag = True
    
    def calculate_approach_velocity(self):
        """计算接近速度"""
        if self.linear_velocity is None or self.pod_angle_in_map is None:
            raise ValueError("请先设置速度和角度！")
        
        los_map = self.pod_angle_in_map
    
        # 归一化得到单位方向向量
        norm = np.linalg.norm(los_map)
        if norm < 1e-6:
            rospy.logwarn_throttle(2.0, "PN:Wait for pod_angle_in_map")
        
        direction = los_map / norm
        
        vel_array = np.array([
            self.linear_velocity.x,
            self.linear_velocity.y,
            self.linear_velocity.z
        ])
        # 计算接近速度（点积）
        approach_velocity = np.dot(vel_array, direction)
        
        return approach_velocity, direction
    
    def velocity_callback(self, msg: TwistStamped):
        self.linear_velocity = msg.twist.linear
        self.angular_velocity = msg.twist.angular
        self.est_Vc, direction = self.calculate_approach_velocity()
        self.drone_velocity_cb_flag = True

    def extract_spherical_rates(self, omega_enu, az, el):
        """
        将 ENU 三维角速度转换为二维球坐标角速度
        
        注意：此转换用于显示，数学上存在天顶奇点
        """
        wx, wy, wz = omega_enu
        cos_az, sin_az = np.cos(az), np.sin(az)
        cos_el, sin_el = np.cos(el), np.sin(el)
        
        # 俯仰率（抬头为正）
        # 几何：投影到俯仰轴 [-sin(az), cos(az), 0]
        el_rate = wy * cos_az - wx * sin_az
        
        # 方位率（逆时针为正）
        # 严格公式：包含 w_z 和倾斜耦合项
        if abs(cos_el) > 1e-3:
            # 方法A：完整公式（考虑倾斜耦合）
            az_rate = wz - wx * sin_el / cos_el * cos_az - wy * sin_el / cos_el * sin_az
            # 等价于：az_rate = wz - (wx * cos_az + wy * sin_az) * np.tan(el)
            
            # 方法B：简化公式（仅水平转动，忽略俯仰对方位的耦合）
            # 如果你的吊舱角速度已包含补偿，可用此简化版
            # az_rate = wz / cos_el  # 仅当 wx=wy=0 时正确，一般不推荐
        else:
            # 天顶附近：方位失去定义，显示绕Z轴转速
            az_rate = wz
            rospy.logwarn_throttle(2.0, "接近天顶，方位角速度无意义")
        
        return az_rate, el_rate


    def calc_lambda_los_from_vector(self):
        """
        从ENU视线方向向量计算
        lambda_los: 视线方位角 (rad)，目标相对于速度矢量的夹角，正前方为0，
                        左侧为正，右侧为负，范围[-pi, pi]
        """
        current_velocity = np.array([self.linear_velocity.x, self.linear_velocity.y, self.linear_velocity.z])
        # 视线方位角（北为0，东为正）
        psi_los = np.arctan2(self.pod_angle_in_map[0], self.pod_angle_in_map[1])  # atan2(东, 北)
        
        # 速度方位角
        v_bearing = np.arctan2(current_velocity[0], current_velocity[1])
        
        # 相对角度
        lambda_los = psi_los - v_bearing
        lambda_los = np.arctan2(np.sin(lambda_los), np.cos(lambda_los))
        
        return lambda_los

    # 你的现有函数（保持不变）
    def predict_omega_fixedwing(self, omega_now, roll_cmd, current_velocity,
                            pre_time=1.0, T_roll=0.8, g=9.81):
        roll_current = self.euler_angles[0]
        v_speed = np.linalg.norm(current_velocity[:2])
        
        if v_speed < 1.0:
            return omega_now
        
        # 预测滚转角
        roll_pred = roll_current + (roll_cmd - roll_current) * (1.0 - np.exp(-pre_time / T_roll))
        
        # 当前与预测转弯率
        turn_rate_now = g * np.tan(roll_current) / v_speed if abs(roll_current) > 0.01 else 0.0
        turn_rate_pred = g * np.tan(roll_pred) / v_speed if abs(roll_pred) > 0.01 else 0.0
        
        # 视线角速度预测
        lambda_los = self.calc_lambda_los_from_vector()
        delta_turn_rate = turn_rate_pred - turn_rate_now
        omega_pred = omega_now + delta_turn_rate * np.cos(lambda_los)
        
        return omega_pred  # 只返回预测值

    def pod_velocity_cb(self, msg: Vector3Stamped):
        """
        msg.vector.x: az 右正
        msg.vector.y: el 上正
        """
        t = msg.header.stamp.to_sec()
        if PREFORE_GUIDANCE:
            self.omega_los_map = np.radians(np.array([msg.vector.x, msg.vector.y]))
        else:
            self.omega_los_map_future = np.radians(np.array([msg.vector.x, msg.vector.y]))


    def pod_angle_in_map_angle_cb(self, msg: Vector3Stamped):
        """
        self.pod_angle_in_map: ENU坐标系下的视线向量 [x_east, y_north, z_up]
        -暂时弃用-self.pod_omega: 角速度 [az_rate, el_rate] rad/s
        """
        angle_new = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.pod_angle_in_map = angle_new

    
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

    def pn_guidance_predict(self, omega_los_map):
        """
        用丢失瞬间的相对运动状态，外推当前 LOS 并执行 PN
        相当于‘零效脱靶量’预测，再 PN
        """
        # 外推时间
        # t_go = max((rospy.Time.now() - self.memory_start_t).to_sec(), 0.0)

        # # 丢失瞬间状态
        # R0 = seeker_state_6d[:3]
        # V0 = seeker_state_6d[3:]
        # # 匀速外推（目标直线假设）
        # R_pred = R0 + V0 * t_go
        # V_pred = V0
        # # 组合成预测 seeker 状态
        # seeker_pred = np.concatenate((R_pred, V_pred))
        # # 标准 PN

        return self.pn_guidance(omega_los_map)

    def pn_guidance(self, omega_los_map):
        """
        使用比例导引律提供指令加速度
        :param seeker_state: 当前导引头状态
        :return: 惯性坐标系下的指令加速度
        """
        # rospy.logwarn("PN")
        #NXY = 4
        #NZ = 4
        r_dead = RMAG_TO_STRIKE
        Vc_max = 800.0

        # ========== 1. 获取 LOS_unit（ENU） ==========
        if not hasattr(self, 'pod_angle_in_map') or self.pod_angle_in_map is None:
            rospy.logerr_throttle(2.0, "PN: pod_angle_in_map not available")
            return np.zeros(3)
        los_map = self.pod_angle_in_map
        x, y, z = los_map  # x:东, y:北, z:上
        
        # 水平投影模长
        h_norm = np.sqrt(x**2 + y**2)
        
        # 方位角 az：从东(X)向北(Y)，逆时针为正
        # 范围 [-π, π]，0°指向正东，90°指向正北，-90°指向正南
        az_rad = np.arctan2(y, x)
        
        # 俯仰角 el：从水平面向上为正
        # 范围 [-π/2, π/2]，0°水平，90°朝天，-90°朝地
        if h_norm > 1e-6:
            el_rad = np.arctan2(z, h_norm)
        else:
            # 万向节锁：垂直朝天或朝地，俯仰角定义为 ±90°，方位角无意义（设为0或保持上周期值）
            el_rad = np.pi/2 if z > 0 else -np.pi/2
            az_rad = 0.0  # 或保持上一个有效值

        # 归一化得到单位视线向量
        norm = np.linalg.norm(los_map)
        if norm < 1e-6:
            rospy.logwarn_throttle(2.0, "PN:Wait for pod_angle_in_map")
            return np.zeros(3)
        LOS_unit = los_map / norm 
        

        # ========== 2. 获取 ENU 角速度矢量 Omega ==========
        waz = -omega_los_map[0]
        wel = -omega_los_map[1]
        Omega = np.array([
            -wel * np.sin(az_rad),   # 绕东向轴分量
            wel * np.cos(az_rad),    # 绕北向轴分量
            waz                      # NI
        ])

        # ========== 3. PN 核心计算 ==========
        Vc = self.est_Vc
        Vc = np.clip(Vc, -Vc_max, Vc_max)

        az_rate, el_rate = self.extract_spherical_rates(Omega, az_rad, el_rad)
        # rospy.logwarn_throttle(1.0, Omega)
        
        
        az_deg = np.degrees(az_rad)
        el_deg = np.degrees(el_rad)
        #rospy.loginfo_throttle(2.0, f"LOS Angles -> Az: {az_deg:.2f}°, El: {el_deg:.2f}°")
        # rospy.loginfo_throttle(2.0, f"los_map: {los_map[0]:.2f}, {los_map[1]:.2f}, {los_map[2]:.2f}")

        # 距离加权：self.r_mag 越小，N 越小，即增益退避方法
        distance = 10
        gain = np.tanh(np.maximum(distance, 1e-3) / (15.0 * r_dead)) 
        Nxy_eff = 3 # NXY * np.clip(gain, 0.2, 1.0)
        Nz_eff = 3 # NZ * np.clip(gain, 0.4, 1.0)

        a_base = Vc * np.cross(Omega, LOS_unit)
        # 水平与垂直分量
        a_xy = Nxy_eff * a_base[:2]
        a_z  = Nz_eff * a_base[2]
        a_cmd = np.array([a_xy[0], a_xy[1], a_z])

        ayaw_rad = np.arctan2(a_xy[0], a_xy[1])
        ayaw_deg = np.degrees(ayaw_rad)  
        
        
        #rospy.logwarn_throttle(2.0,f"[PN Guidance] Vc:{Vc:.2f}")
        rospy.logwarn_throttle(2.0, 
             f"[Omega ENU] Omega_Az:{waz:.2f}, Omega_El:{wel:.2f} deg/s"
        )
        # rospy.logwarn_throttle(2.0, f"[PN Guidance] Omega:{Omega[0]:.2f}, {Omega[1]:.2f}, {Omega[2]:.2f}")
        # rospy.logwarn_throttle(2.0, f"[PN Guidance] Com_a:[{a_cmd[0]:.2f}, {a_cmd[1]:.2f}, {a_cmd[2]:.2f}")
        rospy.loginfo_throttle(
                2.0 ,
                f"AYaw_Command: [{ayaw_deg:.2f} deg/(s*s)，{a_xy[0]:.2f}, {a_xy[1]:.2f}, {a_z:.2f}]"
            )
        return a_cmd

    def los_guidance_no_ranges(self, az_error, el_error, dt, k_p=2.0, k_d=0.5):
        """
        az_error: 大地坐标 方位脱靶量（弧度）
        el_error: 大地坐标 俯仰脱靶量（弧度）
        """
        # PD控制跟踪角度误差归零
        omega_az = k_p * az_error + k_d * (az_error - self.last_az_error) / dt
        omega_el = k_p * el_error + k_d * (el_error - self.last_el_error) / dt
        
        return omega_az, omega_el

    def get_commanded_accel(self, omega_los_map):
        """
        统一封装：丢目标判别 + 记忆制导 / 正常 PN 选择
        seeker_now : 当前量测（EKF 分支用 new_seeker_est，其余分支用 seeker_state）
        返回 -> commanded_accel, 是否强制 break
        """
        # ---- 丢目标判别 ----
        # if self.target_in_img:
        #     self.ready_to_memory = True
        #     self.memory_active = False
        #     #self.memory_active = False
        
        # # ---- 首次丢失 → 启动记忆制导 ----
        # if MEMORY_GUIDANCE and self.target_lost and (not self.memory_active):
        #     self.memory_active = True
        #     self.memory_start_t  = rospy.Time.now()
        #     # self.last_valid_seeker = seeker_now.copy()
        #     self.last_distance = self.distance

        #     # self.last_vm = np.linalg.norm(missile_state[3:6]) 
        #     print(f"[{rospy.Time.now().to_sec():.2f}s] 目标丢失 → 进入记忆制导")
        #     #print(f"self.target_lost:{self.target_lost}")
        #     #print(f"self.memory_active:{self.memory_active}")
        #     return self.pn_guidance_predict(), False
        
        # if MEMORY_GUIDANCE and self.memory_active and self.target_lost:
        #     if rospy.Time.now() - self.memory_start_t < rospy.Duration(MEMORY_TIMEOUT):
        #         print("记忆制导中", (rospy.Time.now() - self.memory_start_t).to_sec())
        #         #print(f"self.target_lost:{self.target_lost}")
        #         #print(f"self.memory_active:{self.memory_active}")
        #         return self.pn_guidance_predict(), False
        #     else:
        #         # 超时
        #         self.memory_active = False
        #         #print(f"self.target_lost:{self.target_lost}")
        #         #print(f"self.memory_active:{self.memory_active}")
        #         print(f"[{rospy.Time.now().to_sec():.2f}s] 记忆制导超时，放弃捕获")
        #         # print("--seeker_state2--", seeker_state)
        #         self.ready_to_memory = False
        #         return np.zeros(3), True
        
        # #print(f"self.target_lost:{self.target_lost}")
        # #rospy.loginfo_throttle(5, f"self.memory_active:{self.memory_active}")
        return self.pn_guidance(omega_los_map), False
    
    def compute_target_velocity_3d(self):
        """
        保持3D总速度大小不变，将速度方向转向 pod_angle_in_map 指定的方位
        ！！！以下部分已弃用，pod_angle_in_map为向量而非欧拉角。
        """
        # 当前速度向量
        v_current = np.array([
            self.linear_velocity.x,
            self.linear_velocity.y, 
            self.linear_velocity.z
        ])
        
        # 当前总速度大小（标量）
        speed_total = np.linalg.norm(v_current)
        
        if speed_total < 1e-6:
            # 速度接近零，返回零向量或保持当前（避免除零）
            return np.array([0.0, 0.0, 0.0])

        azimuth = self.pod_angle_in_map[0]
        elevation = self.pod_angle_in_map[1]
        
        cos_el = np.cos(elevation)
        sin_el = np.sin(elevation)
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)

        # 单位方向向量 [East, North, Up]
        unit_dir = np.array([
            cos_el * cos_az,  # X (东)
            cos_el * sin_az,  # Y (北)
            sin_el            # Z (上)
        ])
        
        # 保持原速率，乘向量转向
        target_vel = unit_dir * speed_total
        
        return target_vel
    
    def vector_to_attitude(self, ax_cmd, ay_cmd, az_cmd, current_velocity):
        """
        方案B：将ENU加速度指令转换为机体姿态（四元数）
        返回: (quaternion, thrust_cmd)
        """
        # 期望的总升力方向 = 重力 + 指令加速度
        L_enu = np.array([ax_cmd, ay_cmd, az_cmd + self.g])
        L_mag = np.linalg.norm(L_enu)
        
        if L_mag < 0.1:
            # 指令过小，保持平飞
            return (tf.transformations.quaternion_from_euler(0, 0, 0), 0.5)
        
        # 归一化升力方向（ENU系）
        L_hat = L_enu / L_mag
        
        # 将升力方向旋转到机体坐标系（减去当前航向）
        # 机体前向(X)在ENU中的方向：[cos(yaw), sin(yaw), 0]
        cos_y = np.cos(self.euler_angles[2])
        sin_y = np.sin(self.euler_angles[2])


        max_roll = np.radians(20)
        max_pitch = np.radians(20)

        a_body_x = ax_cmd * cos_y + ay_cmd * sin_y
        a_body_y = -ax_cmd * sin_y + ay_cmd * cos_y
        a_body_z = az_cmd

        a_body_y = np.clip(a_body_y, -self.g * np.tan(max_roll), self.g * np.tan(max_roll))
        a_body_z = np.clip(a_body_z, -0.3*self.g, 0.5*self.g)
        
        roll = np.arctan2(-a_body_y, self.g)  # 协调转弯公式
        pitch = np.arctan2(a_body_x, self.g + a_body_z)  # 简化能量关系
        
        # # ENU到机体的旋转（仅偏航）：X_body = X_enu*cos(yaw) + Y_enu*sin(yaw), Y_body = -X_enu*sin(yaw) + Y_enu*cos(yaw)
        # L_body_x = L_hat[0] * cos_y + L_hat[1] * sin_y
        # L_body_y = -L_hat[0] * sin_y + L_hat[1] * cos_y
        # L_body_z = L_hat[2]

        # # 计算Roll和Pitch（机体Z轴对齐L_hat）
        # # roll: 绕X轴旋转，使Z向Y倾斜（右滚产生向左加速度）
        # roll = np.arctan2(-L_body_y, L_body_z)
        
        # # pitch: 绕Y轴旋转，使Z向X倾斜（抬头产生向前加速度）
        # pitch = np.arctan2(L_body_x, L_body_z)
        
        rospy.loginfo_throttle(
                 1.0,
                 f"roll: {roll}, pitch: {pitch}"
            )
        # 限制姿态角（防止失速）
        roll = np.clip(roll, -max_roll, max_roll)
        pitch = np.clip(pitch, -max_pitch, max_pitch)

        airspeed = np.linalg.norm(current_velocity[:2])
        if airspeed > 5:
            yaw_rate_cmd = self.g * np.tan(roll) / airspeed
        else:
            yaw_rate_cmd = 0
        
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw_rate_cmd) #self.euler_angles[2]

        compen_pitch = np.radians(-pitch_deg)#-pitch_deg
        compen_yaw = np.radians(90-yaw_deg)#
        # 生成四元数（保持当前航向，因为固定翼航向由协调转弯决定）
        # 注意：这里yaw设为0是相对于当前航向，因为我们已经旋转到了机体坐标系
        # 实际上应该使用绝对航向
        q = tf.transformations.quaternion_from_euler(roll, compen_pitch, compen_yaw, axes='sxyz') ##roll, pitch, self.euler_angles[2], axes='sxyz'
        
        # 计算油门：补偿滚转/俯仰导致的升力需求增加
        # L_mag / g 表示需要提供的总加速度倍数（1为平飞）

        thrust_cmd = 0.60  # 根据你的飞机平飞油门调整（通常是0.5-0.7）
        if abs(roll) > np.radians(30):
            thrust_cmd = 0.75  
        thrust_cmd = np.clip(thrust_cmd, 0.0, 1.0)
        
        return q, thrust_cmd, roll, pitch

if __name__ == "__main__":
    rospy.init_node("guidance")
    rospy.loginfo("Guidance Raw Init")
    pn_law = Proportional_Navigation_Law()
    rospy.Subscriber("/mavros/local_position/odom", Odometry, pn_law.position_callback)
    rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, pn_law.velocity_callback)
    rospy.Subscriber("/pod/odom/target_odom_angle", Vector3Stamped, pn_law.pod_angle_in_map_angle_cb)
    rospy.Subscriber("/pod/odom/map_angle_velocity", Vector3Stamped, pn_law.pod_velocity_cb)
    command_vel_pub = rospy.Publisher("/navigation/command_velocity_local", TwistStamped, queue_size=5)
    command_acc_pub = rospy.Publisher("/navigation/command_accel_local", AccelStamped, queue_size=5)
    rospy.Subscriber("/pod/camera_focal", Float32, pn_law.camera_focal_cb)
    rospy.Subscriber("/pod/mode", Float32, pn_law.pod_mode_cb)
    pose_control_pub = rospy.Publisher("/navigation/attitude_cmd", PoseStamped, queue_size=5)
    rate = rospy.Rate(1/DT)
    # os.system('clear')
    while not rospy.is_shutdown():
        #print(pn_law.target_receive_flag, pn_law.drone_position_cb_flag)
        if pn_law.pod_mode == 2 and pn_law.drone_position_cb_flag == True and pn_law.drone_velocity_cb_flag == True:
            command_t0 = rospy.Time.now().to_sec()
            commanded_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            tau_xy = 1.19  # 无人机水平速度响应时间常数
            tau_z = 0.34 * 1.1  # 无人机垂直速度响应时间常数
            max_acc_xy = 80.0
            max_acc_z  = 35.0
            dt = DT
            current_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            current_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            simulation_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            simulation_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            last_com_accel = None
            break

    while not rospy.is_shutdown():
            time1 = rospy.Time.now().to_sec()
            time0 = rospy.Time.now().to_sec()
            command_vel = TwistStamped()

            # waz = pn_law.pod_omega[0]  # rad/s
            # wel = pn_law.pod_omega[1]
            # az = pn_law.pod_angle_in_map[0] # rad
            # el = pn_law.pod_angle_in_map[1]
            # rospy.logwarn_throttle(2.0, f"angle: {np.degrees(az)}, {np.degrees(el)}")
            # rospy.logwarn_throttle(2.0, f"wangle: {np.degrees(waz)}, {np.degrees(wel)}")
            # pn_law.target_lost = False
            # if not pn_law.target_in_img and pn_law.ready_to_memory:
            # # rospy.loginfo(np.linalg.norm(seeker_now[3:5]))
            #     pn_law.target_lost = True
            # else:
            #     pn_law.target_lost = False
            
            # if pn_law.target_in_img == False and pn_law.ready_to_memory == False:
            #     # print("------------------reset----------------------")
            #     command_t0 = rospy.Time.now().to_sec()
            #     commanded_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            #     dt = DT
            #     current_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            #     current_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            #     simulation_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            #     simulation_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])

            #     # 无图像中端，先通过角度控制
            #     if pn_law.pod_mode == 2:
            #         v_cmd = pn_law.compute_target_velocity_3d()
            #         command_vel.header.frame_id = "PN_Control"
            #         command_vel.twist.linear.x = v_cmd[0]
            #         command_vel.twist.linear.y = v_cmd[1] 
            #         # (PX4) “无人机指令坐标系“z轴向下 
            #         # (APM) “无人机指令坐标系“z轴up 
            #         command_vel.twist.linear.z = v_cmd[2]
            #         yaw_rad_0 = np.arctan2(v_cmd[0], v_cmd[1])
            #         yaw_deg_0 = np.degrees(yaw_rad_0)  
            #         if command_vel.header.frame_id == "PN_Control":
            #             rospy.loginfo_throttle(
            #                 2,
            #                 f"[Mode 0:Track angle] Yaw_Command: [{yaw_deg_0:.2f}]"
            #             )
            #         command_vel_pub.publish(command_vel)
            #     continue

                
            # 提前估计无人机状态
            # simulation_position = np.array([pn_law.position.x, pn_law.position.y, pn_law.position.z])
            # simulation_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            
            # simulation_position += simulation_velocity * dt
            # missile_state = pn_law.uav_state_to_missile_state(simulation_position, simulation_velocity)
            # target_state = pn_law.target_state_rec
            # print_missile_state(missile_state)
            # print_target_state(target_state) # 调试时两者只开其一
            
            # distance = np.linalg.norm(missile_state[:3]-target_state[:3])
            # seeker_state = pn_law.get_seeker_state(target_state, missile_state)
            # commanded_accel, force_break = pn_law.get_commanded_accel(seeker_state, missile_state)
            
            # 只能预测水平
            if PREFORE_GUIDANCE:
                commanded_accel_raw, force_break = pn_law.get_commanded_accel(pn_law.omega_los_map)
                q_raw, thrust_cmd_raw, roll_cmd_raw, pitch_cmd_raw = pn_law.vector_to_attitude(commanded_accel_raw[0], commanded_accel_raw[1], commanded_accel_raw[2], current_velocity)
                omega_pred = pn_law.predict_omega_fixedwing(pn_law.omega_los_map, 
                                                                roll_cmd_raw, 
                                                                current_velocity, 
                                                                pre_time=0.5, 
                                                                T_roll=0.6)
                # alpha = 0.2 + 0.6 * np.clip((pn_law.est_Vc - 10) / 30, 0, 1)
                rospy.loginfo_throttle(1.0, f"Pre_LosOmega: Az={np.degrees(omega_pred[0])}°"
                                  f"El={np.degrees(omega_pred[1]):.1f}°")
                alpha = 1
                omega_level_pre = (1 - alpha) * pn_law.omega_los_map + alpha * omega_pred
                pn_law.omega_los_map_future = np.array([omega_level_pre[0], pn_law.omega_los_map[1]])
            
            commanded_accel, force_break = pn_law.get_commanded_accel(pn_law.omega_los_map_future)
            q, thrust_cmd, roll_cmd, pitch_cmd = pn_law.vector_to_attitude(commanded_accel[0], commanded_accel[1], commanded_accel[2], current_velocity)

            current_velocity = np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z])
            commanded_velocity = current_velocity + commanded_accel * dt
            pn_law.commanded_accel_last = commanded_accel 
        
            msg_pose = PoseStamped()
            msg_pose.header.stamp = rospy.Time.now()
            msg_pose.header.frame_id = "base_link"
            
            # Orientation: 四元数 [x, y, z, w]
            msg_pose.pose.orientation.x = q[0]
            msg_pose.pose.orientation.y = q[1]
            msg_pose.pose.orientation.z = q[2]
            msg_pose.pose.orientation.w = q[3]
            
            # Position字段复用：x=油门(0-1)，y,z=预留
            msg_pose.pose.position.x = thrust_cmd
            
            pose_control_pub.publish(msg_pose)
            # rospy.loginfo_throttle(1.0, f"Cmd Att: Row={np.degrees(tf.euler_from_quaternion(q)[0]):.1f}°, "
            #                         f"Pitch={np.degrees(tf.euler_from_quaternion(q)[1]):.1f}°, "
            #                         f"Thrust={thrust_cmd:.2f}")

            if pn_law.pod_mode == 2:
                command_vel.header.frame_id = "PN_Control"
            else:
                command_vel.header.frame_id = "Cruise_Control"

            # print("com_acc",commanded_accel)
            #print("simulation",simulation_velocity)
            #print("current",np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z]))
            #print("self.position", pn_law.position)
            #print("missile_state:", np.array([missile_state[0], missile_state[1], missile_state[2],missile_state[3], missile_state[4], missile_state[5]]))
            #print("target_pos:", np.array([target_state[0], target_state[1], target_state[2],target_state[3], target_state[4], target_state[5]]))
            #print("[PN Guidance] Distance:",distance)
            # print("[PN Guidance] commanded_accel:",commanded_accel)
            #rospy.loginfo_throttle(5, f"[PN Guidance] Distance: {pn_law.distance:.2f}m")
            
            # print("SEEK_pos:", np.array([seeker_state[0], seeker_state[1], seeker_state[2]]))
            # print("current_pos:", np.array([target_state[0], target_state[1], target_state[2]]))
            # print("SEEK_V:", np.array([seeker_state[3], seeker_state[4], seeker_state[5]]))
            # print("current",np.array([pn_law.linear_velocity.x, pn_law.linear_velocity.y, pn_law.linear_velocity.z]))
            # print("simulation",simulation_velocity)
            #print("——————————————————")

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
            # (PX4) “无人机指令坐标系“z轴向下 
            # (APM) up 
            command_vel.twist.linear.z = commanded_velocity[2]
            rospy.loginfo_throttle(
                 1.0,
                 "current_vel: [% .2f, % .2f, % .2f]" % tuple(current_velocity)
            )
            rospy.loginfo_throttle(
                 1.0,
                 "command_vel: [% .2f, % .2f, % .2f]" % tuple(commanded_velocity)
            )
            
            yaw_rad_real = np.arctan2(pn_law.linear_velocity.x, pn_law.linear_velocity.y)
            yaw_deg_real = np.degrees(yaw_rad_real)
            yaw_rad = np.arctan2(commanded_velocity[0], commanded_velocity[1])
            yaw_deg = np.degrees(yaw_rad)  
            if pn_law.pod_angle_in_map_header == 1 and command_vel.header.frame_id == "PN_Control":
                rospy.loginfo_throttle(
                    2,
                    f"[Mode 1:Track Motor(Aim)] Yaw_Command: [{yaw_deg:.2f}] Yaw_Current: [{yaw_deg_real:.2f}]"
                )
            # elif pn_law.pod_angle_in_map_header == 2 and command_vel.header.frame_id == "PN_Control":
            #     rospy.loginfo_throttle(
            #         2,
            #         f"[Mode 2:Visual Cal] Yaw_Command: [{yaw_deg:.2f}]"
            #     )
            command_vel_pub.publish(command_vel)

            command_acc = AccelStamped()
            command_acc.accel.linear.x = commanded_accel[0]
            command_acc.accel.linear.y = commanded_accel[1]
            command_acc.accel.linear.z = commanded_accel[2]
            command_acc_pub.publish(command_acc)
            
            # print(force_break)
            if force_break:
                continue

            rate.sleep()