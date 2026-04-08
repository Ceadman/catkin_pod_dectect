#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PointStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import socket
import threading
import time
import numpy as np
import tf
import struct
import math
import signal
import sys
from image_detect import UltralyticsROS
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from image_detect import IMAGE_HEIGHT, IMAGE_WIGTH, REFER_TARGET_BOX_PIX_AREA
from filter2d import MedianThenLowPass2D
from config import fs, fc

DEBUG_RECV = False
COM_CONTROL = True

# ----------------- 可配置参数 -----------------
T_SUPPORT_IN_DRONE = np.array([0.3, 0, -0.3])
R_SUPPORT_TO_DRONE = np.eye(3)

CAM_FX = 16 / (6.5 / 1920)

MAX_RECV_BUF = 8192  
# ---------------------------------------------

class Command:
    # ----------------- 可配置参数 -----------------
    TARGET_IP   = "192.168.144.119"
    TARGET_PORT = 2000

    # 默认命令
    CMD_HEX_STR = (
        "55 AA 00 01 01 05 10 A2 00 00 00 B9 F0"
    )
    # END TRACK
    END_TRACK_HEX_STR = (
        "55 AA 00 03 02 09 80 00 00 00 00 00 00 00 00 8E F0"
    )

    # 停止命令
    STOP_HEX_STR = (
        "55 AA 00 01 01 05 10 A2 00 00 00 B9 F0"
    )

    T_pod_in_support = [0, 0, -0.03]
    T_support_in_drone = T_SUPPORT_IN_DRONE
    R_support_to_drone = R_SUPPORT_TO_DRONE
    
    # ---------------------------------------------

    def __init__(self):
        rospy.Subscriber("/detection/ultralytics/target_point_pub", PointStamped, self.target_point_cb)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.position_callback)
        rospy.Subscriber('/detection/ultralytics/target_pixel_size', Float32, self.target_pixel_size_cb)
        rospy.Subscriber("/navigation/command_velocity_local", TwistStamped, self.cmd_vel_local_cb)
        self.target_map_angle_pub = rospy.Publisher("/pod/odom/target_odom_angle", Vector3Stamped, queue_size=5)
        self.map_pod_angle_velocity_pub = rospy.Publisher("/pod/odom/map_angle_velocity", Vector3Stamped, queue_size=5)
        # self.odom_link_target_point_pub = rospy.Publisher("/detection/odom/target_point", PointStamped, queue_size=5)
        self.detection_distance = rospy.Publisher("/detection/distance", Float32, queue_size=3)
        self.detected_img_pub = rospy.Publisher("/detection/image_detected", Image, queue_size=3)
        self.pod_f_pub = rospy.Publisher("/pod/camera_focal", Float32, queue_size=3)
        self.pod_mode_pub = rospy.Publisher("/pod/mode", Float32, queue_size=3)
        
        self._cmd_bytes  = bytes.fromhex(self.CMD_HEX_STR.replace(" ", ""))
        self._stop_bytes = bytes.fromhex(self.STOP_HEX_STR.replace(" ", ""))
        self._sock = None
        self._target_addr = (self.TARGET_IP, self.TARGET_PORT)
        self._connect_with_retry()    
        self.filter = MedianThenLowPass2D(fs, fc)
        self._running    = False
        self._recv_thread = None
        self.laser_i = 0
        self.image_in = False
        self.orientation = None
        self._last_cmd_bytes = None

        self.count_start_time = None
        self._switch_timer = False

        self.prev_time = time.time()
        self.x_integral = 0.0
        self.y_integral = 0.0
        self.x_prev_error = 0.0
        self.y_prev_error = 0.0
        self.target_pixel_size = 0.0

        self.detect_receive_flag = False
        self.drone_position_cb_flag = False
        self.bridge = CvBridge()
        self._reconnect_pause = 2.0
        self.x_prev_meas = 0.0
        self.y_prev_meas = 0.0
        self.x_integral  = 0.0
        self.y_integral  = 0.0
        self.angle_vx_last = 0.0
        self.angle_vy_last = 0.0
        self.prev_time   = time.time()
        self.home = True
        self.mode = 0
        self.linear_velocity = 0
        self.angular_velocity = 0

        self.enable_kff_tune = True  # 标完记得改 False
        self.tune_buffer = [] 
        self.tune_max_len = 50 * 10  
    
    def cmd_vel_local_cb(self, msg: TwistStamped):
        if msg.header.frame_id == "PN_Control":
            self.home = False
        else:
            self.home = True 

    def position_callback(self, msg: Odometry):
        self.drone_position_cb_flag = True
        self.position = msg.pose.pose.position
        self.orientation = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(self.orientation, 'sxyz')
        self.euler_angles = (roll, pitch, yaw)
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular
        # rospy.logwarn(f"{np.degrees(msg.twist.twist.angular.x):.2f}, {np.degrees(msg.twist.twist.angular.y):.2f}, {np.degrees(msg.twist.twist.angular.z):.2f}")
        
    def _connect_with_retry(self, interval=1.0, timeout=30):
        """阻塞式重连，直到吊舱 TCP 端口打开"""
        t0 = time.time()
        while not rospy.is_shutdown():
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(2.0)
                self._sock.connect(self._target_addr)
                rospy.loginfo(f"吊舱 TCP 已连接 {self._target_addr}")
                self._sock.settimeout(1.0)      # 后续正常超时
                return
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                rospy.logwarn_throttle(1, f"吊舱未就绪，{interval}s 后重连... ({e})")
                if self._sock:
                    self._sock.close()
                    self._sock = None
                if time.time() - t0 > timeout:
                    rospy.logerr("TCP 重连超时，退出节点")
                    sys.exit(1)
                time.sleep(interval)
                
    def _reconnect(self):
        """阻塞直到重新连上吊舱"""
        while self._running:            # 节点没退出就持续重试
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.connect((self.TARGET_IP, self.TARGET_PORT))
                time.sleep(self._reconnect_pause)
                rospy.loginfo("吊舱 TCP 重连成功")
                return                  # 连上立即退出
            except Exception as e:
                rospy.logerr_throttle(2, f"重连失败: {e}，1 s 后重试…")
                self._sock = None
                time.sleep(1)
        rospy.logwarn("节点已停止，重连线程退出")

    def target_point_cb(self, msg: PointStamped):
        self.detect_receive_flag = True
        if msg.header.frame_id == "None":
            self.image_in = False
            if self._switch_timer == False:
                self.count_start_time = rospy.Time.now()
                self._switch_timer = True
            if rospy.Time.now() - self.count_start_time > rospy.Duration(1.5):
                self._switch_timer = False
        elif msg.header.frame_id == "Camera_Optical_Frame":
            self._switch_timer = False
            self.image_in = True
        self.target_px = msg.point.x
        self.target_py = msg.point.y

    def target_pixel_size_cb(self, msg: Float32):
        self.target_pixel_size = msg.data
        

    # ------------ 对外接口 ------------
    def start(self):
        """启动接收线程 + 发送循环（阻塞，直到 Ctrl-C）"""
        self._running = True
        # self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        # self._recv_thread.start()
        self._send_loop()

    def shutdown_hook(self, sig, frame):
        print("\nCtrl-C 捕获，正在退出…")
        self.stop()
        sys.exit(0)

    def stop(self):
        self._running = False
        # if self._recv_thread and self._recv_thread.is_alive():
        #     self._recv_thread.join(timeout=1.0)
        if self._sock:
            try:
                self._sock.sendall(self._stop_bytes)
                self._sock.shutdown(socket.SHUT_WR)
            except Exception:
                pass
            finally:
                self._sock.close()
                self._sock = None

    # ------------ 内部实现 ------------
    def reset_pid(self):
        self.x_integral = 0.0
        self.y_integral = 0.0
        self.x_prev_meas = 0.0
        self.y_prev_meas = 0.0
        self.angle_vx_last = 0.0
        self.angle_vy_last = 0.0
        self.prev_time = time.time() 

    def pid_control(self, x_error, y_error):
        """
        升级 PID：微分先行 + 速度前馈 + 条件积分抗饱和
        输入/输出接口与原函数完全一致
        """
        # ---------- 时间更新 ----------
        now = time.time()
        dt = now - self.prev_time
        if dt < 1e-3:
            return self.angle_vx_last, self.angle_vy_last
        if dt > 0.1:
            self.reset_pid()
            self.prev_time = now
            return self.angle_vx_last, self.angle_vy_last

        # ---------- 滤波 ----------
        # rospy.loginfo_throttle(0.5, f"error: {x_error}, {y_error}")
        # xy_err  = np.array([x_error, y_error])
        # xy_flt  = self.filter.update(xy_err)   # 你原来的滤波器
        # x_err   = xy_flt[0]
        # y_err   = xy_flt[1]
        x_err = x_error
        y_err = y_error

        # ---------- 参数 ----------
        kp, ki, kd   = 3.0, 0.006, 0.10
        # kp, ki, kd   = 0, 0, 0
        kff          = 0                   # 速度前馈初值，现场再微调
        max_int      = 50.0
        max_out      = 30.0                    # 对应吊舱最大角速度 30 °/s

        x_d = -kd * (x_err - self.x_prev_meas) / dt
        y_d = -kd * (y_err - self.y_prev_meas) / dt

        # 像素速度→前馈
        x_vel = (x_err - self.x_prev_meas) / dt
        y_vel = (y_err - self.y_prev_meas) / dt
        x_ff  = kff * x_vel
        y_ff  = kff * y_vel

        # 仅当“未饱和”且“误差较小”才积分, 积分阈值自适应
        x_int_thresh = 0.037 * IMAGE_WIGTH   # ≈ 70 px @1920
        y_int_thresh = 0.037 * IMAGE_HEIGHT
        x_int_allowed = abs(x_err) < x_int_thresh and abs(self.x_integral) < max_int
        y_int_allowed = abs(y_err) < y_int_thresh and abs(self.y_integral) < max_int

        x_sat = abs(self.angle_vx_last) >= 0.95 * max_out
        y_sat = abs(self.angle_vy_last) >= 0.95 * max_out
        if not x_sat and x_int_allowed:
            self.x_integral += x_err * dt
        if not y_sat and y_int_allowed:
            self.y_integral += y_err * dt

        # ---------- 总输出 ----------
        x_out = kp*x_err + ki*self.x_integral + x_d + x_ff
        y_out = kp*y_err + ki*self.y_integral + y_d + y_ff

        # 钳位
        angle_vx_cmd = np.clip(x_out, -max_out, max_out)
        angle_vy_cmd = np.clip(y_out, -max_out, max_out)

        # ---------- 更新记忆 ----------
        self.x_prev_meas = x_err
        self.y_prev_meas = y_err
        self.angle_vx_last = angle_vx_cmd
        self.angle_vx_last = angle_vy_cmd

        return angle_vx_cmd, angle_vy_cmd
    
    def euler_to_rotation_matrix(self, az, el, rl):
                    # 角度转弧度
                    az_rad = np.radians(az)
                    el_rad = np.radians(el)
                    rl_rad = np.radians(rl)

                    # 构造旋转矩阵（Z-Y-X顺序）
                    Rz = np.array([
                        [np.cos(az_rad), -np.sin(az_rad), 0],
                        [np.sin(az_rad),  np.cos(az_rad), 0],
                        [0,               0,              1]
                    ])

                    Ry = np.array([
                        [np.cos(el_rad),  0, np.sin(el_rad)],
                        [0,               1, 0],
                        [-np.sin(el_rad), 0, np.cos(el_rad)]
                    ])

                    Rx = np.array([
                        [1, 0,              0],
                        [0, np.cos(rl_rad), -np.sin(rl_rad)],
                        [0, np.sin(rl_rad),  np.cos(rl_rad)]
                    ])

                    return Rz @ Ry @ Rx
    
    def pixel_to_angle(self, dx_pix, dy_pix, K):
        """
        dx_pix, dy_pix: 相对于图像中心的脱靶量（像素），右/上为正
        K: [[fx, 0, cx], [0, fy, cy], [0,0,1]] 相机内参矩阵
        返回: (theta_az, theta_el) 弧度，分别对应方位（yaw）和俯仰（pitch）偏移
        """
        fx, fy = K[0,0], K[1,1]
        # 精确计算（光轴为Z，像平面为X-Y）
        theta_az = np.arctan2(dx_pix, fx)   # 水平方向角
        theta_el = np.arctan2(dy_pix, fy)   # 垂直方向角
        return theta_az, theta_el
    
    # def calc_target_point(self, K):
    #     target_z = REFER_TARGET_BOX_PIX_AREA ** 0.5 / self.target_pixel_size ** 0.5
        
    #     #rospy.loginfo_throttle(0.5, f"self.target_pixel_size, {self.target_pixel_size}")
    #     #rospy.loginfo_throttle(1.0, target_z)
    #     pixel_homogeneous = np.array([self.target_px, self.target_py, 1])
    #     normalized_coords = np.linalg.inv(K).dot(pixel_homogeneous)
    #     target_x = normalized_coords[0] * target_z
    #     target_y = normalized_coords[1] * target_z
    #     target_pod = np.array([target_z, target_x, target_y])
    #     # rospy.loginfo_throttle(0.5, f"target_pod, {target_pod}")

    #     # 吊舱坐标系 --> 舱座坐标系
    #     R_pod_to_support = self.euler_to_rotation_matrix(
    #         self.az_pod_support, self.el_pod_support, self.rl_pod_support
    #     ).T
    #     target_support = R_pod_to_support.dot(target_pod) + self.T_pod_in_support
    #     # print(f"[recv] 来自吊舱的反馈：{hex_line} | \n {att_str}")
    #     # rospy.loginfo_throttle(0.5, f"target_support: {target_support}")

    #     # 舱座坐标系 (FRD: X前 Y右 Z下) --> 机体坐标系 (FRD: X前 Y右 Z下)
    #     target_drone_frd  = self.R_support_to_drone.dot(target_support) + self.T_support_in_drone

    #     # 【KEY】机体 FRD --> 机体 FLU(FLU: X前, Y左, Z上)
    #     target_drone_flu = np.array([
    #         target_drone_frd[0],   # X: 前 -> 前
    #         -target_drone_frd[1],  # Y: 右 -> -左  
    #         -target_drone_frd[2]   # Z: 下 -> -上
    #     ])
        
    #     # rospy.logwarn_throttle(2, f"Target_drone_flu: {target_drone_flu}")
    #     # 无人机坐标 --> 世界坐标系
    #     T_drone_in_map = np.array([self.position.x,
    #                             self.position.y,
    #                             self.position.z])   
    #     R_drone_to_map = tf.transformations.quaternion_matrix(self.orientation)[:3, :3].T
    #     target_position = R_drone_to_map.dot(target_drone_flu) + T_drone_in_map

    #     # rospy.loginfo_throttle(2, f"Drone_in_Map_Position: {T_drone_in_map}")
    #     # rospy.loginfo_throttle(2, f"Drone_in_Map_Euler: {np.degrees(self.euler_angles)}")
        
    #     return target_position

    def calc_los_angles(self, pod_theta_az_deg, pod_theta_el_deg):
        """
        输入: 视线角（FRD坐标系定义）
            pod_theta_az_deg: 方位角，右偏为正（deg）
            pod_theta_el_deg: 俯仰角，上偏为正（deg）
        输出: 
            los_map：大地坐标系（ENU）下的视线角方向向量
        """

        az = np.radians(pod_theta_az_deg)
        el = np.radians(pod_theta_el_deg)
        
        # 吊舱 FRD 坐标系(X前, Y右, Z下)
        los_frd = np.array([
            np.cos(el) * np.cos(az),   # X: 前
            np.cos(el) * np.sin(az),   # Y: 右
            -np.sin(el)                # Z: 下
        ])
        
        # 机体 FLU 坐标系(X前, Y左, Z下)
        los_body = np.array([
            los_frd[0],     # X: 前 -> 前
            -los_frd[1],    # Y: 右 -> -左
            -los_frd[2]     # Z: 下 -> -上
        ])

        # los_body = np.array([
        #     np.cos(el) * np.cos(az),   # X: 前
        #     np.cos(el) * np.sin(az),   # Y: 右
        #     -np.sin(el)                 # Z: 下
        # ])
        
        # 机体(FLU:X前, Y左, Z上) -> 大地(ENU:X东, Y北, Z上)
        R_frd_to_enu = tf.transformations.quaternion_matrix(self.orientation)[:3, :3]
        
        los_map = R_frd_to_enu.dot(los_body)
         
        # # 大地坐标系(ENU: X东, Y北, Z上)
        # map_az = np.arctan2(los_map[1], los_map[0])  # 东为0，北为90，逆时针
        # map_el = np.arctan2(los_map[2], np.sqrt(los_map[0]**2 + los_map[1]**2))  # 向上为正
        
        return los_map 


    # def calc_los_rates(self, az_velocity, el_velocity, az_deg, el_deg):
    #     """
    #     输入:
    #         az_velocity: 吊舱方位电机角速度（顺时针为正, deg/s 或 rad/s）
    #         el_velocity: 吊舱俯仰电机角速度（向上为正, deg/s 或 rad/s）
    #         az: 当前方位角（deg，FRD定义：右偏为正，0=前）
    #         el: 当前俯仰角（deg，FRD定义：上偏为正，0=水平）
    #         self.angular_velocity: 无人机角速度（机体坐标系, rad/s）
    #         self.orientation: 无人机姿态四元数 [x, y, z, w]（机体系FRD->ENU）
    #     输出:
    #         omega_los_enu: ENU下角

    #     """
    #     az_rate = np.deg2rad(az_velocity)
    #     el_rate = np.deg2rad(el_velocity)
    #     az = np.radians(az_deg)
    #     el = np.radians(el_deg)

    #     # Pod_angle
    #     omega_pod_frd = np.array([
    #         -el_rate * np.sin(az),   # X: 抬头时绕X分量
    #         el_rate * np.cos(az),    # Y: 抬头时绕Y分量  
    #         az_rate                  # Z: 方位角速度（顺时针为正，因Z向下）
    #     ])

    #     # UAV角速度（FLU坐标系，已由IMU提供，单位rad/s）
    #     omega_body_body = np.array([
    #         self.angular_velocity.y,      # 绕Y轴（俯仰率，抬头为负）
    #         self.angular_velocity.x,      # 绕X轴（滚转率）
    #         self.angular_velocity.z       # 绕Z轴（偏航率，逆时针为正）
    #     ])

    #     omega_pod_flu = np.array([
    #         omega_pod_frd[0],   # X相同
    #         -omega_pod_frd[1],  # Y反向：右→左
    #         -omega_pod_frd[2]   # Z反向：下→上
    #     ])
            
    #     # 机体坐标系下合成
    #     # omega_los_body = omega_pod_flu + omega_body_body
        
    #     # 转到 ENU 坐标系（必须用 orientation）
        
    #     R_frd_to_enu = tf.transformations.quaternion_matrix(self.orientation)[:3, :3]
    #     # omega_los_enu = R_frd_to_enu @ omega_los_body

    #     omega_los_enu_1 = R_frd_to_enu @ omega_pod_flu 
    #     omega_los_enu_2 = R_frd_to_enu @ omega_body_body
    #     omega_los_enu = omega_los_enu_1  + omega_los_enu_2
    #     rospy.logwarn_throttle(1.0, f"pod_omega:{omega_los_enu_1[0]:.2f},{omega_los_enu_1[1]:.2f},{omega_los_enu_1[2]:.2f}")
    #     rospy.logwarn_throttle(1.0, f"body_omega:{omega_los_enu_2[0]:.2f},{omega_los_enu_2[1]:.2f},{omega_los_enu_2[2]:.2f}")
    #     rospy.logwarn_throttle(1.0, omega_los_enu)

    #     return omega_los_enu


    def _parse_and_publish(self, data):
        """ frame 保证 71 B 且校验通过 """
        # 解算K, 姿态角
        if data[69] != (sum(data[2:69]) & 0xFF):
            print(data[69], sum(data[2:69]))
            return False, data, f"校验失败：calc={sum(data[2:69]) & 0xFF} rx={data[69]}"
        
        f_x = CAM_FX
        f_y = f_x
        self.pod_f_pub.publish(Float32(f_x))
        SCALE_ANGLE = 360.0 / 65536.0

        c_x = IMAGE_WIGTH / 2
        c_y = IMAGE_HEIGHT / 2
        # 构造 K 矩阵
        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])

        az_raw = int.from_bytes(data[11:13], 'little', signed=True)
        el_raw = int.from_bytes(data[13:15], 'little', signed=True)
        rl_raw = 0.0
        self.az_pod_support, self.el_pod_support, self.rl_pod_support = az_raw * SCALE_ANGLE, el_raw * SCALE_ANGLE, rl_raw * SCALE_ANGLE
        att_str = f"吊舱方位角={self.az_pod_support:7.2f}° 吊舱俯仰角={self.el_pod_support:7.2f}° 吊舱横滚角={self.rl_pod_support:7.2f}°"
        # print(att_str)

        if self.orientation is None or len(self.orientation) != 4:
            rospy.logwarn_throttle(1.0, "orientation not ready")
            return True, data, ""
        # ========== 新增：检测飞控数据是否更新 ==========
        current_time = rospy.Time.now()
        
        # 初始化检查点（首次调用）
        if not hasattr(self, '_last_odom_check'):
            self._last_odom_check = {
                'time': current_time,
                'orientation': self.orientation.copy()
            }
        else:
            dt = (current_time - self._last_odom_check['time']).to_sec()
            
            # 每满1秒检查一次
            if dt >= 1.0:
                # 四个分量完全相等
                if np.array_equal(self.orientation, self._last_odom_check['orientation']):
                    rospy.logwarn_throttle(1.0, "飞控数据未更新")
                
                # 更新检查基准
                self._last_odom_check['time'] = current_time
                self._last_odom_check['orientation'] = self.orientation.copy()

        if self.drone_position_cb_flag == False:
            rospy.logwarn_throttle(1.0, "No Position Info")
            return True, data, ""
        
        # -------------------------目标脱靶量----------------------------
        error_pod_x = int.from_bytes(data[27:29], 'little', signed=True)
        error_pod_y = int.from_bytes(data[29:31], 'little', signed=True)
        # 像素脱靶 ->弧度脱靶（像素：右正上正 -> 角度：右偏为正，上偏为正）
        pod_theta_az_offset, pod_theta_el_offset = self.pixel_to_angle(error_pod_x, error_pod_y, K)
        # print(np.degrees(pod_theta_az_offset), np.degrees(pod_theta_el_offset))
        pod_theta_az = self.az_pod_support + np.degrees(pod_theta_az_offset)
        pod_theta_el = self.el_pod_support + np.degrees(pod_theta_el_offset)
        los_map = self.calc_los_angles(pod_theta_az, pod_theta_el) 
        target_map_angle = Vector3Stamped()
        target_map_angle.header.stamp = rospy.Time.now()
        # rospy.logwarn_throttle(0.1, f" {error_pod_x}{error_pod_y}")
        if abs(error_pod_x) < IMAGE_WIGTH/4 or abs(error_pod_y) < IMAGE_HEIGHT/4:
            target_map_angle.header.frame_id = "aim"
        else: 
            target_map_angle.header.frame_id = "out"
        target_map_angle.vector.x, target_map_angle.vector.y, target_map_angle.vector.z = los_map
        self.target_map_angle_pub.publish(target_map_angle)

        # -------------------------电机转速------------------------------
        az_velocity_raw = int.from_bytes(data[23:25], 'little', signed=True)
        el_velocity_raw = int.from_bytes(data[25:27], 'little', signed=True)
        az_velocity_map = az_velocity_raw * 0.01
        el_velocity_map = el_velocity_raw * 0.01
        # omega_los_map = self.calc_los_rates(az_velocity, el_velocity, pod_theta_az, pod_theta_el)
        map_pod_angle_velocity = Vector3Stamped()
        map_pod_angle_velocity.header.stamp = rospy.Time.now()
        map_pod_angle_velocity.header.frame_id = "odom"
        map_pod_angle_velocity.vector.x, map_pod_angle_velocity.vector.y = az_velocity_map, el_velocity_map
        self.map_pod_angle_velocity_pub.publish(map_pod_angle_velocity)
        
        # -------------------------目标点--------------------------------
        if self.target_pixel_size > 0:
            target_z = REFER_TARGET_BOX_PIX_AREA ** 0.5 / self.target_pixel_size ** 0.5
            self.detection_distance.publish(Float32(target_z))
            # rospy.logwarn_throttle(1, "Waiting for target_pixel_size input")
            # return False, data, f"no target_pixel_size"
        
        
        
        # target_position = self.calc_target_point(K) # z轴向下为正
        # target_position_pub = PointStamped()
        # target_position_pub.header.stamp = rospy.Time.now()
        # target_position_pub.header.frame_id = "odom"
        # target_position_pub.point.x = target_position[0]
        # target_position_pub.point.y = target_position[1]
        # target_position_pub.point.z = target_position[2]
        # target_position_pub.point.x = target_position[1]
        # target_position_pub.point.y = target_position[0]
        # target_position_pub.point.z = -target_position[2]
        # rospy.logwarn_throttle(2, f"Target_in_Map: {target_position_pub.point.x, target_position_pub.point.y, target_position_pub.point.z}")
        # self.odom_link_target_point_pub.publish(target_position_pub)

    def _recv_one_frame(self, timeout=0.2):
        """带自动调试的拆帧函数"""
        if not hasattr(self, '_recv_buf'):
            self._recv_buf = bytearray()

        while True:
            # ---------------- 先扫已有缓存 ----------------
            while len(self._recv_buf) >= 71:
                idx = 0
                # 找 55 AA
                while idx <= len(self._recv_buf) - 2:
                    if self._recv_buf[idx] == 0x55 and self._recv_buf[idx+1] == 0xAA:
                        break
                    idx += 1
                else:                       # 没找到帧头
                    if DEBUG_RECV:
                        rospy.loginfo_throttle(2.0, "[recv] no 55 AA header, drop %d B" % (len(self._recv_buf)-1))
                    self._recv_buf = self._recv_buf[-1:]
                    break

                if len(self._recv_buf) - idx < 71:        # 长度不够
                    break

                rsp = self._recv_buf[idx:idx+71]
                cs_calc = sum(rsp[2:69]) & 0xFF
                cs_expect = rsp[69]
                if cs_calc == cs_expect:                  # 校验 OK
                    self._recv_buf = self._recv_buf[idx+71:]
                    if DEBUG_RECV:
                        rospy.loginfo_throttle(1.0,"[recv] good frame  %s" % rsp.hex(' '))
                    return True, rsp, ""
                else:                                     # 校验失败
                    if DEBUG_RECV:
                        rospy.logwarn_throttle(2.0,
                            "[recv] checksum NG  calc=0x%02X expect=0x%02X\nframe=%s" %
                            (cs_calc, cs_expect, rsp.hex(' ')))
                    self._recv_buf = self._recv_buf[idx+1:]   # 只滑 1 字节继续找
                    continue

            # ---------------- 缓存不足，读 TCP ----------------
            try:
                chunk = self._sock.recv(1024)
                if not chunk:
                    if DEBUG_RECV:
                        rospy.logwarn_throttle(2.0, "[recv] peer closed")
                    return False, b'', 'peer closed'
                self._recv_buf.extend(chunk)

                if DEBUG_RECV:
                    rospy.loginfo_throttle(2.0,
                        "[recv] tcp read %d B  buf now %d B\nhead 16 B: %s" %
                        (len(chunk), len(self._recv_buf), self._recv_buf[:16].hex(' ')))

                if len(self._recv_buf) > MAX_RECV_BUF:
                    if DEBUG_RECV:
                        rospy.logerr("[recv] buffer overflow  dump 256 B: %s",
                                    self._recv_buf[:256].hex(' '))
                    self._recv_buf = bytearray()
                    return False, b'', "buffer overflow"

            except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError) as e:
                if DEBUG_RECV:
                    rospy.logwarn_throttle(2.0, "[recv] exception: %s" % e)
                return False, b'', str(e)
            
    def send_once_receive_once(self, payload: bytes, timeout=0.2, retries=3):
        """线程安全地发一次并等待回包，返回是否成功"""
        for _ in range(retries):
            if self._sock is None:
                self._reconnect()
            try:
                # self._sock.sendall(payload) # Auto Control
                # print("555")
                ok, rsp, reason = self._recv_one_frame(timeout)
                
                if ok:
                    # rospy.loginfo_throttle(5, f"接收字节位数 {len(rsp)}")
                    # rospy.loginfo_throttle(0.5, f"接收字节位数 {len(rsp)}: {rsp.hex(' ')}")
                    if rsp[7] == 0x00:
                        self.mode = 0
                    elif rsp[7] == 0x01:
                        self.mode = 1
                    elif rsp[7] == 0x02:
                        self.mode = 2
                    else:
                        rospy.logwarn_throttle(1, f"Unpredict Mode: {rsp[7]}")
                    self.pod_mode_pub.publish(Float32(self.mode))
                    self._parse_and_publish(rsp)
                    return True, rsp 
                # 失败分两条打
                if not rsp: 
                    rospy.logwarn_throttle(1, f"[recv] 未收到任何吊舱数据")
                else:
                    rospy.logwarn_throttle(1, f"[recv] {reason}")
            except Exception as e:
                self._sock = None
                rospy.logwarn_throttle(1, f"[recv] 异常：{e}")
        return False, b''
   
    
    def _send_loop(self):
        while self._running:
            if COM_CONTROL == True:
                header = bytearray([0x55, 0xAA, 0x00, 0x03, 0x02])
                # 0-255 循环
                header[2] = (getattr(self, 'frame_cnt', -1) + 1) & 0xFF
                long_cmd = bytearray([0x05, 0x85])
                cmddata = bytearray([0x00, 0x00, 0x00, 0x00])
                if self.image_in:
                    cmddata[3] = 0x01
                # cmddata.append(0x00)
                # extra = struct.pack('<h', int(angle_vy_cmd))   # 再来 2 字节
                # cmddata.extend(extra)  
                end =  bytearray([0x00, 0xF0])

                frame = header + long_cmd + cmddata + end
                checksum = sum(frame[2:-2]) & 0xFF
                frame[-2] = checksum
                self._cmd_bytes = bytes(frame)
                
                # rospy.loginfo_throttle(0.01, f"Mode: {self.mode}")
                #if self.mode == 1:
                #    self._cmd_bytes = bytes.fromhex(self.END_TRACK_HEX_STR.replace(" ", ""))
                #print(self.home)
                #if not self.image_in and self.home == True:
                #    self._cmd_bytes = bytes.fromhex(self.CMD_HEX_STR.replace(" ", ""))
                # if self._cmd_bytes == self._last_cmd_bytes:
                #     time.sleep(0.1)
                #     continue

                #rospy.loginfo_throttle(0.01, f"cmd_send: {self._cmd_bytes.hex(' ')}")
                ok, rsp = self.send_once_receive_once(self._cmd_bytes)
                if ok:
                    self._last_cmd_bytes = self._cmd_bytes
                    # rospy.loginfo_throttle(1, f"send {self._cmd_bytes.hex(' ')} | recv {rsp.hex(' ')}")
                else:
                    # rospy.logerr_throttle(1, "发送失败，准备重连")
                    self._last_cmd_bytes = None



if __name__ == "__main__":
    rospy.init_node("communication")
    rospy.loginfo("pod communication and detection init")
    
    cmd = Command()
    signal.signal(signal.SIGINT, cmd.shutdown_hook)
    
    cmd.start()