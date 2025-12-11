#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PointStamped
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

# ----------------- 可配置参数 -----------------
IMAGE_HEIGHT = 1080
IMAGE_WIGTH = 1920
# ---------------------------------------------

class Command:
    # ----------------- 可配置参数 -----------------
    TARGET_IP   = "192.168.144.160"
    TARGET_PORT = 10000
    LOCAL_PORT  = 10000

    # 默认命令
    CMD_HEX_STR = (
        "FB 2C 71 00 00 00 00 94 11 28 23 BC 34 18 04 0B "
        "0A 1D 28 03 AA D8 E4 42 64 A1 F3 41 02 0A 00 64 "
        "00 D0 07 20 4E 00 00 00 00 00 2A F0"
    )

    # 停止命令（用户给定）
    STOP_HEX_STR = (
        "FB 2C 71 00 00 00 00 94 11 28 23 BC 34 18 04 0B "
        "0A 1D 28 03 AA D8 E4 42 64 A1 F3 41 02 0A 00 64 "
        "00 D0 07 20 4E 00 00 00 00 00 2A F0"
    )

    T_pod_support = [0, 0, 0.13]
    T_support_drone =[0, 0, 0.03]
    R_support_drone = np.eye(3)
    
    # ---------------------------------------------

    def __init__(self):
        rospy.Subscriber("/detection/ultralytics/img_link_target_pub", PointStamped, self.target_callback)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.position_callback)
        self.odom_link_target_point_pub = rospy.Publisher("/detection/camera_link/target_position", PointStamped, queue_size=5)
        
        self._cmd_bytes  = bytes.fromhex(self.CMD_HEX_STR.replace(" ", ""))
        self._stop_bytes = bytes.fromhex(self.STOP_HEX_STR.replace(" ", ""))
        self._sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self.LOCAL_PORT))
        self._running    = False
        self._recv_thread = None
        self.aim_flag = False
        self.laser_i = 0
        self.image_in = False

        self.prev_time = time.time()
        self.x_integral = 0.0
        self.y_integral = 0.0
        self.x_prev_error = 0.0
        self.y_prev_error = 0.0

        self.detect_receive_flag = False
        self.target_px = IMAGE_WIGTH / 2
        self.target_py = IMAGE_HEIGHT / 2
    
    def target_callback(self, msg:PointStamped):
        self.detect_receive_flag = True
        if msg.header.frame_id == "Image_Frame_AIM":
            self.aim_flag = True
            self.image_in = True
        elif msg.header.frame_id == "None":
            self.image_in = True
        else:
            self.aim_flag = False
            self.image_in = False
        self.target_px = msg.point.x
        self.target_py = msg.point.y

    def position_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position
        orientation = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        self.euler_angles = tf.transformations.euler_from_quaternion(orientation, 'sxyz')
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular

    # ------------ 对外接口 ------------
    def start(self):
        """启动接收线程 + 发送循环（阻塞，直到 Ctrl-C）"""
        self._running = True
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()
        self._send_loop()

    def shutdown_hook(self, sig, frame):
        self.stop()
        sys.exit(0)

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.sendto(self._stop_bytes, (self.TARGET_IP, self.TARGET_PORT))
            except Exception as e:
                print("[stop] 发送停止报文失败:", e)
            finally:
                self._sock.close()

    # ------------ 内部实现 ------------
    def pid_control(self, x_error, y_error):
        """
        PID控制器，用于根据x和y方向的误差计算控制量。

        参数:
        x_error : float
            x方向的误差
        y_error : float
            y方向的误差

        返回:
        angle_vx_cmd : float
            x方向的控制量
        angle_vy_cmd : float
            y方向的控制量
        """
        # 硬编码PID参数
        x_kp, x_ki, x_kd, x_ol, x_sp = 1, 0.006, 0.08, 30.0, 0.0
        y_kp, y_ki, y_kd, y_ol, y_sp = 1, 0.006, 0.08, 30.0, 0.0

        current_time = time.time()
        dt = current_time - self.prev_time

        # 防止除零
        if dt <= 0:
            dt = 1e-6

        # X方向PID计算
        self.x_integral += x_error * dt
        x_derivative = (x_error - self.x_prev_error) / dt
        x_output = x_kp * x_error + x_ki * self.x_integral + x_kd * x_derivative
        angle_vx_cmd = x_sp + x_output
        if angle_vx_cmd > x_ol:
            angle_vx_cmd = x_ol
        elif angle_vx_cmd < -x_ol:
            angle_vx_cmd = -x_ol

        # Y方向PID计算
        self.y_integral += y_error * dt
        y_derivative = (y_error - self.y_prev_error) / dt
        y_output = y_kp * y_error + y_ki * self.y_integral + y_kd * y_derivative
        angle_vy_cmd = y_sp + y_output
        if angle_vy_cmd > y_ol:
            angle_vy_cmd = y_ol
        elif angle_vy_cmd < -y_ol:
            angle_vy_cmd = -y_ol

        # 更新变量
        self.prev_time = current_time
        self.x_prev_error = x_error
        self.y_prev_error = y_error

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
    
    def xor_checksum(self, data: bytes):
        """对 data 内所有字节做 XOR，返回低 8 位"""
        chk = 0
        for b in data:
            chk ^= b
        return chk & 0xFF
    
    def _receive_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(1024)
                region_receive = data[2:62]
                hex_line = data.hex(' ').upper()
                if len(data) == 64 and data[62] == self.xor_checksum(region_receive): 
                    target_z = int.from_bytes(data[27:29], 'little', signed=False)
                    if(self.image_in):
                        
                        if(target_z != 0):
                            # 解算K, 姿态角
                            f_x = int.from_bytes(data[25:27], 'little', signed=False)
                            f_y = f_x
                            support_az_raw = int.from_bytes(data[9:11], 'little', signed=True)
                            support_el_raw = int.from_bytes(data[11:13], 'little', signed=True)
                            support_rl_raw = int.from_bytes(data[13:15], 'little', signed=True)
                            support_az, support_el, support_rl = support_az_raw * 0.01, support_el_raw * 0.01, support_rl_raw * 0.01
                            support_att_str = f"舱座方位角={support_az:7.2f}° 舱座俯仰角={support_el:7.2f}° 舱座横滚角={support_rl:7.2f}°"
                            az_raw = int.from_bytes(data[41:43], 'little', signed=True)
                            el_raw = int.from_bytes(data[43:45], 'little', signed=True)
                            rl_raw = int.from_bytes(data[45:47], 'little', signed=True)
                            az, el, rl = az_raw * 0.01, el_raw * 0.01, rl_raw * 0.01
                            att_str = f"吊舱方位角={az:7.2f}° 吊舱俯仰角={el:7.2f}° 吊舱横滚角={rl:7.2f}°"

                            c_x = IMAGE_WIGTH / 2
                            c_y = IMAGE_HEIGHT / 2
                            # 构造 K 矩阵
                            K = np.array([
                                [f_x, 0, c_x],
                                [0, f_y, c_y],
                                [0, 0, 1]
                            ])

                            pixel_homogeneous = np.array([self.target_px, self.target_px, 1])
                            normalized_coords = np.linalg.inv(K).dot(pixel_homogeneous)
                            target_x = normalized_coords[0] * target_z
                            target_y = normalized_coords[1] * target_z
                            target_pod = np.array([target_z, target_x, target_y])

                            # 吊舱坐标系 --> 舱座坐标系
                            R_support = self.euler_to_rotation_matrix(support_az, support_el, support_rl)
                            R_pod = self.euler_to_rotation_matrix(az, el, rl)
                            R_pod_support = R_support.T @ R_pod
                            target_support = R_pod_support @ target_pod + self.T_pod_support
                            print(f"[recv] 来自 {addr} 的反馈：{hex_line} | \n {att_str} \n {support_att_str}")

                            # 舱座坐标系 --> 机体坐标系
                            target_drone = self.R_support_drone.dot(target_support) + self.T_support_drone
                            
                            # 无人机坐标 --> 世界坐标系
                            self.T_drone_map = np.array([self.position.x,
                                                    self.position.y,
                                                    self.position.z])   
                            self.R_drone_map = tf.transformations.euler_matrix(*self.euler_angles, axes='sxyz')[:3, :3]
                            target_position = self.R_drone_map.dot(target_drone) + self.T_drone_map

                            target_position_pub = PointStamped()
                            target_position_pub.header.stamp = rospy.Time.now()
                            target_position_pub.header.frame_id = "odom_link"
                            target_position_pub.point.x = target_position[0]
                            target_position_pub.point.y = target_position[1]
                            target_position_pub.point.z = target_position[2]
                            self.odom_link_target_point_pub.publish(target_position_pub)
                else:
                    att_str = "数据长度无法解析"

            except Exception as e:
                if self._running:
                    print("[recv] 异常：", e)

    def _send_loop(self):
        try:
            while self._running:
                if(self.detect_receive_flag):
                    template = bytes.fromhex(self.CMD_HEX_STR.replace(" ", ""))
                    x_error = 0.01 * (self.target_px - IMAGE_WIGTH/2)
                    y_error = - 0.01 * (self.target_py - IMAGE_HEIGHT/2)
                    angle_vx_cmd, angle_vy_cmd = self.pid_control(x_error, y_error)
                    self.pod_velocity = math.sqrt(angle_vx_cmd**2 + angle_vy_cmd**2)
                    self.prev_time = time.time()
                    raw = bytearray(template)
                    if(self.aim_flag and self.pod_velocity < 1 and self.laser_i < 3):
                        # self.laser_i < 3 为测距次数
                        print("LASING")
                        raw[2] = 0x3D
                        raw[37] = 0x00
                        raw[38:40] = struct.pack('<h', 0)
                        raw[40:42] = struct.pack('<h', 0)
                        self.laser_i = self.laser_i + 1
                    else:
                        raw[2] = 0x00
                        raw[37] = 0x70
                        raw[38:40] = struct.pack('<h', int(angle_vx_cmd))
                        raw[40:42] = struct.pack('<h', int(angle_vy_cmd)) 
                        
                        print("angle_v:", angle_vx_cmd, angle_vy_cmd)
                    region_send = raw[2:42]
                    raw[42] = self.xor_checksum(region_send)
                    self._cmd_bytes = bytes(raw)
                    if self.image_in == False:
                        self._cmd_bytes  = bytes.fromhex(self.CMD_HEX_STR.replace(" ", ""))
                self._sock.sendto(self._cmd_bytes, (self.TARGET_IP, self.TARGET_PORT))
                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\n 用户中断")


if __name__ == "__main__":
    rospy.init_node("command")
    rospy.loginfo("command init")
    
    cmd = Command()
    signal.signal(signal.SIGINT, cmd.shutdown_hook)
    
    cmd.start()
    
    

