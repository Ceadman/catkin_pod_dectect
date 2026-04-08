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
from image_detect import UltralyticsROS
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32 
from image_detect import IMAGE_HEIGHT, IMAGE_WIGTH, REFER_TARGET_BOX_PIX_AREA
from filter2d import MedianThenLowPass2D
from config import fs, fc

DEBUG_RECV = False
COM_CONTROL = True
TRACK_MODE = 1  # 1:指点跟踪 2：开启识别转跟踪

# ----------------- 可配置参数 -----------------
T_SUPPORT_DRONE = np.array([-0.3, 0, 0.17])
R_SUPPORT_DRONE = np.eye(3)

CAM_FX = 16
PIXEL_SIZE = 2.9 * 10e-3  # mm
CAM_PIXEL_FX = CAM_FX / PIXEL_SIZE  # mm/mm

MAX_RECV_BUF = 8192  
# ---------------------------------------------

class Command:
    # ----------------- 可配置参数 -----------------
    TARGET_IP   = "192.168.144.119"
    TARGET_PORT = 2000

    # 起始命令(开启目标识别)
    POD_TARGET_DETECT_STR = (
        "EB 90 16 EB 90 07 00 00 00 00 00 00 00 00 00 00 00 00 82 04"
    )

    # 默认命令
    CMD_HEX_STR = (
        "EB 90 16 EB 90 2b 00 00 00 00 00 00 00 00 00 00 00 00 A6 4C"
    )
    # CMD_HEX_STR = (
    #     "EB 90 16 EB 90 26 00 00 30 F8 00 00 00 00 00 00 00 00 EB D6"
    # )
    
    # 开启识别转跟踪
    DETECT_TO_TRACK_STR = (
        "EB 90 16 EB 90 0F 00 00 00 00 00 00 00 00 00 00 00 00 8A 14"
    )
    # 结束跟踪
    END_TRACK_HEX_STR = (
        "EB 90 16 EB 90 0E 00 00 00 00 00 00 00 00 00 00 00 00 89 12"
    )

    # 停止命令
    STOP_HEX_STR = (
        "EB 90 16 EB 90 2b 00 00 00 00 00 00 00 00 00 00 00 00 A6 4C"
    )

    T_pod_support = [0, 0, 0.13]
    T_support_drone = T_SUPPORT_DRONE
    R_support_drone = R_SUPPORT_DRONE
    
    # ---------------------------------------------

    def __init__(self):
        rospy.Subscriber("/detection/ultralytics/target_point_pub", PointStamped, self.target_point_cb)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.position_callback)
        rospy.Subscriber('/detection/ultralytics/target_pixel_size', Float32, self.target_pixel_size_cb)
        self.odom_link_target_point_pub = rospy.Publisher("/detection/odom/target_point", PointStamped, queue_size=5)
        self.detected_img_pub = rospy.Publisher("/detection/image_detected", Image, queue_size=3)
        self.pod_f_pub = rospy.Publisher("/pod/camera_focal", Float32, queue_size=3)
        
        self._start_bytes = bytes.fromhex(self.POD_TARGET_DETECT_STR.replace(" ", ""))
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
        self._switch_timer = rospy.Time.now()

        self.prev_time = time.time()
        self.x_integral = 0.0
        self.y_integral = 0.0
        self.x_prev_error = 0.0
        self.y_prev_error = 0.0
        self.target_pixel_size = 0.0

        self.detect_receive_flag = False
        self.drone_position_cb_flag = False
        self.pod_prejudiced_flag = False
        self.bridge = CvBridge()
        self._reconnect_pause = 2.0
        self.x_prev_meas = 0.0
        self.y_prev_meas = 0.0
        self.x_integral  = 0.0
        self.y_integral  = 0.0
        self.angle_vx_last = 0.0
        self.angle_vy_last = 0.0
        self.prev_time   = time.time()
        self.mode = 1

        self.enable_kff_tune = True  # 标完记得改 False
        self.tune_buffer = [] 
        self.tune_max_len = 50 * 10  

    def position_callback(self, msg: Odometry):
        self.drone_position_cb_flag = True
        self.position = msg.pose.pose.position
        self.orientation = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        self.euler_angles = tf.transformations.euler_from_quaternion(self.orientation, 'sxyz')
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular
    
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
                self._sock.sendall(self._start_bytes)
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
                self._sock.sendall(self._start_bytes)
                return                  # 连上立即退出
            except Exception as e:
                rospy.logerr_throttle(2, f"重连失败: {e}，1 s 后重试…")
                self._sock = None
                time.sleep(1)
        rospy.logwarn("节点已停止，重连线程退出")

    def target_point_cb(self, msg: PointStamped):
        self.detect_receive_flag = True
        if msg.header.frame_id == "None":
            if rospy.Time.now() - self._switch_timer > rospy.Duration(1.0):
                self.image_in = False

        elif msg.header.frame_id == "Camera_Optical_Frame":
            self._switch_timer = rospy.Time.now()
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
    
    def unpack_status_byte(self, st: int):
        """
        返回 (video_src, algo, auto_tip, locked)
        """
        video_src = (st >> 6) & 0b11          # Bit7-6
        algo      = (st >> 4) & 0b11          # Bit5-4
        auto_tip  = bool(st & (1 << 3))       # Bit3
        locked    = bool(st & (1 << 2))       # Bit2
        return video_src, algo, auto_tip, locked

    def angle2pixel(
            self,
            angle_x: float,          # 水平脱靶角 [rad]
            angle_y: float,          # 垂直脱靶角 [rad]
        ):
        """
        把脱靶角（rad）转成像素坐标偏移量（pixel）
        返回: (delta_u, delta_v)  中心为原点的像素差
        """
        alpha = PIXEL_SIZE / CAM_FX                  # 每像素对应角度 [rad]
        delta_u = angle_x / alpha                    # 水平像素差
        delta_v = angle_y / alpha                    # 垂直像素差
        return int(round(delta_u)), int(round(delta_v))
                                    
    def _parse_and_publish(self, data):
        """ frame 保证 32 B 且校验通过 """
        # 解算K, 姿态角
        if data[31] != (sum(data[:31]) & 0xFF):
            # print(data[31], sum(data[:31]))
            return False, data, f"校验失败：calc={sum(data[:31]) & 0xFF} rx={data[31]}"
        if self.target_pixel_size <= 0:
            rospy.logwarn_throttle(1, "Waiting for target_pixel_size input")
            return False, data, f"no target_pixel_size"

        f_x = CAM_PIXEL_FX
        f_y = f_x 
        self.pod_f_pub.publish(Float32(f_x))
        # 左负右正，上正下负
        az_raw = int.from_bytes(data[11:13], 'little', signed=True) # 方位角
        el_raw = int.from_bytes(data[13:15], 'little', signed=True) # 俯仰角
        rl_raw = 0.0                                                # 偏转角
        self.az_pod_support, self.el_pod_support, self.rl_pod_support = az_raw * 0.01, el_raw * 0.01, rl_raw * 0.01
        att_str = f"吊舱方位角={self.az_pod_support:7.2f}° 吊舱俯仰角={self.el_pod_support:7.2f}° 吊舱横滚角={self.rl_pod_support:7.2f}°"
        # print(f"[recv] 来自吊舱的反馈：{hex_line} | \n {att_str}")

        c_x = IMAGE_WIGTH / 2
        c_y = IMAGE_HEIGHT / 2
        # 构造 K 矩阵
        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])
        
        target_z = REFER_TARGET_BOX_PIX_AREA ** 0.5 / self.target_pixel_size ** 0.5
        # rospy.loginfo_throttle(0.5, f"self.target_pixel_size, {self.target_pixel_size}")

        pixel_homogeneous = np.array([self.target_px, self.target_py, 1])
        # rospy.loginfo_throttle(1.0, pixel_homogeneous)

        normalized_coords = np.linalg.inv(K).dot(pixel_homogeneous)
        target_x = -normalized_coords[0] * target_z
        target_y = -normalized_coords[1] * target_z
        target_pod = np.array([target_z, target_x, target_y])
        # rospy.loginfo_throttle(0.5, f"target_pod, {target_pod}")

        # 吊舱坐标系 --> 舱座坐标系（X 前、Y 左、Z 上）
        R_pod_support = self.euler_to_rotation_matrix(self.az_pod_support, self.el_pod_support, self.rl_pod_support)
        target_support = R_pod_support.dot(target_pod) + self.T_pod_support
        
        # rospy.loginfo_throttle(0.5, f"target_support: {target_support}")

        # 舱座坐标系 --> 机体坐标系 Base_link（X 前、Y 左、Z 上）
        target_drone = self.R_support_drone.dot(target_support) + self.T_support_drone

        # rospy.loginfo_throttle(0.5, f"target_drone: {target_drone}")

        if self.drone_position_cb_flag == False:
            rospy.logwarn_throttle(1.0, "No Position Info")
            return True, data, ""
        if (self.orientation is None or len(self.orientation) != 4):
            rospy.logwarn_throttle(1, "orientation not ready")
            return True, data, ""
        # 无人机坐标 --> 世界坐标系
        self.T_drone_map = np.array([self.position.x,
                                self.position.y,
                                self.position.z])   
        self.R_drone_map = tf.transformations.quaternion_matrix(self.orientation)[:3,:3]
        target_position = self.R_drone_map.dot(target_drone) + self.T_drone_map
        
        target_position_pub = PointStamped()
        target_position_pub.header.stamp = rospy.Time.now()
        target_position_pub.header.frame_id = "odom"
        target_position_pub.point.x = target_position[0]
        target_position_pub.point.y = target_position[1]
        target_position_pub.point.z = target_position[2]
        self.odom_link_target_point_pub.publish(target_position_pub)

        az_error = int.from_bytes(data[5:7], 'little', signed=True)    # x-方位脱靶
        el_error = int.from_bytes(data[13:15], 'little', signed=True)  # y-俯仰脱靶
        az_pod_support_error = az_error * 0.05
        el_pod_support_error = el_error * 0.05

        pod_algo_target_px_error, pod_algo_target_py_error = self.angle2pixel(az_pod_support_error, el_pod_support_error)
        yolo_target_px_error = (self.target_px - IMAGE_WIGTH  / 2)   # x 右正
        yolo_target_py_error = -(self.target_py - IMAGE_HEIGHT / 2)  # y 上正

        if TRACK_MODE == 1:
            if pod_algo_target_px_error - yolo_target_px_error > IMAGE_WIGTH / 4 or pod_algo_target_py_error - yolo_target_py_error > IMAGE_HEIGHT / 4:
                self.pod_prejudiced_flag = True
                rospy.loginfo_throttle(1.0, f"Wrong Aiming: Podx-{pod_algo_target_px_error}, Yolox-{yolo_target_px_error}")


    def _recv_one_frame(self, timeout=0.2):
        """带自动调试的拆帧函数"""
        if not hasattr(self, '_recv_buf'):
            self._recv_buf = bytearray()

        while True:
            # ---------------- 先扫已有缓存 ----------------
            while len(self._recv_buf) >= 32:
                idx = 0
                # 找 55 AA
                while idx <= len(self._recv_buf) - 2:
                    if self._recv_buf[idx] == 0xEE and self._recv_buf[idx+1] == 0x16:
                        break
                    idx += 1
                else:                       # 没找到帧头
                    if DEBUG_RECV:
                        rospy.loginfo_throttle(2.0, "[recv] no 55 AA header, drop %d B" % (len(self._recv_buf)-1))
                    self._recv_buf = self._recv_buf[-1:]
                    break

                if len(self._recv_buf) - idx < 32:        # 长度不够
                    break

                rsp = self._recv_buf[idx:idx+32]
                cs_calc = sum(rsp[:31]) & 0xFF
                cs_expect = rsp[31]
                if cs_calc == cs_expect:                  # 校验 OK
                    self._recv_buf = self._recv_buf[idx+32:]
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
                condition0 = not (self.image_in and self.mode == 2) # 未进入跟踪锁定
                condition1 = payload != self._last_payload and condition0 # 未进入跟踪锁定下的未重复命令
                condition2 = self.pod_prejudiced_flag # yolo校验判断吊舱跟偏
                if TRACK_MODE == 1:
                    if condition1 or condition2:
                        self._sock.sendall(payload)
                        self.pod_prejudiced_flag = False
                elif TRACK_MODE == 2:
                    if condition1:
                        self._sock.sendall(payload)
                else:
                    rospy.logwarn_throttle(2.0, "Unexpected mode")
                self._last_payload = payload
                ok, rsp, reason = self._recv_one_frame(timeout)
                track_video_src, track_algo, target_auto_tip, target_locked = self.unpack_status_byte(rsp)
                if ok:
                    # rospy.loginfo_throttle(0.5, f"接收字节位数 {len(rsp)}: {rsp.hex(' ')}")
                    if target_locked == 0:
                        self.mode = 1
                    elif target_locked == 1:
                        self.mode = 2
                    else:
                        rospy.logwarn_throttle(1, "Unpredict Mode")
                    self._parse_and_publish(rsp)
                    return True, rsp 
                # 失败分两条打
                if not rsp:
                    pass
                    # rospy.logwarn_throttle(1, f"[recv] 未收到任何数据")
                else:
                    rospy.logwarn_throttle(1, f"[recv] {reason}")
            except Exception as e:
                self._sock = None
                rospy.logwarn_throttle(1, f"[recv] 异常：{e}")
        return False, b''
   
    
    def _send_loop(self):
        while self._running:
            if COM_CONTROL == True:
                if self.image_in:
                    if TRACK_MODE == 1:
                        header = bytearray([0xEB, 0x90, 0x16])
                        header_core = bytearray([0xEB, 0x90])
                        cmdhead = bytearray([0x0D])
                        target_px_error = (self.target_px - IMAGE_WIGTH  / 2)   # x 右正
                        target_py_error = -(self.target_py - IMAGE_HEIGHT / 2)  # y 上正
                        byte_target_px = struct.pack('<h', int(target_px_error))
                        byte_target_py = struct.pack('<h', int(target_py_error))
                        cmddata = bytearray(byte_target_px + byte_target_py)
                        byte_other = bytearray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
                    
                        checksum_core = bytearray([0x00])
                        frame_core = header_core + cmdhead + cmddata + byte_other + checksum_core
                        checksum_core = sum(frame_core[0:-1]) & 0xFF
                        frame_core[-1] = checksum_core

                        checksum = bytearray([0x00])
                        frame = header + frame_core + checksum
                        checksum = sum(frame[0:-1]) & 0xFF
                        frame[-1] = checksum
                        self._cmd_bytes = bytes(frame)
                        # rospy.loginfo_throttle(0.01, f"Mode: {self.mode}")
                        if self.pod_prejudiced_flag:
                            self._cmd_bytes = bytes.fromhex(self.END_TRACK_HEX_STR.replace(" ", ""))
                    elif TRACK_MODE == 2:
                        self._cmd_bytes = bytes.fromhex(self.DETECT_TO_TRACK_STR.replace(" ", ""))
                    else:
                        rospy.logwarn_throttle(2.0, "Unexpected mode")

                    
                elif self.mode == 2:
                    self._cmd_bytes = bytes.fromhex(self.END_TRACK_HEX_STR.replace(" ", ""))
                else:
                    rospy.loginfo_throttle(1, "Return to Center")
                    self._cmd_bytes = bytes.fromhex(self.CMD_HEX_STR.replace(" ", ""))

                # rospy.loginfo_throttle(0.01, f"cmd_send: {self._cmd_bytes.hex(' ')}")
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