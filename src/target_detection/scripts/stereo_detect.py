#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
import tf
import cv2
from cv_bridge import CvBridge

REFER_TARGET_BOX_PIX_AREA = 3.82e5 * 10 ** 2 # 10m深度目标识别框像素面积
STEREO_B = 0.07 # 双目左右相机距离

class UltralyticsROS:
    def __init__(self):
        # 相机相对无人机坐标
        self.T_cam_drone = np.array([0.12, 0.035, 0])
        self.R_cam_drone = np.eye(3)

        # 初始化参数
        self.drone_position_cb_flag = False
        self.last_left_target_px = None
        self.last_right_target_px = None
        self.left_height = None
        ROOT = Path(__file__).resolve().parent
        # weights_path = ROOT / "../ultralytics/runs/detect/train/weights/yolo11n.pt" 
        weights_path = ROOT / "../model/ball.pt"
        self.detection_model = None
        self.last_target_px = None

        self.bridge = CvBridge()
        rospy.Subscriber("/tailsitter_0/mavros/local_position/odom", Odometry, self.position_callback)
        rospy.Subscriber("/stereo/left/camera_info", CameraInfo, self.left_cam_info_callback)
        rospy.Subscriber("/stereo/right/camera_info", CameraInfo, self.right_cam_info_callback)
        rospy.Subscriber("/stereo/left/image_raw", Image, self.left_image_callback)
        rospy.Subscriber("/stereo/right/image_raw", Image, self.right_image_callback)
        self.left_target_point_pub = rospy.Publisher("/detection/ultralytics/left_target_point_pub", PointStamped, queue_size=5)
        self.right_target_point_pub = rospy.Publisher("/detection/ultralytics/right_target_point_pub", PointStamped, queue_size=5)
        self.odom_link_target_point_pub = rospy.Publisher("/detection/odom/target_point", PointStamped, queue_size=5)
        rospy.Subscriber("/detection/ultralytics/left_target_point_pub", PointStamped, self.target_processor)

        # 订阅参数
        self.detection_model = YOLO(weights_path)
        # YOLO预热
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.detection_model(dummy, imgsz=640, conf=0.5)
        rospy.loginfo("YOLO warm-up done")
        while self.left_height == None:
            pass

        # 算法参数
        self.max_track_dist = self.left_height/4
        self.target_pixel_size = None

    def position_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position
        self.orientation = np.array([msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w])
        self.euler_angles = tf.transformations.euler_from_quaternion(self.orientation, 'sxyz')
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular
        self.drone_position_cb_flag = True

    def left_cam_info_callback(self, msg: CameraInfo):
        self.left_K = np.array(msg.K).reshape(3, 3)
        self.left_D = msg.D
        self.left_f_x = self.left_K[0, 0]
        self.left_f_y = self.left_K[1, 1]
        self.left_width = msg.width
        self.left_height = msg.height

    def right_cam_info_callback(self, msg: CameraInfo):
        self.right_K = np.array(msg.K).reshape(3, 3)
        self.right_D = msg.D
        self.right_f_x = self.right_K[0, 0]
        self.right_f_y = self.right_K[1, 1]
        self.right_width = msg.width
        self.right_height = msg.height
    
    def left_image_callback(self, msg):
        """Callback function to process image and publish annotated images."""
        if self.detection_model != None:
            # 使用ros_numpy获取np格式图像-->便于直接输入YOLO
            array = ros_numpy.numpify(msg)
            if array.ndim == 2:
                h, w = array.shape          # h=行数，w=列数
            # 彩色图（BGR 或 RGB）
            elif array.ndim == 3:
                h, w = array.shape[:2]      # 前两个维度就是高、宽
            # print('size =', self.target_pixel_size)
            # print('width  =', w)

            # 发布离中心点最近的识别框中心
            det_result = self.detection_model(array, imgsz=640, conf=0.5)
            boxes = det_result[0].boxes.xyxy.cpu().numpy()
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            if 'ball' not in names:
                rospy.loginfo("No ball class in this frame")
                left_pub_point = PointStamped()
                left_pub_point.header.stamp = rospy.Time.now()
                left_pub_point.header.frame_id = "None"
                self.left_target_point_pub.publish(left_pub_point)
                return
            names_dict = det_result[0].names
            reverse_map = {v: k for k, v in names_dict.items()}
            car_cls_id = reverse_map['ball']
            car_idx = np.where(classes == car_cls_id)[0]

            # car框中心(N, 2)
            car_boxes = boxes[car_idx]
            centers = np.stack([(car_boxes[:, 0] + car_boxes[:, 2]) / 2,
                    (car_boxes[:, 1] + car_boxes[:, 3]) / 2], axis=1) 
            area = abs(car_boxes[:, 0] - car_boxes[:, 2]) * abs(car_boxes[:, 1] - car_boxes[:, 3])
            # 镜头中心
            img_center = np.array([self.left_width / 2.0, self.left_height / 2.0])

            if self.last_target_px is None:
                # 第一次：选离镜头中心最近的框中心
                dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                target_idx = np.argmin(dist_to_center)
                self.left_target_px = centers[target_idx]
            else:
                # 计算与上次目标点的距离
                dist_to_last = np.linalg.norm(centers - self.last_target_px, axis=1)
                within_range = dist_to_last <= self.max_track_dist

                if np.any(within_range):
                    # 在阈值范围内选最近的
                    target_idx = np.argmin(dist_to_last[within_range])
                    target_idx = np.where(within_range)[0][target_idx]
                    self.left_target_px = centers[target_idx]
                else:
                    # 全部超出阈值，退回到镜头中心最近
                    dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                    target_idx = np.argmin(dist_to_center)
                    self.left_target_px = centers[target_idx]
            self.target_pixel_size = float(area[target_idx])
            self.last_left_target_px = self.left_target_px.copy()
            
            left_pub_point = PointStamped()
            left_pub_point.header.stamp = rospy.Time.now()
            left_pub_point.header.frame_id = "Left_Camera_Optical_Frame"   # 改成你相机的 frame
            left_pub_point.point.x = float(self.left_target_px[0])   # x 像素
            left_pub_point.point.y = float(self.left_target_px[1])   # y 像素 
            self.left_target_point_pub.publish(left_pub_point)

    def right_image_callback(self, msg):
        """Callback function to process image and publish annotated images."""
        if self.detection_model != None:
            # 使用ros_numpy获取np格式图像-->便于直接输入YOLO
            array = ros_numpy.numpify(msg)
            if array.ndim == 2:
                h, w = array.shape          # h=行数，w=列数
            # 彩色图（BGR 或 RGB）
            elif array.ndim == 3:
                h, w = array.shape[:2]      # 前两个维度就是高、宽
            # print('height =', h)
            # print('width  =', w)

            # 发布离中心点最近的识别框中心
            det_result = self.detection_model(array, imgsz=640, conf=0.5)
            boxes = det_result[0].boxes.xyxy.cpu().numpy()
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            if 'ball' not in names:
                rospy.loginfo("No ball class in this frame")
                right_pub_point = PointStamped()
                right_pub_point.header.stamp = rospy.Time.now()
                right_pub_point.header.frame_id = "None"
                self.right_target_point_pub.publish(right_pub_point)
                return
            names_dict = det_result[0].names
            reverse_map = {v: k for k, v in names_dict.items()}
            car_cls_id = reverse_map['ball']
            car_idx = np.where(classes == car_cls_id)[0]

            # car框中心(N, 2)
            car_boxes = boxes[car_idx]
            centers = np.stack([(car_boxes[:, 0] + car_boxes[:, 2]) / 2,
                    (car_boxes[:, 1] + car_boxes[:, 3]) / 2], axis=1) 
            areas = abs(car_boxes[:, 0] - car_boxes[:, 2]) * abs(car_boxes[:, 1] - car_boxes[:, 3])
            # 镜头中心
            img_center = np.array([self.right_width / 2.0, self.right_height / 2.0])

            if self.last_target_px is None:
                # 第一次：选离镜头中心最近的框中心
                dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                target_idx = np.argmin(dist_to_center)
                self.right_target_px = centers[target_idx]
            else:
                # 计算与上次目标点的距离
                dist_to_last = np.linalg.norm(centers - self.last_target_px, axis=1)
                within_range = dist_to_last <= self.max_track_dist

                if np.any(within_range):
                    # 在阈值范围内选最近的
                    target_idx = np.argmin(dist_to_last[within_range])
                    target_idx = np.where(within_range)[0][target_idx]
                    self.right_target_px = centers[target_idx]
                else:
                    # 全部超出阈值，退回到镜头中心最近
                    dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                    target_idx = np.argmin(dist_to_center)
                    self.right_target_px = centers[target_idx]
            self.last_right_target_px = self.right_target_px.copy()
            # rospy.loginfo(f"Target point: ({self.right_target_px[0]:.1f}, {self.right_target_px[1]:.1f})")
            right_pub_point = PointStamped()
            right_pub_point.header.stamp = rospy.Time.now()
            right_pub_point.header.frame_id = "Right_Camera_Optical_Frame"   # 改成你相机的 frame
            right_pub_point.point.x = float(self.last_right_target_px[0])   # x 像素
            right_pub_point.point.y = float(self.last_right_target_px[1])   # y 像素 
            self.right_target_point_pub.publish(right_pub_point)

    def target_processor(self, msg: PointStamped):
        if msg.header.frame_id == "None" or self.last_left_target_px is None or self.last_right_target_px is None:
            return
        # 检查双目相机参数相同
        if np.array_equal(self.left_K, self.right_K):
            stereo_K = self.left_K
        else:
            rospy.logwarn("Stereo Uncalibrated")
            return
        
        if self.drone_position_cb_flag == False:
            print("No Position Info")
            return
        
        # 像素平面坐标 -stereo_K-> 相机坐标系
        diffs = abs(self.last_left_target_px[0] - self.last_right_target_px[0])
        # print(f"size: {self.target_pixel_size}")

        # target_z = STEREO_B * self.left_f_x / diffs
        target_z = REFER_TARGET_BOX_PIX_AREA ** 0.5 / self.target_pixel_size ** 0.5

        pixel_homogeneous = np.array([self.last_left_target_px[0], self.last_left_target_px[1], 1])
        print("-----------target_x_y---------", pixel_homogeneous)

        normalized_coords = np.linalg.inv(stereo_K).dot(pixel_homogeneous)
        target_x = -normalized_coords[0] * target_z
        target_y = -normalized_coords[1] * target_z
        target_camera = np.array([target_z, target_x, target_y])
        print("-----------target_z---------", target_z)

        # 相机坐标系 -cam_drone-> 无人机坐标
        # self.T_cam_drone = np.array([0.12, 0.035, 0])
        # self.R_cam_drone = np.eye(3)
        target_drone = self.R_cam_drone.dot(target_camera) + self.T_cam_drone
        print(target_drone)

        # 无人机坐标 --> 世界坐标系
        self.T_drone_map = np.array([self.position.x,
                                 self.position.y,
                                 self.position.z])   
        self.R_drone_map = tf.transformations.quaternion_matrix(self.orientation)[:3,:3]
        target_position = self.R_drone_map.dot(target_drone) + self.T_drone_map

        target_position_pub = PointStamped()
        target_position_pub.header.stamp = rospy.Time.now()
        target_position_pub.header.frame_id = "map"
        target_position_pub.point.x = target_position[0]
        target_position_pub.point.y = target_position[1]
        target_position_pub.point.z = target_position[2]
        self.odom_link_target_point_pub.publish(target_position_pub)
        # print(target_position_pub)     


# 加载训练好的模型，改为自己的路径
if __name__ == "__main__":
    rospy.init_node("ultralytics")
    rospy.loginfo("ultralytics init")
    node = UltralyticsROS()
    rospy.spin()
    # print(model.names)
    # detection_model.predict(source, save=True)

