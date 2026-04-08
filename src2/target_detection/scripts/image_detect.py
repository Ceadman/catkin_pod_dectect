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
import sys
from cv_bridge import CvBridge
from std_msgs.msg import Float32 
import torch, gc
from std_msgs.msg import Empty
IMAGE_HEIGHT = 1080
IMAGE_WIGTH = 1920

# 目标识别框像素面积 * 深度 * 深度
REFER_TARGET_BOX_PIX_AREA = 1.3* 96000 * 11.72 * 11.72
class UltralyticsROS:
    def __init__(self):
        # GPU 预占
        torch.cuda.empty_cache()
        gc.collect()
        dummy = torch.empty((64,1024,1024), dtype=torch.uint8, device=0)
        del dummy; 
        torch.cuda.empty_cache()

        ROOT        = Path(__file__).resolve().parent
        weights_path = ROOT / "../model/balloon1.engine"
        self.detection_model = YOLO(weights_path)
        # 预热一次（uint8 -> float32 由 Ultralytics 自动做）
        # _ = self.detection_model.predict(
        #         np.zeros((640,640,3), np.uint8), imgsz=640, device=0)
        # rospy.loginfo("YOLO warm-up done")
        self.wait_gpu_continuous(min_gb=0.4)
        self.ready_pub = rospy.Publisher("/yolo/ready", Empty, queue_size=1, latch=True)
        self.ready_pub.publish(Empty()) # 通知rstp开始硬解
        # 初始化参数
        self.drone_position_cb_flag = False
        self.last_target_pt = None
        self.last_right_target_pt = None
        self.height = None
        
        self.last_target_pt = None

        self.bridge = CvBridge()
        rospy.Subscriber("/pod/image_raw", Image, self.image_callback)
        self.target_point_pub = rospy.Publisher("/detection/ultralytics/target_point_pub", PointStamped, queue_size=5)   
        self.pub_target_pixel_size = rospy.Publisher('/detection/ultralytics/target_pixel_size', Float32, queue_size=10)   

        # 算法参数
        self.max_track_dist = IMAGE_HEIGHT/4
        self.target_pixel_size = None
    
    def wait_gpu_continuous(self, min_gb=0.6, interval=1.0, timeout=60):
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            # 读系统剩余物理内存 ≈ GPU 可用（共享内存）
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable'):
                        kb = int(line.split()[1])
                        free_gb = kb / 1e6
                        break
            if free_gb >= min_gb:
                rospy.loginfo(f"系统剩余内存充足：{free_gb:.2f} GB > {min_gb} GB")
                return True
            rospy.logwarn_throttle(1, f"系统剩余内存不足：{free_gb:.2f} GB < {min_gb} GB，等待中...")
            if rospy.Time.now().to_sec() - t0 > timeout:
                rospy.logerr("等待 GPU 内存超时，节点退出")
                sys.exit(1)
            rospy.sleep(interval)

    def image_callback(self, msg: Image):
        """Callback function to process image and publish annotated images."""   
        self.height = msg.height
        self.width = msg.width
        # print(self.height,self.width)
        if self.detection_model != None:
            # 使用ros_numpy获取np格式图像-->便于直接输入YOLO
            array = ros_numpy.numpify(msg).copy()
            if array.ndim == 2:
                h, w = array.shape          # h=行数，w=列数
            # 彩色图（BGR 或 RGB）
            elif array.ndim == 3:
                h, w = array.shape[:2]      # 前两个维度就是高、宽
            # print('size =', self.target_pixel_size)

            # 发布离中心点最近的识别框中心
            det_result = self.detection_model(array, imgsz=640, conf=0.7, device=0)
            show = det_result[0].plot()
            show = cv2.resize(show,None,fx=0.1,fy=0.1)
            #cv2.imshow('YOLO', show)
            #cv2.waitKey(1)
            boxes = det_result[0].boxes.xyxy.cpu().numpy()
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            if 'balloon' not in names:
                rospy.loginfo("No balloon class in this frame")
                pub_point = PointStamped()
                pub_point.header.stamp = rospy.Time.now()
                pub_point.header.frame_id = "None"
                self.target_point_pub.publish(pub_point)
                return
            names_dict = det_result[0].names
            reverse_map = {v: k for k, v in names_dict.items()}
            car_cls_id = reverse_map['balloon']
            car_idx = np.where(classes == car_cls_id)[0]

            # car框中心(N, 2)
            car_boxes = boxes[car_idx]
            centers = np.stack([(car_boxes[:, 0] + car_boxes[:, 2]) / 2,
                    (car_boxes[:, 1] + car_boxes[:, 3]) / 2], axis=1) 
            area = abs(car_boxes[:, 0] - car_boxes[:, 2]) * abs(car_boxes[:, 1] - car_boxes[:, 3])
            # 镜头中心
            img_center = np.array([self.width / 2.0, self.height / 2.0])

            if self.last_target_pt is None:
                # 第一次：选离镜头中心最近的框中心
                dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                target_idx = np.argmin(dist_to_center)
                self.target_pt = centers[target_idx]
            else:
                # 计算与上次目标点的距离
                dist_to_last = np.linalg.norm(centers - self.last_target_pt, axis=1)
                within_range = dist_to_last <= self.max_track_dist

                if np.any(within_range):
                    # 在阈值范围内选最近的
                    target_idx = np.argmin(dist_to_last[within_range])
                    target_idx = np.where(within_range)[0][target_idx]
                    self.target_pt = centers[target_idx]
                else:
                    # 全部超出阈值，退回到镜头中心最近
                    dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                    target_idx = np.argmin(dist_to_center)
                    self.target_pt = centers[target_idx]
            self.target_pixel_size = float(area[target_idx])
            self.last_target_pt = self.target_pt.copy()
            
            pub_point = PointStamped()
            pub_point.header.stamp = rospy.Time.now()
            pub_point.header.frame_id = "Camera_Optical_Frame"   # 改成你相机的 frame
            pub_point.point.x = float(self.last_target_pt[0])   # x 像素
            pub_point.point.y = float(self.last_target_pt[1])   # y 像素 
            #rospy.logwarn_throttle(0.5, f"balloon_point:({pub_point.point.x},{pub_point.point.y})")
            self.target_point_pub.publish(pub_point)

            self.pub_target_pixel_size.publish(Float32(self.target_pixel_size))



# 加载训练好的模型，改为自己的路径
if __name__ == "__main__":
    rospy.init_node("ultralytics")
    rospy.loginfo("ultralytics init")
    node = UltralyticsROS()
    rospy.spin()
    # print(model.names)
    # detection_model.predict(source, save=True)

