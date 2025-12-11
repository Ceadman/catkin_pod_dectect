#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import cv2
import threading
from solution_and_control import IMAGE_HEIGHT, IMAGE_WIGTH


class UltralyticsROS:
    def __init__(self):
        # 相机相对无人机坐标
        self.T_cam_drone = np.array([0.2, 0.035, 0])
        self.R_cam_drone = np.eye(3)
        
        # 初始化参数
        self.frame = None
        self.img_receive_flag = False
        self.last_target_px = None
        self.img_width = IMAGE_WIGTH
        self.img_height = IMAGE_HEIGHT
        self.img_update_flag = False
        ROOT = Path(__file__).resolve().parent
        weights_path = ROOT / "../ultralytics/detect/train/weights/yolo11n.pt" 
        # 可用自己训练的数据集路径替换
        # weights_path = ROOT / "../ultralytics/detect/train2/weights/best.pt"
        self.detection_model = None
        
        self.bridge = CvBridge()
        
        rospy.Subscriber("/pod/image_raw", Image, self.pod_img_callback)
        self.img_link_target_point_pub = rospy.Publisher("/detection/ultralytics/img_link_target_pub", PointStamped, queue_size=5)

        # 订阅参数
        self.detection_model = YOLO(weights_path)
        self.max_track_dist = self.img_height / 4
        self.pub_2dpoint = PointStamped()
        self.pub_2dpoint.point.x = IMAGE_WIGTH / 2 
        self.pub_2dpoint.point.y = IMAGE_HEIGHT / 2
        

        # 画图线程
        # self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        # self.display_thread.start()

    # ========== 画图线程 ==========
    # def _display_loop(self):
    #     """死循环显示最新带标注的图像"""
    #     cv2.namedWindow("YOLO-ROS", cv2.WINDOW_NORMAL)
    #     while True:
    #         if hasattr(self, 'drawn_frame') and self.drawn_frame is not None:
    #             cv2.imshow("YOLO-ROS", self.drawn_frame)
    #         if cv2.waitKey(1) & 0xFF == 27:   # ESC 退出
    #             break
    #     cv2.destroyAllWindows()

    # ==========  画框+中心 ==========
    def _draw_boxes_and_center(self, frame, boxes, classes, target_px):
        """返回画好框/中心点的 BGR 图像"""
        overlay = frame.copy()
        names_dict = self.detection_model.names
        for b, c in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, b)
            cls_name = names_dict[int(c)]
            color = (0, 255, 0) if cls_name == 'car' else (0, 255, 255)  # 车绿 人黄
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, cls_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # 画全局中心点
        cv2.circle(overlay, (int(self.img_width // 2), int(self.img_height // 2)),
                   4, (0, 0, 255), -1)
        # 画当前跟踪目标中心
        if target_px is not None:
            cv2.circle(overlay, (int(target_px[0]), int(target_px[1])),
                       6, (255, 0, 0), -1)
        return overlay
    
    def pod_img_callback(self, msg):
        self.img_receive_flag = True
        if msg.header.frame_id == "none":
            self.img_update_flag = False
        if msg.header.frame_id == "pod":
            self.img_update_flag = True
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        return
    
    def detect_from_pod(self):
        while self.img_receive_flag == False:
            pass
        if self.frame.ndim == 2:
            self.img_height, self.img_width = self.frame.shape          # h=行数，w=列数
        # 彩色图（BGR 或 RGB）
        elif self.frame.ndim == 3:
            self.img_height, self.img_width = self.frame.shape[:2]      # 前两个维度就是高、宽
        det_result = self.detection_model(self.frame, imgsz=640, conf=0.5)
        print("1")
        cv2.imshow("YOLO", det_result[0].plot())
        if cv2.waitKey(1) & 0xFF == 27:
            return
        boxes = det_result[0].boxes.xyxy.cpu().numpy()
        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        
        names_dict = det_result[0].names
        reverse_map = {v: k for k, v in names_dict.items()}
        car_cls_id   = reverse_map.get('car',   -1)
        person_cls_id= reverse_map.get('person',-1)

        valid_idx = np.where((classes == car_cls_id) | (classes == person_cls_id))[0]
        if valid_idx.size == 0:
            rospy.loginfo("No car or person in this frame")
            self.pub_2dpoint.header.stamp = rospy.Time.now()
            self.pub_2dpoint.header.frame_id = "None"
            self.img_link_target_point_pub.publish(self.pub_2dpoint)
            
            return
        # 框中心(N, 2)
        valid_boxes = boxes[valid_idx]
        centers = np.stack([(valid_boxes[:, 0] + valid_boxes[:, 2]) / 2,
                (valid_boxes[:, 1] + valid_boxes[:, 3]) / 2], axis=1) 
        areas = abs(valid_boxes[:, 0] - valid_boxes[:, 2]) * abs(valid_boxes[:, 1] - valid_boxes[:, 3])
        # 镜头中心
        img_center = np.array([self.img_width / 2.0, self.img_height / 2.0])

        if self.last_target_px is None:
            # 第一次：选离镜头中心最近的框中心
            dist_to_center = np.linalg.norm(centers - img_center, axis=1)
            target_idx = np.argmin(dist_to_center)
            self.target_px = centers[target_idx]
        else:
            # 计算与上次目标点的距离
            dist_to_last = np.linalg.norm(centers - self.last_target_px, axis=1)
            within_range = dist_to_last <= self.max_track_dist

            if np.any(within_range):
                # 在阈值范围内选最近的
                target_idx = np.argmin(dist_to_last[within_range])
                target_idx = np.where(within_range)[0][target_idx]
                self.target_px = centers[target_idx]
            else:
                # 全部超出阈值，退回到镜头中心最近
                dist_to_center = np.linalg.norm(centers - img_center, axis=1)
                target_idx = np.argmin(dist_to_center)
                self.target_px = centers[target_idx]
        self.last_target_px = self.target_px.copy()
        
        self.drawn_frame = self._draw_boxes_and_center(
            self.frame, valid_boxes, classes[valid_idx], self.target_px)
        
        # 判断镜头中心是否落在目标框内AIM区域
        target_box = valid_boxes[target_idx]
        w_half = (target_box[2] - target_box[0]) * 0.5
        h_half = (target_box[3] - target_box[1]) * 0.5
        cx, cy = img_center
        if (abs(img_center[0] - cx) <= w_half) and (abs(img_center[1] - cy) <= h_half):
            self.pub_2dpoint.header.frame_id = "Image_Frame_AIM"
        else:
            self.pub_2dpoint.header.frame_id = "Image_Frame"
        self.pub_2dpoint.header.stamp = rospy.Time.now()
        self.pub_2dpoint.point.x = float(self.target_px[0])   # x 像素
        self.pub_2dpoint.point.y = float(self.target_px[1])   # y 像素 
        self.img_link_target_point_pub.publish(self.pub_2dpoint)

# 加载训练好的模型，改为自己的路径
if __name__ == "__main__":
    rospy.init_node("ultralytics")
    rospy.loginfo("ultralytics init")
    detection = UltralyticsROS()

    while not rospy.is_shutdown():
        if detection.img_update_flag == True:
            detection.detect_from_pod()
        else:
            detection.last_target_px = None
            cv2.destroyAllWindows
    
    # print(model.names)
    # detection_model.predict(source, save=True)

