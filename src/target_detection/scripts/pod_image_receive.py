#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import socket
from sensor_msgs.msg import Image
import numpy as np
import rospy
import time
from cv_bridge import CvBridge

def reopen():
    global cap
    cap.release()
    rospy.logwarn("ffmpeg session dead, reopening udp stream …")
    time.sleep(0.5)
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3_000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  1_000)

if __name__ == "__main__":
    rospy.init_node("image_receive")
    rospy.loginfo("image_receive init")
    img_pub = rospy.Publisher("/pod/image_raw", Image, queue_size=5)
    img = Image()
    bridge = CvBridge()

    URL = "udp://@192.168.144.160:10004"
    MAX_FAIL = 30 # 连续 30 帧失败就重建
    fail_cnt = 0
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3_000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  1_000)

    while not rospy.is_shutdown():
        ret = cap.grab()         
        if ret:
            fail_cnt = 0
            _, frame = cap.retrieve()
            img = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img.header.frame_id = "pod"
            img.header.stamp = rospy.Time.now()
        else:
            fail_cnt += 1
            if fail_cnt >= MAX_FAIL:
                reopen()
                fail_cnt = 0
            else:
                rospy.logwarn_throttle(2, "Waiting for Image …")
            img.header.frame_id = "none"
            img.header.stamp = rospy.Time.now()
        img_pub.publish(img)
    cap.release()
