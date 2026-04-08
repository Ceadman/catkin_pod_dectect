#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

RTSP_URL  = "rtsp://192.168.144.119:554"   # 端口是 554，路径留空即可
ROS_TOPIC = "/pod/image_raw"
QUEUE     = 5

RETRY_BASE = 0.5   # 首次重连间隔
RETRY_MAX  = 5.0   # 最大重试间隔
retry_wait = RETRY_BASE
LOST_THR = 60
lost_cnt = 0

def open_cap():
    """返回一个成功打开的 cv2.VideoCapture 实例（CPU 软解，零 GPU 显存）"""
    gst_str = (
        "rtspsrc location={}"
        " latency=1 buffer-mode=1 drop-on-latency=false "
        " ntp-sync=false tcp-timeout=5000000 protocols=tcp ! "
        "rtph265depay ! "
        "h265parse disable-passthrough=true ! "
        "nvv4l2decoder enable-frame-type-reporting=false ! "  # ← 硬解
        "nvvidconv output-buffers=2 ! " 
        "video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink max-buffers=2 drop=false sync=false emit-signals=false"
    ).format(RTSP_URL)
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    return cap

def main():
    global retry_wait
    rospy.init_node("rtsp_cam_publisher", anonymous=True)
    rospy.loginfo("waiting for yolo ready...")
    rospy.wait_for_message("/yolo/ready", Empty)   # 阻塞直到收到
    rospy.loginfo("YOLO ready, start streaming")

    img_pub = rospy.Publisher(ROS_TOPIC, Image, queue_size=QUEUE)
    bridge = CvBridge()
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        cap = open_cap()
        if cap and cap.isOpened():
            break
        rospy.logerr("初始打开失败，%.1f s 后重试", retry_wait)
        rospy.sleep(retry_wait)
        retry_wait = min(retry_wait * 2, RETRY_MAX)
    retry_wait = RETRY_BASE          # 成功后复位
    rospy.loginfo("RTSP 流已连接，开始发布 /pod/image_raw")
    # cv2.namedWindow("pod", cv2.WINDOW_NORMAL)   # 调试窗口，可删

    lost_cnt = 0
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            lost_cnt = 0
            msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = rospy.Time.now()
            img_pub.publish(msg)
            # cv2.imshow("pod", frame)          # 调试用
            # cv2.waitKey(1)
            retry_wait = RETRY_BASE           # 成功后重置
            continue
        
        lost_cnt += 1
        if lost_cnt < LOST_THR:
            continue                      # 轻微丢包，不重建

        # 丢帧 -> 释放并重新建 capture
        rospy.logwarn("帧丢失，%.1f s 后重连", retry_wait)
        cap.release()
        rospy.sleep(retry_wait)
        while not rospy.is_shutdown():
            cap = open_cap()
            if cap and cap.isOpened():
                lost_cnt = 0
                retry_wait = RETRY_BASE
                rospy.loginfo("RTSP 重连成功")
                break
            rospy.logerr("重连失败，%.1f s 后再试", retry_wait)
            rospy.sleep(retry_wait)
            retry_wait = min(retry_wait * 2, RETRY_MAX)

    cap.release()
    del cap
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass