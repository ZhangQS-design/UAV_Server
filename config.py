# -*- coding: utf-8 -*-
import socket
import cv2
import numpy as np
import struct
from server.data_lab import *


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def Socket_UDP(image):
    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    data = data_encode.tobytes()
    fhead = struct.pack("l", len(data))

    #print(len(data))
    # 发送数据:
    # 无人机地址和端口
    s.sendto(fhead, ('192.168.123.45', 9909))
    for i in range(len(data) // 1024 + 1):
        if (i + 1) * 1024 > len(data):
            s.sendto(data[i * 1024:], ('192.168.123.45', 9909))
        else:
            s.sendto(data[i * 1024:(i + 1) * 1024], ('192.168.123.45', 9909))

    # s.sendto(img_encode, ('192.168.1.21', 9909))
    # 接收数据:
    # print(s.recv(1024).decode('utf-8'))
    # s.close()

# 调用摄像头
# import cv2
# from datetime import datetime
# import time
#
# FILENAME = 'DJI_Demo_Detect.avi'
# WIDTH = 1280
# HEIGHT = 720
# FPS = 10
#
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# cap.set(cv2.CAP_PROP_FPS, 10)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
#
# out = cv2.VideoWriter(FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
#
# start_time = datetime.now()
# #
# while True:
#      ret, frame = cap.read()
#      if ret:
#          out.write(frame)
#          if (datetime.now()-start_time).seconds == 100:
#              cap.release()
#              break
#
# out.release()
# cv2.destroyAllWindows()
