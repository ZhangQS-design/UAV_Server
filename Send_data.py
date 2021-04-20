# -*- coding: utf-8 -*-
import socket
import cv2
import numpy as np
import time
import struct

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def Socket_UDP(image):
    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    data = data_encode.tobytes()
    fhead = struct.pack("l", len(data))

    print(len(data))
    print(data)
    # 发送数据:
    s.sendto(fhead, ('192.168.1.22', 9909))
    for i in range(len(data) // 1024 + 1):
        if (i + 1) * 1024 > len(data):
            s.sendto(data[i * 1024:], ('192.168.1.22', 9909))
        else:
            s.sendto(data[i * 1024:(i + 1) * 1024], ('192.168.1.22', 9909))

    # s.sendto(img_encode, ('192.168.1.21', 9909))
    # 接收数据:
    # print(s.recv(1024).decode('utf-8'))
    # s.close()

