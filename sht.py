#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# In[2]:


import time
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import cv2
from server.data_lab import *


#from utils import label_map_util
#from utils import visualization_utils as vis_util
#model_path = 'colab_luck/frozen_inference_graph.pb'
#config_path = 'colab_luck/detect_person.pbtxt'
model_path = 'model/person.pb'
config_path = 'model/person.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
count_p = 0
a = []

def cvdetect_person(image_np):

    global count_p
    count_p += 1
    rows = image_np.shape[0]
    cols = image_np.shape[1]
    start = time.time()
    net.setInput(cv2.dnn.blobFromImage(image_np,size=(300, 300), swapRB=True, crop=False))
    cvOut = net.forward()

    #print(cvOut)
    for detection in cvOut[0, 0, :, :]:
        label = detection[1]
        score = float(detection[2])
        if score > 0.7 and label == 1:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            x_center = (left + right) / 2  # 计算每个帧各个对象的中心坐标
            y_center = (top + bottom) / 2

            #绘制
            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)
            cv2.putText(image_np, "person", (int(left), int(top - 10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)



            # print("----------------")
            # print("盲区有行人经过，注意减速！！！！！")
            # print("----------------")
    # end = time.time()
    # a.append(end - start)
    # if count_p == 200:
    #     print(a)
    #     data_write("data_delay/ssd_delay.xlsx", a)
    #print("Execution Time: ", end - start)




    return image_np

def plt_photo(a):
    global count_p
    count_p += 1
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('图片帧数')
    plt.ylabel('时延 /ms')
    plt.title("处理单元的推断时延")
    plt.xlim(xmax=200, xmin=0)
    plt.ylim(ymax=1000, ymin=0)
    # 画两条（0-9）的坐标轴并设置轴标签x，y
    x1 = 1
    for i in a:
        i = i-0.1
        y1 = i*1000*10
        colors1 = '#00CED1'  # 点的颜色
        colors2 = '#DC143C'
        area = np.pi * 4 ** 2  # 点面积
        # 画散点图
        plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
        x1 = x1+1

    plt.grid(linestyle='-.')
    plt.show()




#(mod)tf来检测 def detect_person(image_np):
#
#     print("函数进")
#
#     #print(image_np)
#
#     PATH_TO_CKPT = 'colab_luck/frozen_inference_graph.pb'
#     PATH_TO_LABELS = 'data/label_map.pbtxt'
#     NUM_CLASSES = 1
#     confident = 0.5
#     detection_graph = tf.Graph()
#     with detection_graph.as_default():
#         od_graph_def = tf.GraphDef()
#         with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#             od_graph_def.ParseFromString(fid.read())
#             tf.import_graph_def(od_graph_def, name='')
#
#     label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#     categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)
#
#     with detection_graph.as_default():
#         with tf.Session(graph=detection_graph) as sess:
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#             #while cap.isOpened(): #用于判断视频流是否正常打开
#                 #ret, image_np = cap.read() #用于截取一帧图片
#             if len((np.array(image_np)).shape) == 0:
#                 print("This si image is error!")#break
#
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)   #BGR转换为RBG
#             image_np_expanded = np.expand_dims(image_np, axis = 0)
#             start = time.time()
#             (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
#             end = time.time()
#             print("Execution Time: ", end - start)
#             vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=4)
#                 #out.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#
#             s_boxes = boxes[scores > confident]
#             s_classes = classes[scores > confident]
#             s_scores = scores[scores > confident]
#             #print(s_boxes)                                  # 打印检测框坐标
#             #print(s_scores)                                 # 打印每个检测框的概率
#             #print(s_classes)                                # 打印检测框对应的类别
#             #print(category_index)                           # 打印类别的索引，其是一个嵌套的字典
#     cv2.destroyAllWindows()
#
#
#     return image_np

'''mod   final_score = np.squeeze(scores)
            count = 0                                        # 记录检测框个数
            for i in range(100):
                if scores is None or final_score[i] > 0.5:  # 显示大于50%概率的检测框
                    count = count + 1
            print("该帧存在的检测对象数目: ", count)       # 打印该帧检测的行人数目

            for i in range(count):
                #print(boxes[0][i])
                y_min = boxes[0][i][0]*im_height
                x_min = boxes[0][i][1]*im_width
                y_max = boxes[0][i][2]*im_height
                x_max = boxes[0][i][3]*im_width
                x_center = (x_max+x_min)/2                   # 计算每个帧各个对象的中心坐标
                y_center = (y_max+y_min)/2



                if 720 <= x_center <= 778:                              #判断每个对象是否位于盲区内
                    if 128 <=  y_center <= 203:
                        print("警报！盲区有行路人通过")
                        text = "dangers in blind spots!!!"
                        AddText = image_np.copy()
                        cv2.putText(AddText, text, (620, 115), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2) #在图像上添加报警文本
                        #image_np = np.hstack([image_np, AddText])
                        image_np = AddText

                        #cv2.imshow('text', AddText)
                        cv2.waitKey()                                   #不断刷新图像
                        cv2.destroyAllWindows()                       #释放窗口



                    #print()
                    #print(x_min,y_min,x_max,y_max)
                    #print()

            #out.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))        #将该帧写入到视频文件中



    #cap.release()
    #out.release()
'''




