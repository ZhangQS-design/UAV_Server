

from flask import Flask,request
import os
import threading
import cv2
import json
import time
import numpy as np
from server.sht import *
from pynput.keyboard import Listener

from server.Persistence import Persistence
import logging
from server.Configuration import Configuration
from server.CheckSecure import CheckSecure
from server.config import Socket_UDP

# ------无需配置初始化-----
original_area = None
app = Flask(__name__)
handleImageCondition = threading.Condition()
uploadFinish = threading.Condition()

image_center = [320,180]
images =[""]
img_4 = []
data_lab = []

track_result=[]
movement_sequance=[]
pre_box=None

import datetime
con_image = threading.Condition()
con_track = threading.Condition()
con_result = threading.Condition()
imageNameLock=threading.Lock() #申请一把锁
lastUploadImageTime = datetime.datetime.now()
lastHandleImageTime = datetime.datetime.now()
startgotinfo = True
# 深度估计
countstep = 0
# 控制和回传信号相关
getcontrolindex = 0
chTimes = 1
time_new = 0
time_old = 0
time_old1 = 0
tempchTimes = 0
uploadfinish =False
# 是否需要输出网络连接信息，如果需要请删掉
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
# 乱飘问题解决 未成功
rollTemp =0
rollpro = 0.05
# 内容初始化
move = {"mPitch": 0, "mRoll": 0, "mYaw": 0, "mThrottle": 0, "gPitch": 0, "gRoll": 0,"gYaw": 0 , "handleImageName": "","chTimes":1}
state = {'statePitch': 0, 'stateRoll': 0, 'stateYaw': 0, 'velocityX': 0, 'velocityY': 0,'curTimes':0}


# ------配置初始化-----
'''
# 这里换可能更方便
aimRotation = 90
pitchVelo = 0.12
rotationAngle = 20.5
maxdeep = 8
propotion = 0.3576415797642299 #mean 0.2575861972 29
version = "0.7"
nameImagesDir = 'deepEstRL0627load_model' + version
model = "drone"#mono+stereo_640x192"
'''
def loadConfig():
    config = Configuration()
    args = config.parse()
    global version
    global nameImagesDir
    global aimRotation
    global pitchVelo
    global rotationAngle
    global maxdeep
    global propotion
    global model
     # ------配置初始化-----
    aimRotation = args.aim_angle
    pitchVelo = args.velocity
    rotationAngle = args.max_angle
    maxdeep = args.max_deep
    propotion = args.propotion #mean 0.2575861972 29
    version =  args.version
    nameImagesDir = args.name + version
    model = args.DE_model#mono+stereo_640x192"

loadConfig()
rotation =aimRotation

# 路径
model_name = model
model_path = os.path.join(f"../static/DEmodels/", model_name)

indirectpath = f"../static/images/"
#(mod)indirectpath = os.path.join(indirectpath, nameImagesDir)
#if not os.path.exists(indirectpath) :
#        os.mkdir(indirectpath)

# 简易分离日志
persistence = Persistence("serverInfo" + nameImagesDir)
# 安全解决
checkSecure = CheckSecure(nameImagesDir)
global time_count
time_count = 0
global detect_path
detect_path = 0

# ------具体服务------
# 连接测试
@app.route('/')
def hello_world():
    print("初始化连接成功")
    global move
    global aimRotation
    move = {"mPitch": 0, "mRoll": 0, "mYaw": aimRotation, "mThrottle": 0, "gPitch": 0, "gRoll": 0,
                                    "gYaw": 0 , "handleImageName": "","chTimes":1}
    global startgotinfo
    startgotinfo = True
    global uploadfinish
    uploadfinish = True
    checkSecure.reset()

    return 'Hello World!'

@app.route('/upload',methods=['POST', 'GET'])
def upload():
    print("上传测试初始化连接成功")
    global move
    global aimRotation
    move = {"mPitch": 0, "mRoll": 0, "mYaw": aimRotation, "mThrottle": 0, "gPitch": 0, "gRoll": 0,
                                    "gYaw": 0 , "handleImageName": "","chTimes":1}
    global startgotinfo
    startgotinfo = True
    global uploadfinish
    uploadfinish = True
    checkSecure.reset()
    return "ok"


# 状态上传
# @app.route('/uploadState',methods=['POST', 'GET'])
# def uploadState():
#     global rotation
#     global state
#     global uploadfinish
#     global rollTemp
#     global startgotinfo
#     global chTimes
#     droneState = request.json
#     print(droneState)
#     state = droneState
# #    rotation = droneState['stateYaw']
#  #   rollTemp = droneState['velocityY']
#
#     persistence.saveTerminalRecord("dronestate","time " + str(datetime.datetime.now()) +" " +str(droneState))
#     global tempchTimes
#
#     global checkSecure
    # danger,info = checkSecure.checkState(droneState)
    # if danger:
    #     move = info
    #
    # if droneState['curTimes'] >tempchTimes :
    #     startgotinfo = True
    #     tempchTimes = droneState['curTimes']
    #
    # if droneState['curTimes'] >= chTimes : #产生新行为时
    #     if startgotinfo:
    #         time.sleep(0.5) #等待drone下一次行动完成
    #         uploadfinish = True
    #         startgotinfo = False
    #     else:
    #         pass

 #   return "ok"


# 图片上传
@app.route('/uploadImage',methods=['POST', 'GET'])
def uploadImage():

    global handleImageCondition
    global img_4
    global time_count
    global time_new
    global time_old
    global time_old1

    f = request.files["files"]
    filename = f.filename
    time_count += 1
    if time_count / 1 == time_count:
        img_2 = f.read()
        # print(type(f.read()))   #类型为字节类型
        img_3 = np.frombuffer(img_2, dtype=np.uint8)
        handleImageCondition.acquire()
        img_4 = cv2.imdecode(img_3, 1)
        # print(img_4)
        handleImageCondition.notify()
        handleImageCondition.release()

    # 图片下载速率
    # time_now = time.time()
    # if time_count % 100 == 1:
    #     time_old1 = time_now
    # if time_count % 100 == 0:
    #     print('--------------')
    #     print('--------------')
    #     print('--------------')
    #     print('----')
    #     print(time_now - time_old1)
    #     print('----')
    #     print('--------------')
    #     print('--------------')
    #     print('--------------')

    # f.save(upload_path)         # 存储视频流
    # 上传200帧的时延
    #end = time.time()
    #data_lab.append(end - start)
    #print("Execution Time UAV to PC upload", end - start)
    if time_count == 200:
        print("**********")
        print(data_lab)
        print("**********")
    return filename


def per_detect():
    while (1):
        global handleImageCondition
        global img_4
        #detect_path = uploadImage()
        handleImageCondition.acquire()
        handleImageCondition.wait()
        #print('----')
        #print(img_4)
        #image_np1 = cvdetect_person(img_4)
        #cv2.imshow('Blind Area Monitoring', image_np1)
        #cv2.waitKey(1)
        Socket_UDP(img_4)

        handleImageCondition.release()
    #cv2.destroyAllWindows()

    # time_now = time.time()            #图片下载速率
    # if time_count == 1:
    #     time_old1 = time.time()
    # if time_count == 100:
    #     print('--------------')
    #     print('--------------')
    #     print('--------------')
    #     print('----')
    #     print(time_now - time_old1)
    #     print('----')
    #     print('--------------')
    #     print('--------------')
    #     print('--------------')
    #     count = 0
    # upload_path = ('test_images/')
    # image_np1 = cv2.imread(upload_path + upload_path)
    # out = cv2.VideoWriter('out_video/output_video.mp4', -1, 10, (image_np1.shape[1], image_np1.shape[0]))
    # image_np1 = detect_person(image_np1)
    # out.write(cv2.cvtColor(image_np1, cv2.COLOR_RGB2BGR))
    #cv2.imshow('hello',img)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()
    #images.append(upload_path)



#控制信号获取
@app.route('/getcontrol',methods=['POST', 'GET'])
def getcontrol():
    #print(move)
    global move
    global move
    global move
    global getcontrolindex
    getcontrolindex += 1
    persistence.saveTerminalRecord("getcontrol","time " + str(datetime.datetime.now()) +" index " +str(getcontrolindex))
    flag, action = checkSecure.checkControl(move)
    if flag:
        move = action.copy()
    return json.dumps(move)
#

# 键盘修改控制信号，默认不开
def press(key):
      print(key.char)
      if key.char == 'j':
          move['mPitch'] -= 0.1
          print("pitch down{}".format(move['mPitch']))
      if key.char == 'l':
          move['mPitch'] += 0.1
          print("pitch up{}".format(move['mPitch']))
      if key.char == 'k':
          move['mThrottle'] -= 0.5
          print("high down{}".format(move['mThrottle']))
      if key.char == 'i':
          move['mThrottle'] += 0.5
          print("high up{}".format(move['mThrottle']))
      if key.char == 'p':
          move['mYaw'] = 0
          move['mPitch'] = 0
          move['mRoll'] = 0
          move['mThrottle'] = 0
          move['handleImageName'] = ""
          print("reset")


def listen():
    with Listener(on_press = press) as listener:
        listener.join()


class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, func, param=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.func = func
        self.param = param
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        #print(self.func)
        if self.param==None:
            self.func()
        else:
            self.func(self.param)


if __name__ == '__main__':

    print("show info")

    #thread3 = myThread(3,"key",listen)
    #thread2 = myThread(2,"show", load_images_use_DE)
    #thread1 = myThread(1, "flask", app.run(host='0.0.0.0',port=5000))
    thread1 = myThread(1,"flask",app.run,'0.0.0.0')
    thread4 = myThread(4,"per_detect",per_detect)


    #thread3.start() # 键盘直接调整控制信号控制UAV
    thread1.start()
    thread4.start()



    #测试
    # upload_path = "F://DJI_SDK//copy//UAVServer//server//server//test_images//6601.jpg"
    # image_np1 = cv2.imread(upload_path)
    # image_np2 = cvdetect_person(image_np1)
    # cv2.imshow('opencv-dnn-ssd-detect', image_np2)
    # cv2.waitKey(5000)


