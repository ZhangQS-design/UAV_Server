from flask import Flask,request
import cv2
import os
import threading
#import dlib
import json
import time
from pynput.keyboard import Listener
import math
from server.server.RelaDeepEnergyHandler import RelaDeepEnergyHandler
from server.server.utils import download_model_if_doesnt_exist, Normalize

handleImageCondition = threading.Condition()

image_center = [320,180]
images =[""]
track_result=[]
movement_sequance=[]
pre_box=None
basepath = os.path.dirname(os.path.abspath(__file__))
original_area = None
app = Flask(__name__)
move = {"mPitch": 0, "mRoll": 0, "mYaw": 0, "mThrottle": 0, "gPitch": 0, "gRoll": 0,
                                    "gYaw": 0 , "handleImageName": ""}

state = {'statePitch': 0, 'stateRoll': 0, 'stateYaw': 0, 'velocityX': 0, 'velocityY': 0, 'velocityZ': 0}

import datetime
con_image = threading.Condition()
con_track = threading.Condition()
con_result = threading.Condition()


imageNameLock=threading.Lock() #申请一把锁

'''
trackers={
    'KCF': cv2.TrackerKCF_create,
   # 'DSST': dlib.correlation_tracker,
    'TLD': cv2.TrackerTLD_create,
    'Boosting':cv2.TrackerBoosting_create,
    'CSRT':cv2.TrackerCSRT_create,
    'MedianFlow':cv2.TrackerMedianFlow_create,
    'MOSSE':cv2.TrackerMOSSE_create
}
'''
key ='DSST'

nameit = 'deepEstCon2'

indirectpath = os.path.join(basepath, 'static')
indirectpath = os.path.join(indirectpath, 'images')
indirectpath = os.path.join(indirectpath, nameit)
if not os.path.exists(indirectpath) :
        os.mkdir(indirectpath)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload',methods=['POST', 'GET'])
def upload():
    return "ok"

lastUploadImageTime = datetime.datetime.now()

@app.route('/uploadState',methods=['POST', 'GET'])
def uploadState():
    global rotation
    global state
    droneState = request.json
    print(droneState)
    state = droneState
    rotation = droneState['stateYaw']
    return "ok"



@app.route('/uploadImage',methods=['POST', 'GET'])
def uploadImage():
    f = request.files["files"]
    global images
    global movement_sequance
    global nameit
    filename = f.filename
    global basepath
    global lastUploadImageTime
    global handleImageCondition
    indirectpath = os.path.join(basepath, 'static')
    indirectpath = os.path.join(indirectpath, 'images')
    it = os.path.join(indirectpath, nameit)
    if not os.path.exists(it) :
        os.mkdir(it)
    upload_path = os.path.join(indirectpath, nameit,filename)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    #print(upload_path)

    imageNameLock.acquire()
    f.save(upload_path)
    #images.append(upload_path)
    images[0] = upload_path
    imageNameLock.release()
    with handleImageCondition:
        handleImageCondition.notify()

    endtime = datetime.datetime.now()
    internaltime = (endtime - lastUploadImageTime).total_seconds()
    lastUploadImageTime = endtime
    print("internal time of UploadImage {}".format(internaltime))

    #if con_image.acquire():
        # 当获得条件变量后
    #con_image.notify()
    #con_image.release()

    return filename


@app.route('/getcontrol',methods=['POST', 'GET'])
def getcontrol():
    #print(move)
    return json.dumps(move)


def load_images():
    global images
    global pre_box
    global move
    global original_area
    global handleImageCondition
    start_tracking = False
    first = True
    conter = 0
    tracker=None
    print("start handle image")
    while True:
        '''
       # if con_image.acquire():
            # 当获得条件变量后
              # if con_image.acquire():
            # 当获得条件变量后
            if images[0]=="":
                # 图像缓存中没有图片
                #print("it has been zero")
                #con_image.wait()
                time.sleep(0.1)
                print("陷入等待")
                # 该进程处于wait状态

            else:
                handleImageCondition.wait()
                imageNameLock.acquire()
                if images[0] == "":
                    imageNameLock.release()
                    continue
                else:
                    picpath = images[0]
                    images[0] = ""

                img = cv2.imread(picpath)
                img = cv2.resize(img, (640, 360))
                print(conter)
                print(picpath)
                #cv2.imshow('img', img)
                #os.remove(picpath)
                del img
                #if cv2.waitKey(1) & 0xff==ord('s'):
                #    first = Trueddddddddddddddaaaaaaadddddddddddddddddddaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaauuuuuwaaaaaaaaa
                 #   start_tracking = ~start_tracking
                # 通过notify方法通知上传进程
                #con_image.notify()
                conter+=1
        # 条件变量释放
        #con_image.release()
'''

#from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import datetime

import torch
from torchvision import transforms, datasets

import server.server.networks as networks
from server.server.layers import disp_to_depth
from server.server.utils import download_model_if_doesnt_exist
initrotation = -90
rotation = initrotation

countstep = 0
def load_images_use_DE():
    """Function to predict for a single image or folder of images
    """
    global images
    global  rotation
    global pre_box
    global move
    global original_area
    global basepath
    global  countstep
    indirectpath = os.path.join(basepath, 'static')
    indirectpath = os.path.join(indirectpath, 'images')
    indirectpath = os.path.join(indirectpath, nameit)

    agent = RelaDeepEnergyHandler((768,1024))

    if torch.cuda.is_available() :#cuda可以做到1秒5张以上，而cpu大概是1秒1张
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")
        print("use cpu")

    model_name = "mono+stereo_640x192"

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES

    if os.path.isdir(indirectpath):
        # Searching folder for images
        #paths = glob.glob(os.path.join(indirectpath, '*.{}'.format("jpg")))
        output_directory = indirectpath
    else:
        raise Exception("Can not find args.image_path: {}".format(indirectpath))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        idx = 0
        print("图片处理加载完成")
        while True:
            with handleImageCondition:
                handleImageCondition.wait()
                imageNameLock.acquire()
                if images[0] == "":
                    imageNameLock.release()
                    continue
                else:
                    picpath = images[0]
                    images[0] = ""
                    imageNameLock.release()

                starttime = datetime.datetime.now()
                """
                image_path = images[0]
                images.remove(image_path)
                """

                input_image = pil.open(picpath).convert('RGB')
                image_path = picpath



                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
                output_name = os.path.splitext(os.path.basename(image_path))[0]
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()

                matrix = Normalize(disp_resized_np)
                angle = agent.getRelaDeep2(matrix, 290, 470)
                print("angle {}".format(angle))

                vmax = np.percentile(disp_resized_np, 95) # 锁掉最大的max
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) # 等比例缩放 最小无限 Normlize是用来把数据标准化(归一化)到[0,1]这个期间内,vmin是设置最小值, vmax是设置最大值
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')# mapper? cm? 归一化后配色方案
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                im = pil.fromarray(colormapped_im)# 转图过程

                name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)
                endtime = datetime.datetime.now()
                internaltime = (endtime - starttime).total_seconds()
                print("  need time {} Processed {:d}  images - saved prediction to {}".format(internaltime ,
                idx + 1, name_dest_im))
                global initrotation
                if countstep % 10 == 0:
                    rotation = initrotation
                else:
                    rotation += angle
                if rotation > 180:
                    rotation -= 360
                if rotation < -180:
                    rotation += 360
                move['mPitch'] = 0
                move['mRoll'] = 0.15
                move['mYaw'] = rotation
                countstep += 1



                move['handleImageName'] = picpath
                print("Now action mYaw {} mPitch {} mRoll{} handleImageName{}".format(move['mYaw'], move['mPitch'], move['mRoll'],move['handleImageName']))





def area(b):
    return (b[2]-b[0])*(b[3]-b[1])

def center(b):
    return (b[2]+b[0])/2,(b[3]+b[1])/2

def infer_track_result(p_b,b):
    return area(b),area(p_b),center(p_b),center(b)



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

# 生产者消费者
# 1. 线程首先acquire一个条件变量
# 2. 判断条件：  如果条件不满足则wait；
#               如果条件满足，进行一些 处理改变条件后，通过notify方法通知其他线程，
#               其他处于wait状态的线程接到通知后会重新判断条件
if __name__ == '__main__':
    print(basepath)
    print("show info")

    thread3 = myThread(3,"key",listen)
    thread2 = myThread(2,"show", load_images_use_DE)#load_images)
    thread1 = myThread(1,"flask",app.run,'0.0.0.0')

    #thread3.start()
    thread1.start()
    thread2.start()

