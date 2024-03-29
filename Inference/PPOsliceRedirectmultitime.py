import os

import sys
import pickle
import time
import math
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import queue
from server.InferenceUtils import *
from server.RLGoInBitMap.Agent2 import DroneAgent
import argparse
from server.Persistence import Persistence
import numpy as np

import datetime
from server.RLGoInBitMap.models.mlp_policy import Policy
from server.RLGoInBitMap.models.mlp_critic import Value
from server.RLGoInBitMap.models.mlp_policy_disc import DiscretePolicy

class InferenceModel():
    def __init__(self, aimRotation,actiondim=1):
        print("init InferenceModel")
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        # device = torch.device('cuda', index=0)  # if torch.cuda.is_available() else
        device = torch.device('cpu')
        # if torch.cuda.is_available():
        #	torch.cuda.set_device(0)


        parser = argparse.ArgumentParser(description='PyTorch PPO example')
        parser.add_argument('--env-name', default="continueRealEnvPpo", metavar='G',
                            help='name of the environment to run')
        parser.add_argument('--version', default="4.3.1.8.9", metavar='G',
                            help='version')

        args = parser.parse_args()
        path = os.path.join(assets_dir(), args.version)
        print(path)

        randomSeed = 2
        render = False
        state_dim = 64 + 12 +15 + (64 +2+2)* 3  # env.observation_space.shape[0]#[0]
        running_state = ZFilter((state_dim,), clip=5)
        """define actor and critic"""
        policy_net =Policy(state_dim, 3)#DiscretePolicy(75,5) #Policy(75, 4)
        value_net = Value(state_dim)
        policy_net.load_state_dict(torch.load(os.path.join(path, 'policy_net_{}_ppo.pth'.format(args.env_name)), 'cpu'))
        value_net.load_state_dict(torch.load(os.path.join(path, 'value_net_{}_ppo.pth'.format(args.env_name)), 'cpu'))

        # policy_net = torch.load(os.path.join(path, 'policy_net_{}_ppo.pth'.format(args.env_name)), 'cpu')
        # value_net = torch.load(os.path.join(path, 'value_net_{}_ppo.pth'.format(args.env_name)), 'cpu')
        running_state, saveavgreward = pickle.load(
            open(os.path.join(path, 'running_state_{}_ppo.p'.format(args.env_name)), "rb"))
        print("get reward {}".format(saveavgreward))
        policy_net.to(device)
        value_net.to(device)

        self.persistence = Persistence("real_0515_" + args.version)

        """create agent"""
        self.agent = DroneAgent(policy_net, value_net, device, running_state=running_state, render=render,
                                num_threads=1)

        self.lastRotation = aimRotation
        self.lastLeftRightFeel = [8, 8]
        self.lastaction = [0, -0.5,0]
        self.lasttime = datetime.datetime.now()
        self.forceWallMenory =0
        self.aimRotaion = aimRotation
        self.aimRotation = aimRotation
        self.finalAimRotaion =aimRotation
        self.stoptime = 0
        self.stoplong = 5
        self.lastalphadirect = 0
        self.lastalphacos = 0.5

        self.stopfrequency = 0
        self.stopfrequencylong = 5
        self.saveTimes =3
        self.redirectNum = 0
        # 历史内容暂存
        self.saveDeepFeels = queue.Queue()
        self.savelastActions = queue.Queue()
        self.savelastAngles = queue.Queue()
        for i in range(3):
            temp = [8] * 64  #初始观感长预设为8
            self.saveDeepFeels.put(temp)
            self.savelastActions.put([0,0])
            self.savelastAngles.put([0,1])

        self.tempPoCache = queue.Queue()

        for i in range(20):
            temp = [-1000,-1000]
            self.tempPoCache.put(temp)
        print("init succ")

    def inference(self, imgstate, rotation, aimRotation, time):

        deepfeel = self.caculateObs(imgstate)

        state = self.getState(deepfeel, rotation,self.lastaction[2])
        action, value = self.agent.predictTakeValue(state)
        info = f"time {time} action {action} critic {value} state {state} deepfeel avg {np.mean(deepfeel)} value {deepfeel} "
        info2 = f"action {action} critic {value} state {state[0:64]}"
        if action[1] > 0:
            self.stoptime = 0
        else:
            self.stoptime += 1
            self.stopfrequency +=1
        print(info2)
        self.persistence.saveTerminalRecord("stateaction", info)
        self.lastaction = action.copy()

        if self.stopfrequency > 0:
            self.stopfrequency -= 0.5
        if self.forceWallMenory > 0:
            pass
        return action


    def sliceWindow(self, deepfeel,sliceSize=8, proValue=2, threshold=1.8):
        go = int(sliceSize / 2)
        sliceRes = []
        temp = 0
        for i in range(0, len(deepfeel) - sliceSize + 1, go):
            temp = 0
            for j in range(0, sliceSize):
                if deepfeel[i + j] < threshold:
                    if temp == 0:
                        temp = 10
                    temp = (threshold - deepfeel[i + j]) * proValue
            sliceRes.append(temp)
        return sliceRes


    def judgeForceWall(self, deepfeel, alphacos, max=6.5, avgmax=3.2,twoavgmax = 1.8,smallthreshold = 0.55):# 待修改
        #看局部最优
        tempbest = False
        if self.stopfrequency >self.stopfrequencylong:
            tempbest = True



        if alphacos < 0.7:
            return False
        totalLength = 0
        maxLength = 0
        smallnum = 0
        for i in deepfeel:
            if i > maxLength:
                maxLength = i
            if i < smallthreshold:
                smallnum += 1
            totalLength += i
        avgLength = totalLength / len(deepfeel)
        if avgLength < avgmax and maxLength < max or avgLength < twoavgmax or smallnum > 5 or tempbest:
            self.forceWallMenory = random.randint(15, 20)
            return True
        if self.stoptime > self.stoplong:
            self.forceWallMenory = random.randint(15, 20)
            return True
        return False

    def chTarget(self, direct):
        print("更改临时目标")
        pos = np.array([self.drone.center.x, self.drone.center.y])
        '''
        aimDirectX = self.aim[0] - pos[0]
        aimDirectY = self.aim[1] - pos[1]
        if direct >= 0:
            self.aim[0] = aimDirectY + pos[0]
            self.aim[1] = (-1) * aimDirectX + pos[1]
        else:
            self.aim[0] = (-1) * aimDirectY + pos[0]
            self.aim[1] = aimDirectX + pos[1]
                c = round(math.sin(math.radians(i)),6)
    d = round(math.cos(math.radians(i)),6)
        '''
        aimDirectX = self.aim[0] - pos[0]
        aimDirectY = self.aim[1] - pos[1]

        if direct > 1:
            direct = 1
        elif direct < -1:
            direct = -1
        angle = direct * 80
        sinthlta = 1
        costhlta = 0#round(math.cos(math.radians(abs(angle))),6)
        if direct >= 0:
            self.aimRotation -= 90
        else:
            self.aimRotation -= 90
        if self.aimRotation < -180:
            self.aimRotation +=360
        elif self.aimRotation > 180:
            self.aimRotation -=360



    def getState(self, deepfeel, rotation,direct):  # 极端诡异和空格于制表符显示问题
        rotation = math.radians(rotation)

        sliceRes = self.sliceWindow(deepfeel)
        aimRotation = math.radians(self.aimRotation)
        xDirect = round(math.cos(rotation), 6)
        yDirect = round(math.sin(rotation), 6)
        aimDirectX = round(math.cos(aimRotation), 6)
        aimDirectY = round(math.sin(aimRotation), 6)

        alphadirect = aimDirectX * yDirect - aimDirectY * xDirect
        alphacos = aimDirectX * xDirect + aimDirectY * yDirect  # 直接用目标与行进方向夹角sin cos作为状态

        if alphacos < 0 and alphadirect > 0:
            alphadirect = 1
        if alphacos < 0 and alphadirect < 0:
            alphadirect = -1
        '''
        judgeState = self.judgeForceWall(deepfeel, alphacos)
        if judgeState == True:
            self.chTarget(direct)
            aimRotation = math.radians(self.aimRotation)
            xDirect = round(math.cos(rotation), 6)
            yDirect = round(math.sin(rotation), 6)
            aimDirectX = round(math.cos(aimRotation), 6)
            aimDirectY = round(math.sin(aimRotation), 6)

            alphadirect = aimDirectX * yDirect - aimDirectY * xDirect
            alphacos = aimDirectX * xDirect + aimDirectY * yDirect # 直接用目标与行进方向夹角sin cos作为状态

            if alphacos < 0 and alphadirect > 0:
                alphadirect = 1
            if alphacos < 0 and alphadirect < 0:
                alphadirect = -1
        '''
        free = 0
        if self.forceWallMenory > 0:
            free = 5 + self.forceWallMenory
        stop = 0
        if self.stoptime >= self.stoplong:
            stop = 5


        timenow = datetime.datetime.now()
        internaltime = (timenow - self.lasttime).total_seconds()

        other = [xDirect, yDirect, self.lastaction[0], self.lastaction[1], alphadirect, alphacos,
            self.lastLeftRightFeel[0], self.lastLeftRightFeel[1],free, stop,self.lastalphadirect,self.lastalphacos]
        nextstate = []
        # print(f"other {other}")
        for i in deepfeel:
            nextstate.append(i)
        for i in other:
            nextstate.append(i)
        for i in sliceRes:
            nextstate.append(i)

        for i in range(self.saveTimes):
            queue = self.saveDeepFeels.get()
            for j in queue:
                nextstate.append(j)
            self.saveDeepFeels.put(queue)

        for i in range(self.saveTimes):
            queue = self.savelastActions.get()
            for j in queue:
                nextstate.append(j)
            self.savelastActions.put(queue)

        for i in range(self.saveTimes):
            queue = self.savelastAngles.get()
            for j in queue:
                nextstate.append(j)
            self.savelastAngles.put(queue)

        self.savelastAngles.get()
        self.savelastAngles.put([alphadirect,alphacos])


        self.lasttime = timenow
        self.lastLeftRightFeel = [nextstate[0], nextstate[63]]



        self.saveDeepFeels.get()
        self.saveDeepFeels.put(deepfeel.copy())

        self.savelastActions.get()
        self.savelastActions.put([self.lastaction[0],self.lastaction[1]])


        if self.forceWallMenory > 0:
            self.forceWallMenory -= 1
            if self.forceWallMenory <= 0:
                self.aimRotaion = self.finalAimRotaion.copy()
            print("曾经感受到墙")



        self.lastalphadirect = alphadirect
        self.lastalphacos = alphacos


        return nextstate


    def caculateObs(self, state, uprange=25, downrange=36):  # 压缩转成线
        imageCompact = []
        for i in range(uprange, downrange):
            imageCompact.append(state[i][:])
        imageCompact = np.array(imageCompact)
        power = np.min(imageCompact, axis=0)

        # print(power)
        for i in range(len(power)):
            if power[i] > 8:
                power[i] = 8
            elif power[i] < 0.2 and power[i] > 0.000001:
                power[i] = 0.2
            elif power[i] == 0:
                power[i] = 7
        # print(power)
        return power
