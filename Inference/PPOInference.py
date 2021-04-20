import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    def __init__(self, actiondim=1):
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
        parser.add_argument('--version', default="4.2.4.1", metavar='G',
                            help='version')

        args = parser.parse_args()
        path = os.path.join(assets_dir(), args.version)
        print(path)

        randomSeed = 2
        render = False
        state_dim = 64 + 11  # env.observation_space.shape[0]#[0]
        running_state = ZFilter((state_dim,), clip=5)
        """define actor and critic"""
        policy_net =Policy(75, 2)#DiscretePolicy(75,5) #Policy(75, 4)
        value_net = Value(75)
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

        self.lastRotation = 0
        self.lastLeftRightFeel = [8, 8]
        self.lastaction = [0, -0.5]
        self.lasttime = datetime.datetime.now()

        print("init succ")

    def inference(self, imgstate, rotation, aimRotation, time):
        deepfeel = self.caculateObs(imgstate)

        state = self.getState(deepfeel, rotation, aimRotation)
        action, value = self.agent.predictTakeValue(state)
        info = f"time {time} action {action} critic {value} state {state[64:]} deepfeel avg {np.mean(deepfeel)} value {deepfeel} "
        info2 = f"action {action} critic {value} state {state[64:]}"
        print(info2)
        self.persistence.saveTerminalRecord("stateaction", info)
        self.lastaction = action
        return action

    def inference4(self, imgstate, rotation, aimRotation, time):#仅往前走或者仅拐弯
        deepfeel = self.caculateObs(imgstate)

        state = self.getState(deepfeel, rotation, aimRotation)
        action, value = self.agent.predictTakeValue(state)
        info = f"time {time} action {action} critic {value} state {state[64:]} deepfeel avg {np.mean(deepfeel)} value {deepfeel} "
        info2 = f"action {action} critic {value} state {state[64:]}"
        print(info2)
        self.persistence.saveTerminalRecord("stateaction", info)
        self.lastaction = action
        if action[1] > 0:
            action[0] = 0

        return action

    def inference2(self, imgstate, rotation, aimRotation, time):#分开左右
        deepfeel = self.caculateObs(imgstate)

        state = self.getState(deepfeel, rotation, aimRotation)
        action, value = self.agent.predictTakeValue(state)
        info = f"time {time} action {action} critic {value} state {state[64:]} deepfeel avg {np.mean(deepfeel)} value {deepfeel} "
        info2 = f"action {action} critic {value} state {state[64:]}"
        print(info2)
        self.persistence.saveTerminalRecord("stateaction", info)
        self.lastaction = action
        if action[1] <= 0:
            if action[2] > action[3]:
                action[0] = 1
            else:
                action[0] = -1
        else:
            action[0] /= 3
        return action

    def inference3(self, imgstate, rotation, aimRotation, time):# 完全离散化
        deepfeel = self.caculateObs(imgstate)

        state = self.getState(deepfeel, rotation, aimRotation)
        action, value = self.agent.predictTakeValue(state)
        info = f"time {time} action {action} critic {value} state {state[64:]} deepfeel avg {np.mean(deepfeel)} value {deepfeel} "
        info2 = f"action {action} critic {value} state {state[64:]}"
        print(info2)
        self.persistence.saveTerminalRecord("stateaction", info)
        c = -0.5
        if action >= 1 and action < 4:
            c = 0.5
        actionsingle = action
        action = [action, c]
        self.lastaction = action
        if actionsingle < 1:
            action[0] = -1
            action[1] = -0.5
        elif actionsingle >= 4:
            action[0] = 1
            action[1] = -0.5
        elif actionsingle >= 1 and actionsingle < 2:
            action[0] = -0.5
            action[1] = 0.5
        elif actionsingle >= 3 and actionsingle < 4:
            action[0] = 0.5
            action[1] = 0.5
        else:
            action[0]= 0
            action[1] =0.5

        return action




    def getState(self, deepfeel, rotation, aimRotation):  # 极端诡异和空格于制表符显示问题
        rotation = math.radians(rotation)
        aimRotation = math.radians(aimRotation)

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

        timenow = datetime.datetime.now()
        internaltime = (timenow - self.lasttime).total_seconds()

        other = [rotation, xDirect, yDirect, self.lastaction[0], self.lastaction[1], alphadirect, alphacos,
             self.lastRotation, self.lastLeftRightFeel[0], self.lastLeftRightFeel[1], internaltime]
        nextstate = []
        # print(f"other {other}")
        for i in deepfeel:
            nextstate.append(i)
        for i in other:
            nextstate.append(i)
        self.lastRotation = alphadirect
        # self.lastRotation = rotation
        self.lasttime = timenow
        self.lastLeftRightFeel = [nextstate[0], nextstate[63]]
        return nextstate


    def caculateObs(self, state, uprange=24, downrange=38):  # 压缩转成线
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
