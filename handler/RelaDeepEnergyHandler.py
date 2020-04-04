#  3/13 5/13  上 4 / 13 5/13
import math
import numpy as np

class RelaDeepEnergyHandler:
    def __init__(self,imageSize):
        self.action_bound = [-1, 1]
        self.action_dim = 1
        self.state_shape = 64
        self.angle = 81.9
        self.imageSize = imageSize
        self.thredhold = 0.57 # 0.5远或者无问题 0.55 开始代表远处或看不清的近处 0.6 开始存在危险  0.75 危险 0.85 可能高危
        self.thredholddanger = 0.75
        self.move = {"mPitch": 0, "mRoll": 0, "mYaw": 0, "mThrottle": 0, "gPitch": 0, "gRoll": 0,
                                    "gYaw": 0 , "handleImageName": ""}
        self.initrotation = -90
        self.rotation = self.initrotation
        self.countstep = 0


    def getRelaDeep(self, state, uprange, downrange,picpath):


        smoothLineSee = self.caculateObs(state, uprange, downrange)
        canGo = self.filter(smoothLineSee, len(smoothLineSee))
        haveRoad, directIndex = self.findGoDirect(canGo, len(canGo))
        #print("answer {} {}".format(haveRoad, directIndex))
        if haveRoad:
            angle = self.getAngle(directIndex, len(canGo))
        else:
            angle = -50 # 大转逃避


        if self.countstep % 10 == 0:
                self.rotation = self.initrotation
        else:
                self.rotation += angle
                if self.rotation > 180:
                    self.rotation -= 360
                if self.rotation < -180:
                    self.rotation += 360
                self.move['mPitch'] = 0
                self.move['mRoll'] = 0.15
                self.move['mYaw'] = self.rotation
                self.countstep += 1

        self.move['handleImageName'] = picpath
        print("Now action mYaw {} mPitch {} mRoll{} handleImageName{}".format(self.move['mYaw'], self.move['mPitch'], self.move['mRoll'],self.move['handleImageName']))
        return self.move

    def getRelaDeep2(self, state, uprange, downrange):

        action = [0]*self.action_dim #这里形成偏移角，-180 到 180
        smoothLineSee = self.caculateObs(state, 280, 460)
        canGo = self.filter2(smoothLineSee, len(smoothLineSee),0.75)
        haveRoad, directIndex = self.findGoDirect2(canGo, len(canGo))
        #print("answer {} {}".format(haveRoad, directIndex))
        if haveRoad and directIndex == 0:
            smoothLineSee = self.caculateObs(state, 320, 460)
            canGo = self.filter2(smoothLineSee, len(smoothLineSee),0.5)
            haveRoad, directIndex = self.findGoDirect2(canGo, len(canGo))

        if haveRoad:
            angle = self.getAngle(directIndex, len(canGo))

        else:
            angle = 50 # 大转逃避
        return angle


    def getAngle(self, directIndex,size):
        middle = size / 2
        if directIndex == 0:
            return 0
        if directIndex < 0:
            return (self.angle / 2) *  directIndex / middle
        if directIndex > 0:
            return (self.angle / 2) *  directIndex / middle


    def findGoDirect(self, canGo, size):
        canGoLen = size / 4
        dangerCountRight = 0
        dangerCountLeft = 0
        haveRoad = False
        for i in range(int(size * 3 / 8) , int(size * 5 / 8), 1):
            if canGo[i] == 1:
                dangerCountRight += 1
                dangerCountLeft += 1
        print("dangerCountRight {} dangerCountLeft {}".format(dangerCountRight, dangerCountLeft))
        if dangerCountRight == 0:
            haveRoad = True
            return haveRoad, 0
        leftstart = int(size * 3 / 8)
        leftend = int(size * 5 / 8)
        leftedge = 0
        rightstart = int(size * 5 / 8)
        rightend = int(size * 3 / 8)
        rightedge = size - 1
        p = 1
        while(leftstart - p >= leftedge and rightstart + p <= rightedge):
            if canGo[leftstart - p] == 1:
                dangerCountLeft += 1
            if canGo[leftend - p] == 1:
                dangerCountLeft -= 1

            if canGo[rightstart + p] == 1:
                dangerCountRight += 1
            if canGo[rightend + p] == 1:
                dangerCountRight -= 1

            if dangerCountLeft <= 0:
                haveRoad = True
                return haveRoad, -p

            if dangerCountRight <= 0:
                haveRoad = True
                return haveRoad, p

            p += 1

        return haveRoad, -1

    def findGoDirect2(self, canGo, size):
        canGoLen = size / 4
        dangerCountRight = 0
        dangerCountLeft = 0
        haveRoad = False
        for i in range(int(size * (1 - 0.42)/ 2) , int(size * (1 + 0.42)/ 2), 1):
            if canGo[i] == 1:
                dangerCountRight += 1
                dangerCountLeft += 1
        print("dangerCountRight {} dangerCountLeft {}".format(dangerCountRight, dangerCountLeft))
        if dangerCountRight == 0:
            haveRoad = True
            return haveRoad, 0
        leftstart = int(size * (1 - 0.42)/ 2)
        leftend = int(size * (1 + 0.42)/ 2)
        leftedge = 0
        rightstart = int(size * (1 + 0.42)/ 2)
        rightend = int(size * (1 - 0.42)/ 2)
        rightedge = size - 1
        p = 1
        while(leftstart - p >= leftedge and rightstart + p <= rightedge):
            if canGo[leftstart - p] == 1:
                dangerCountLeft += 1
            if canGo[leftend - p] == 1:
                dangerCountLeft -= 1

            if canGo[rightstart + p] == 1:
                dangerCountRight += 1
            if canGo[rightend + p] == 1:
                dangerCountRight -= 1

            if dangerCountLeft <= 0:
                haveRoad = True
                return haveRoad, -p

            if dangerCountRight <= 0:
                haveRoad = True
                return haveRoad, p

            p += 1

        return haveRoad, -1

    def filter(self, smoothLineSee, size,threhold):
        count = 0
        canGo = [0] * size
        for i in range(0, size):
            if smoothLineSee[i] > threhold:
                canGo[i] = 1
                count += 1
        #print("size {} count {}".format(size, count))
        return canGo

    def filter2(self, smoothLineSee, size,thredhold):
        count = 0
        canGo = [0] * size
        for i in range(0, size):
            if smoothLineSee[i] > thredhold:
                canGo[i] = 1
                count += 1
        #print("size {} count {}".format(size, count))
        return canGo


    def caculateObs(self, state, uprange, downrange):
        # 压缩转成线
        imageCompact = []
        for i in range(uprange, downrange):
            imageCompact.append(state[i][:])
        imageCompact = np.array(imageCompact)
        power = np.mean(imageCompact, axis=0)

        # 加基本平滑
        tail = state.shape[1] - 1
        smooth = power.copy()
        smooth[0] = (power[0] + power[1]) / 2
        smooth[tail] = (power[tail - 1] + power[tail]) / 2
        for i in range(2, tail - 1):
            smooth[i] = (power[i - 1] + power[i] + power[i + 1]) / 3

        return smooth




