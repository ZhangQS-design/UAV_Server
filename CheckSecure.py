from server.Persistence import Persistence
import datetime
import os

class CheckSecure:
    def __init__(self, methodName):
        self.persistence = Persistence("SecureProblem" + methodName)
        self.state = {'statePitch': 0, 'stateRoll': 0, 'stateYaw': 0, 'velocityX': 0, 'velocityY': 0,'curTimes':0}
        self.move = {"mPitch": 0, "mRoll": 0, "mYaw": 0, "mThrottle": 0, "gPitch": 0, "gRoll": 0,"gYaw": 0 , "handleImageName": "","chTimes":1}

        self.sendStopAction = False


    def check(self,input1,input2):
        self.checkSingle(input1)

        return self.sendStopAction, self.state


    def checkState(self,droneState):
        if self.sendStopAction:
            print("曾经存在异常")
        if abs(droneState['statePitch']) > 20 or abs(droneState['stateRoll']) > 20 or droneState['velocityY']>5 or droneState['velocityX']>5 :
            print(f"飞行状态告警 {droneState}")
            self.dangerAction(f"状态异常 {droneState}")
        return self.sendStopAction, self.move


    def checkControl(self,droneControlState):
        dangerMove= 3.5
        if self.sendStopAction:
            print("曾经存在异常")

        if droneControlState['mPitch'] > dangerMove or droneControlState['mRoll'] > dangerMove or abs(droneControlState['mYaw']) >360 :
            print(f"控制信号飞行告警 {str(droneControlState)}")
            self.dangerAction(f"控制信号异常 {str(droneControlState)}")
        return self.sendStopAction, self.move


    def checkSingle(self, input):
        danger = False
        dangerInfo = ""
        # 检查代码
        time = datetime.datetime.now()
        # 记录问题
        if danger:
            self.dangerAction(dangerInfo)



    def reset(self):
        self.sendStopAction = False


    def dangerAction(self,dangerInfo):
        time = datetime.datetime.now()
        self.persistence.saveTerminalRecord("_methodStartInfo", f"dangertime {str(time)}" )
        print(f"{time} 危险告警！ {dangerInfo}")
        self.sendStopAction = True


