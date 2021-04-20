import os
import datetime

DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static/log'))


class Persistence:
    def __init__(self, methodName):
        self.terminalRecordFileName = methodName
        time = datetime.datetime.now()
        self.saveTerminalRecord("_methodStartInfo", f"start doing time {time}")

    def saveTerminalRecord(self, topic, info):
        fileName = self.terminalRecordFileName + topic + ".txt"
        file = os.path.join(DIR, fileName)
        with open(file, 'a') as file_obj:
            file_obj.write(info + '\n')
