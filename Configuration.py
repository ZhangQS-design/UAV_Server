
import argparse
class Configuration:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="UAVaction options")
        self.parser.add_argument("--version",
                                 type=str,
                                 help="version",
                                 default="0.1")
        self.parser.add_argument("--name",
                                 type=str,
                                 help="this method name. to get images and save depth eastimation images",
                                 default="deepEstRL0627load_model")
        self.parser.add_argument("--aim_angle",
                                 type=float,
                                 help="aim rotation, -180 to 180 ,0 is north, positve to east",
                                 default=90.0)
        self.parser.add_argument("--propotion",
                                 type=float,
                                 help="the propotion of relative depth to absolute depth, low to far,high to near",
                                 default=0.3376415797642299)
        self.parser.add_argument("--velocity",
                                 type=float,
                                 help="UAV pitch velocity meter per second ",
                                 default=0.15)
        self.parser.add_argument("--max_angle",
                                 type=float,
                                 help="max change yaw rotation per time",
                                 default=20.5)
        self.parser.add_argument("--max_deep",
                                 type=float,
                                 help="max deep estimation is reasonable you think ",
                                 default=8)
        self.parser.add_argument("--DE_model",
                                 type=str,
                                 help="depth estimation to use",
                                 default="drone",
                                 choices=["mono_640x192", "mono+stereo_640x192", "drone"])
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options





