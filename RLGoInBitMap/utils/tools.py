from os import path


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))# 基本路径对应
