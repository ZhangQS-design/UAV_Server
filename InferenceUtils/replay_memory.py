from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):# 增加
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None: #全取
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size) #随机采样
            return Transition(*zip(*random_batch))

    def append(self, new_memory): #叠加
        self.memory += new_memory.memory

    def __len__(self): # 核心类属性
        return len(self.memory)
