#coding:utf-8
from RLGoInBitMap.envs.BitEnvironment import BitEnvironment
import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

env =  BitEnvironment()

# -*- coding: utf-8 -*-

import torch

import torch
#import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from RLGoInBitMap.models.mlp_rnn_critic  import Value
from RLGoInBitMap.models.mlp_rnn_policy import Policy


if __name__ == "__main__":
	#print(env.action_space)



	torch.manual_seed(1)

	value_net = Value(33)

	policy_net = Policy(33,2)

	dtype = torch.float32
	torch.set_default_dtype(dtype)
	# device = torch.device("cpu") #if torch.cuda.is_available() else
	# torch.device('cpu')
	# if torch.cuda.is_available():
	#    torch.cuda.set_device(args.gpu_index)
	device = torch.device('cuda', index=0)


	policy_net.to(device)
	value_net.to(device)
	#value_net.cuda()

	#lstm = nn.LSTM(3, 3)  # 输入单词用一个维度为3的向量表示, 隐藏层的一个维度3，仅有一层的神经元，
	# 记住就是神经元，这个时候神经层的详细结构还没确定，仅仅是说这个网络可以接受[seq_len,batch_size,3]的数据输入
	#print(lstm.all_weights)

	inputs = torch.randn(1,2, 33)
	# 构造一个由5个单单词组成的句子 构造出来的形状是 [5,1,3]也就是明确告诉网络结构我一个句子由5个单词组成，
	# 每个单词由一个1X3的向量组成，就是这个样子[1,2,3]
	# 同时确定了网络结构，每个批次只输入一个句子，其中第二维的batch_size很容易迷惑人
	# 对整个这层来说，是一个批次输入多少个句子，具体但每个神经元，就是一次性喂给神经元多少个单词。
	print('Inputs:', inputs)

	# 初始化隐藏状态
	#hidden = (torch.randn(1, 1, 3),
	#		  torch.randn(1, 1, 3))
	#print('Hidden:', hidden)


	#inputs = torch.cat(inputs).view(len(inputs), 1, -1)
	#hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
	#out = value_net(inputs)
	out = value_net(inputs.cuda())
	print('out2', out)
	#print('hidden3', hidden)
