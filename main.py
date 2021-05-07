import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import env # 环境
import agent 
import data
import data_processing
import model
import train
import al
import random

random.seed(114514)
# 超参
epoch = 1000
MEMORY_CAPACITY = 20  # 记忆存储容量
FILENAME = 'coordpairs_wiki100.json'
TESTFILENAME = 'coordpairs_wiki100.json'
EMBEDDING_DIM = 128
budget = 200

data_input = data_processing.data_processing(FILENAME, TESTFILENAME, EMBEDDING_DIM)
DATA = data.Data(data_input) 
MODEL = model.model(DATA, data_input, EMBEDDING_DIM)
AL = al.al(DATA, data_input, MODEL)
Env = env.env(DATA, data_input, AL, MODEL)

N_STATES = len(Env.state) # state向量维数
N_ACTIONS = Env.action_space_dim # action种类数
Agent = agent.DQN(DATA, N_STATES, N_ACTIONS)

train.train(MODEL, Agent, AL, epoch, Env, budget, MEMORY_CAPACITY)