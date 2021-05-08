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
import embedding
import warnings

warnings.filterwarnings("ignore")

random.seed(114514)
# 超参
epoch = 1
EMBEDDING_DIM = 50
MEMORY_CAPACITY = 20  # 记忆存储容量
FILENAME = 'data.json'
TESTFILENAME = 'test.json'
budget = 500

data_input = data_processing.data_processing(FILENAME, TESTFILENAME)
Embedding = embedding.embed(data_input)
DATA = data.Data(data_input, Embedding) 

MODEL = model.model(DATA, data_input, Embedding, EMBEDDING_DIM)
AL = al.al(DATA, data_input, Embedding, MODEL)
Env = env.env(DATA, data_input, AL, Embedding, MODEL)

N_STATES = len(Env.state) # state向量维数
N_ACTIONS = Env.action_space_dim # action种类数
Agent = agent.DQN(DATA, N_STATES, N_ACTIONS)

train.train(MODEL, Agent, AL, epoch, Env, budget, MEMORY_CAPACITY)