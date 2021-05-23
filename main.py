import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import env #
import agent 
import data_management
import data_input
import model
import train
import al
import random
import embedding
import test
import warnings
from raw.agent2 import agent2

warnings.filterwarnings("ignore")

random.seed(114)

epoch = 20
EMBEDDING_DIM = 20
MEMORY_CAPACITY = 20  # 记忆存储容量
FILENAME = 'B_train.json'
TESTFILENAME = 'B_test.json'
budget = 1024

data_input = data_input.data_input(FILENAME, TESTFILENAME)
Embedding = embedding.embed(data_input)
DATA = data_management.Data(data_input, Embedding) 

MODEL = model.model(DATA, data_input, Embedding, EMBEDDING_DIM)
AL = al.al(DATA, data_input, Embedding, MODEL)
Env = env.env(DATA, data_input, AL, Embedding, MODEL)

N_STATES = len(Env.state) 
N_ACTIONS = Env.action_space_dim 
Agent = agent.DQN(DATA, N_STATES, N_ACTIONS)
# action space={human, model}
Agent2 = agent2(DATA, N_STATES, N_ACTIONS=2)

train.train(MODEL, DATA, Agent, AL, epoch, Env, budget, MEMORY_CAPACITY, Agent2)
# test.test(MODEL, DATA, Agent, AL, epoch, Env, budget)
