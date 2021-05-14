# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random

def test(model, data, agent, al, epoch, env, budget):
    s = env.initial(al) # 得到初始状态
    data.reset()
    target_net = torch.load('\model.pkl')

    for j in range(budget):
        # 根据dqn来接受现在的状态，得到一个行为
        # 行为：挑选数据 
        s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
        s = s.squeeze(-1)

        actions_value = target_net.forward(s).detach().numpy() 
        actions_value = data.value_filter(actions_value) 
        action = np.argmax(actions_value)

        # baseline2采用随机
        # action = random.sample(data.unlabeled_data_set, 1)
        # action = action[0]
  
        s_next, r = env.feedback(action) 
        s = s_next 
        
        