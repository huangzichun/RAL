# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def train(model, data, agent, al, epoch, env, budget, MEMORY_CAPACITY, agent2, limited_cost=100):
    # says only ${limited_cost} queries by human
    # agent2 is used to determinate that this instance should be labeled by human or model
    #        so it should take (current action, state) as the input
    epoch_rest = []
    epoch_test = []
    for i_episode in range(epoch):
        print('epoch:', i_episode + 1)
        labeled = []
        # in case the information leak when evaluation on the rest of training data
        data.reset()

        s = env.initial(al) # 得到初始状态
        # for j in range(budget):
        while len(labeled) <= limited_cost:
            # 根据dqn来接受现在的状态，得到一个行为
            s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
            # instance to be labeled
            a = agent.choose_action(s)

            # simply version, add action information by position encoding
            position_a = torch.zeros(s.shape).float()
            position_a[a] = 1
            # position_a.require_grad = False
            a2 = agent2.choose_action(s + position_a)

            if a2 == 1:
                # print("queried by human")
                # queries by human, otherwise a2 = 0
                labeled.append(a)

            s_next, r = env.feedback(a, a2) # 根据环境的行为，给出一个反馈

            s2_next = torch.from_numpy(s_next).type(torch.FloatTensor)
            a2_next = agent.choose_action(s2_next) # for agent2 only
            position_a_next = torch.zeros(s.shape).float()
            position_a_next[a2_next] = 1
            
            agent.store_transition(s,a,r,s_next) # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
            agent2.store_transition(s + position_a, a2, r, (s_next + position_a_next).numpy())
            if agent.memory_counter > MEMORY_CAPACITY:
                agent.Learn()

            if agent2.memory_counter > MEMORY_CAPACITY:
                agent2.Learn()

            s = s_next
        if agent.memory_counter > 0:
            print("there are some left")
            agent.Learn()

        # model evaluation on the rest of (training) data
        acc_rest, num = env.get_rest_unlabeled_data_effect()
        print("the rest data acc = {} on {} samples".format(acc_rest, num))
        epoch_rest.append(acc_rest)

        # model evaluation on the test data
        acc_test, num = env.get_test_unlabeled_data_effect()
        print("the test data acc = {} on {} samples".format(acc_test, num))
        epoch_test.append(acc_test)
        
        torch.save(agent.target_net, '\model.pkl')
        torch.save(agent2.target_net, '\model2.pkl')
