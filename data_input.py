# 数据输入及预处理

import json
import numpy as np
import random

class data_input(object):
    def __init__(self, FILENAME, TESTFILENAME):
        with open(FILENAME) as f:
            self.word_pair_list = json.load(f)
        with open(TESTFILENAME) as f:
            self.test_data = json.load(f)

        self.word_total = self.word_pair_list + self.test_data
        self.word_dict, self.id2word = self.read_dict()

        self.n_dict = len(self.word_dict)
        self.n_word = len(self.word_pair_list)

        random.shuffle(self.word_pair_list)
        random.shuffle(self.test_data)

        self.data_list, self.target_list = self.convert(self.word_pair_list, self.n_word)
        self.test_data_list, self.test_target_list = self.convert(self.test_data, len(self.test_data))
    
    def read_dict(self, ):  # 给每个词标号(one-hot)，并建立序号和embed间的对应关系
        length = len(self.word_total)
        number = 0
        word_dict = id2word = {}
        for i in range(length):
            for j in range(2):
                if word_dict.get(self.word_total[i][j]) == None:
                    word_dict[self.word_total[i][j]] = number
                    id2word[number] = self.word_total[i][j]
                    number += 1

        return word_dict, id2word
    
    def convert(self, word_list, num):  # 将词转成序号(one-hot)
        data_list = np.empty((num, 2))
        target_list = np.empty(num)
        for i in range(num):
            target_list[i] = int(word_list[i][2])
            for j in range(2):
                data_list[i][j] = self.word_dict[word_list[i][j]]

        return data_list, target_list