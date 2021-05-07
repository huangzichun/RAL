# 数据处理

import json
import numpy as np

class data_processing(object):
    def __init__(self, FILENAME, TESTFILENAME, EMBEDDING_DIM):
        with open(FILENAME) as f:
            self.word_pair_list = json.load(f)

        self.word_dict = self.read_dict()

        with open(TESTFILENAME) as f:
            self.test_data = json.load(f)
        
        self.n_dict = len(self.word_dict)
        self.n_word = len(self.word_pair_list)
        self.data_list, self.target_list = self.convert(self.word_pair_list, self.n_word)
        self.test_data_list, self.test_target_list = self.convert(self.test_data, len(self.test_data))
    
    def read_dict(self, ):
        length = len(self.word_pair_list)
        number = 0
        word_dict = {}
        for i in range(length):
            for j in range(2):
                if word_dict.get(self.word_pair_list[i][j]) == None:
                    word_dict[self.word_pair_list[i][j]] = number
                    number += 1
        return word_dict
    
    def convert(self, word_list, num):
        data_list = np.empty((num, 2))
        target_list = np.empty(num)

        for i in range(num):
            target_list[i] = int(word_list[i][2])
            for j in range(2):
                data_list[i][j] = self.word_dict[word_list[i][j]]

        return data_list, target_list