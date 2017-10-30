# coding: utf-8
import numpy as np
import random

x_array = [[1, 1],[1,0],[0,1],[0,0]]

def AND(x, y):
    random_result = random.choice([0,1])
    if random_result == 0:
        return [0,1]
    else:
        return [1,0]

y = [AND(i[0], i[1]) for i in x_array]
t = [[0,1],[0,1],[0,1],[1,0]]

def cross_entropy_error(y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

cross_entropy_error(np.array(y), np.array(t)) 