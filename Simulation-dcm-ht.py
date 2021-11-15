import networkx as nx
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import random
import sys
import copy
import gc
import collections
import warnings

warnings.filterwarnings("ignore")

# 全局定义tasks的数量
TASK_NUM = 1000
reserved_price = []
PROVIDERS_NUM = 100

'''
随机生成一个graph
随机生成一堆valuation
'''


def _RandomizeTree(n, seed):
    return nx.full_rary_tree(8, n)
    # return nx.random_tree(n, seed=seed)


def _ShowGraph(g):
    nx.draw(g, with_labels=True)
    plt.show()


class RequesterHT:
    def __init__(self, price):
        '''
        :param price: reserved price vector for all tasks
        '''
        self.idx = 0
        self.tasks_num = [i for i in range(1, TASK_NUM + 1)]
        self.reserved_p = price


class Provider:
    def __init__(self, idx, cost, tasks):
        self.idx = idx
        self.cost = cost
        self.tasks = tasks
        self.distance = float('inf')
