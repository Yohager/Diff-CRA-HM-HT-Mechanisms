'''
define some classes 
class RequesterHM: requester in homogeneous scenario
class RequesterHT: requester in heterogeneous scenario
class Provider: providers type 
'''


class RequesterHM:
    def __init__(self, price, task_num):
        self.idx = 1
        self.tasks_num = task_num
        self.reserved_p = price
        # self.children = []


class RequesterHT:
    def __init__(self, tasks, vals):
        # reserved price
        self.idx = 0
        self.vals = dict()
        # vals vector for different tasks
        for x, y in zip(tasks, vals):
            self.vals[x] = y
        # self.children = []


class Provider:
    def __init__(self, idx, cost, tasks):
        self.idx = idx
        # self.children = []
        # self.parents = []
        self.cost = cost
        self.tasks = tasks
        self.distance = float('inf')

class ReqHT:
    def __init__(self) -> None:
        self.idx = 0
        self.tasks = None 

class SupHT:
    def __init__(self,idx) -> None:
        self.idx = idx
        self.ability = None
        self.cost = None  
        self.distance = float('inf')