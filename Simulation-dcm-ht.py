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
# TASK_NUM = 6
# reserved_price = [3,8,6,8,10,9]
# PROVIDERS_NUM = 4
# providers_cost = [8,6,6,5]
# providers_tasks = {
#     1:[1,3,4,5],
#     2:[2,3,5],
#     3:[3,4,6],
#     4:[3]
# }
TASK_NUM = 10
reserved_price = [3,5,4,8,10,7,12,9,6,5]
PROVIDERS_NUM = 8
providers_cost = [15,12,9,7,5,8,6,11]
providers_tasks = {
    1:[1,2,3,5,7],
    2:[2,4,5,8,9],
    3:[3,5,6,10],
    4:[4,8,9],
    5:[1,10],
    6:[2,4,5,8],
    7:[3,5,6,8],
    8:[1,4,5,8,9]
}

'''
随机生成一个graph
随机生成一堆valuation
'''


def _RandomizeTree(n, seed):
    return nx.full_rary_tree(8, n)
    # return nx.random_tree(n, seed=seed)

'''
Here we can create specific tree for our models and mechanisms
'''
def specific_tree():
    g = nx.Graph()
    g.add_nodes_from([0,1,2,3,4,5,6,7,8])
    g.add_edges_from([[0,1],[0,2],[0,3],[1,4],[1,5],[2,6],[3,7],[7,8]])
    return g 


def _ShowGraph(g):
    nx.draw(g, with_labels=True)
    plt.show()


def CreateSpecificRP():
    #生成一个requester
    requester = RequesterHT(reserved_price)
    #生成一堆providers
    idxs = [i for i in range(1,PROVIDERS_NUM+1)]
    providers = {}
    for i,idx in enumerate(idxs):
        tmp = Provider(idx,providers_cost[i],providers_tasks[idx])
        providers[idx] = tmp 
    return requester, providers


def _graph_to_cr_tree(network):
    """
    To creat the cr_tree of a referral graph,
    where agents are represented by integers and '0' represents the principal (change in PP file).
    """
    principal = 0
    start = time.time()
    node_set = set(network.nodes)
    agent_set = set(network.nodes) - {principal}

    critical_dict = {contestant: set() for contestant in agent_set}
    dominating_dict = {contestant: set() for contestant in agent_set}

    leaf_set = set()

    for node in agent_set:
        sub_graphs = network.subgraph([i for i in node_set if i != node])
        connected_components = list(nx.connected_components(sub_graphs))
        if len(connected_components) != 1:
            for component in connected_components:
                if not (principal in component):
                    dominating_dict[node] = dominating_dict[node] | component

    for node in agent_set:
        if len(dominating_dict[node]) == 0:
            leaf_set.add(node)
        for vertex in dominating_dict[node]:
            critical_dict[vertex].add(node)

    cr_tree = nx.DiGraph()

    for node in leaf_set:
        if len(critical_dict[node]) == 0:
            cr_tree.add_edge(principal, node)
        else:
            critical_sequence = [(vertex, len(critical_dict[vertex])) for vertex in critical_dict[node]]
            critical_sequence.sort(key=lambda x: x[1])
            cr_tree.add_edge(principal, critical_sequence[0][0])
            cr_tree.add_edge(critical_sequence[-1][0], node)
            for i in range(len(critical_sequence) - 1):
                cr_tree.add_edge(critical_sequence[i][0], critical_sequence[i + 1][0])

    # 判断树中是否包含所有的点
    if set(cr_tree.nodes) != set(network.nodes):
        sys.exit("Network to CRTree Error.")

    print("Graph to critical referrer tree process consumes: ", time.time() - start)
    return cr_tree.to_undirected()

def marginal_value(winners,tasks_price,profile,target):
    '''
    para: winners 当前的winners，传入的参数应该是一个数组，其中每个元素是一个provider的类实例
    para: tasks_prices
    para: target 目标的agent所能给当前的requster带来的最高的收益值 target应该传进来是一个对象的实例
    '''
    finished_tasks = set()
    # print(profile.keys())
    for winner in winners:
        for t in profile[winner].tasks:
            finished_tasks.add(t)
        #print(profile[winner].tasks)
    # tasks与finished_tasks求一个差集
    n = len(tasks_price.keys()) #tasks_price表示为1-n号任务的价值
    all_tasks = set(list(range(1,n+1)))
    unfinished_tasks = all_tasks - finished_tasks #求一个差集
    # print('all tasks and finished tasks:',all_tasks,finished_tasks)
    mv = 0 
    for task in unfinished_tasks:
        if task in target.tasks:
            mv += tasks_price[task]
    return mv - target.cost

def maximize_marginal_value(winners,tasks_prices,profile):
    mv_0 = []
    winner_idxs = winners.keys()
    for p in profile:
        if p in winner_idxs:
            continue
        mv_0.append([p,marginal_value(winner_idxs,tasks_prices,profile,profile[p])])
    mv_0.sort(key=lambda x:x[1], reverse=True)
    # print('here the marginal valuation is:',mv_0)
    return mv_0[0] if len(mv_0) > 0 else [-1,-1] 


class RequesterHT:
    def __init__(self, price):
        '''
        :param price: reserved price dict for tasks_num
        '''
        self.idx = 0
        self.tasks_num = [i for i in range(1, TASK_NUM + 1)]
        self.reserved_p = {t:v for t,v in zip(self.tasks_num,price)}


class Provider:
    def __init__(self, idx, cost, tasks):
        self.idx = idx
        self.cost = cost
        self.tasks = tasks 
        self.distance = float('inf')


class NDCrowdsensingMechanism:
    def __init__(self,requester, providers, graph):
        '''
        :param requester: RequesterHT类实例
        :param providers: 字典结构：key表示idx，val表示provider类实例
        :param graph: diffusion graph
        '''
        self.requester = requester
        self.providers = providers
        self.requester_neighbors = {}
        self.tasks = set(requester.tasks_num)
        self.vals = requester.reserved_p
        self.graph = graph
        self.cr_tree = None
        self.winners = {}
        self.payment = {}
        self.left_tasks = set()

    def _ConstructDiffusionTree(self):
        # self._graph_to_cr_tree()
        # self._AddVirtualNode()
        # self.cr_tree = self.graph
        self.cr_tree = _graph_to_cr_tree(self.graph)
        for nei in self.cr_tree.neighbors(0):
            self.requester_neighbors[nei] = self.providers[nei]
        
    # def _calculate_winners(self,pf):
    #     ws = {}


    def AllocationAndPayment(self):
        # profile = self.providers
        # mv_0 = []
        # for p in profile:
        #     mv_0.append([p,marginal_value(self.winners,self.requester.reserved_p,profile[p])])
        # mv_0.sort(key=lambda x:x[1],reverse=True)
        # w = mv_0[0] 
        profile = self.providers
        current_winner = maximize_marginal_value(self.winners,self.requester.reserved_p,profile)
        # print(current_winner)
        '''
        分配规则：优先给出所有winners的id
        '''
        while self.winners.keys() != self.providers.keys() and current_winner[1] >= 0:
            self.winners[current_winner[0]] = 1 #表示当前的这个人被选择了
            # profile.pop(current_winner[0]) #将当前的这个winner从profile中移除
            # print('here the profile is:',profile)
            current_winner = maximize_marginal_value(self.winners,self.requester.reserved_p,profile)
        print(self.winners)
        '''
        首先计算self.winners
        由于可能存在有些任务无法完成，因此计算是否存在一些剩余的tasks需要requester自己去完成
        同时初始化Winners的payment，方便之后更新payments        
        '''
        self.cal_left_tasks()

        for winner in self.winners:
            self.payment[winner] = 0

        for winner in self.winners:
            # print('current winner is:',winner)
            profile_without_i = {k:v for k,v in self.providers.items() if k != winner} # 将除了i的其他人写成一个set
            tau = dict()
            tmp_idx = maximize_marginal_value(tau,self.requester.reserved_p,profile_without_i)
            winner_tau_val = marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[winner]) + self.providers[winner].cost
            if marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[tmp_idx[0]]) >= 0:
                # print(winner_tau_val,marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[tmp_idx[0]]))
                winner_tau_val -= marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[tmp_idx[0]])
            self.payment[winner] = max(self.payment[winner],winner_tau_val)
            '''
            最开始初始化的时候先求一次这个winner的payment
            '''
            while len(tau.keys()) < len(profile_without_i.keys()) and tmp_idx[1] >= 0: #当前还不是所有人都被选择成为winner以及当前选中的人的边际价值还是大于0的则循环继续
                # winner-tau_val求的是v_i(\tau)
                tau[tmp_idx[0]] = 1
                tmp_idx = maximize_marginal_value(tau,self.requester.reserved_p,profile_without_i)
                winner_tau_val = marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[winner]) + self.providers[winner].cost
                if marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[tmp_idx[0]]) >= 0:
                    # print(winner_tau_val,marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[tmp_idx[0]]))
                    winner_tau_val -= marginal_value(tau,self.requester.reserved_p,profile_without_i,self.providers[tmp_idx[0]])
                    
                self.payment[winner] = max(self.payment[winner],winner_tau_val)
                # print(tau,self.payment[winner])
            if tau.keys() == profile_without_i.keys():
                self.payment[winner] = max(self.payment[winner],marginal_value(tau,self.requester.reserved_p,self.providers[winner]) + self.providers[winner])
        # print(self.payment)


    def cal_left_tasks(self):
        finished_tasks = set()
        for winner in self.winners:
            for t in self.providers[winner].tasks:
                finished_tasks.add(t)
        self.left_tasks = self.tasks - finished_tasks

    def AllocationResult(self):
        return self.winners 
    
    def PaymentResult(self):
        return self.payment 
    
    def TotalSocialCost(self):
        sc = 0
        for winner in self.winners:
            sc += self.providers[winner].cost 
        
        for lt in self.left_tasks:
            sc += self.requester.reserved_p[lt]
        return sc 

    def TotalPayment(self):
        t_pay = sum(self.payment.values())
        for lt in self.left_tasks:
            t_pay += self.requester.reserved_p[lt]
        return t_pay


class DiffusionCrowdsensingMechanismHT:
    def __init__(self,requester,providers,graph):
        self.requester = requester
        self.tasks = set(requester.tasks_num)
        self.vals = self.requester.reserved_p
        self.providers = providers
        self.winners = {}
        self.payments = {}
        self.cr_tree = None
        self.graph = graph

    def _ConstructDiffusionTree(self):
        self.cr_tree = _graph_to_cr_tree(self.graph)
    
    def _CalculateDistance(self):
        for provider in self.providers:
            self.providers[provider].distance = nx.shortest_path_length(self.graph, 0, self.providers[provider].idx)
    
    def AllocationAndPayment(self):
        self._CalculateDistance()
        self._ConstructDiffusionTree()  
        distances = [[self.providers[p].idx,self.providers[p].distance] for p in self.providers]
        distances.sort(key=lambda x:x[1])
        print(distances)
        cur_winners = []
        #profile = self.providers
        for p in distances:
            cur_p = self.providers[p[0]]
            # 首先判断其边际价值是否大于0 
            cur_mv = marginal_value(cur_winners,self.requester.reserved_price,self.providers,cur_p)
            if cur_mv < 0:
                continue 
            # 先明确他的竞争对手 N \ C_i \W 
            cur_children = []
            cur_p_competitors = [self.providers[k] for k in self.providers if k not in cur_winners and k not in ]
            





if __name__ == "__main__":
    requester,providers = CreateSpecificRP()
    basic_g = specific_tree()
    # tmp_ndm = NDCrowdsensingMechanism(requester,providers,basic_g)
    # # print(tmp_ndm.requester)
    # # print(tmp_ndm.providers)
    # tmp_ndm.AllocationAndPayment()
    tmp_dcm_ht = DiffusionCrowdsensingMechanismHT(requester,providers,basic_g)
    tmp_dcm_ht.AllocationAndPayment()
