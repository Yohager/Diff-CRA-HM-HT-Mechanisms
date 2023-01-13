import networkx as nx
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import random
import sys
import copy
import tqdm 
import gc
import collections
import warnings
import scipy.stats as stats
from attributes import ReqHT, SupHT

warnings.filterwarnings("ignore")

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

    # check if all the nodes are included
    if set(cr_tree.nodes) != set(network.nodes):
        sys.exit("Network to CRTree Error.")

    print("Graph to critical referrer tree process consumes: ", time.time() - start)
    return cr_tree.to_undirected()

def marginal_value(winners,tasks_price,profile,target):
    '''
    para: winners: current winner -> array -> each element is an instance of the provider class
    para: tasks_prices -> arry 
    para: profile: key-value -> report types 
    para: target: the maximum gain can obtained by one specific target
    '''
    finished_tasks = set()
    # print(profile.keys())
    for winner in winners:
        for t in profile[winner].tasks:
            finished_tasks.add(t)
        #print(profile[winner].tasks)
    # set(tasks) - set(finished_tasks) 
    n = len(tasks_price.keys()) #tasks_price represents the price for all tasks
    all_tasks = set(list(range(1,n+1)))
    unfinished_tasks = all_tasks - finished_tasks
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

def _RandomGraph(n,prob=0.05):
    flag = False 
    while not flag:
        g = nx.Graph()
        g.add_nodes_from(list(range(n+1)))
        for i in range(n+1):
            for j in range(i+1,n+1):
                p = random.random()
                if p < prob:
                    g.add_edge(i,j)
        if nx.is_connected(g):
            flag = True 
    return g 



def Initialization(ts,minv,maxv,sup_nums,smin,smax,sminc,smaxc): 
    vals =  [random.randint(minv,maxv) for _ in range(ts)]# random generate the uniform distribution for tasks cost 
    tasks = list(range(1,ts+1))
    requester = ReqHT()
    requester.tasks = {tidx:v for tidx,v in zip(tasks,vals)}
    # randomly generating the producing cost and producing ability
    suppliers = {}
    for i in range(1,sup_nums+1):
        cur = SupHT(i)
        cur.idx = i 
        # cur_t represents the random number each supplier can finish
        cur.ability = random.sample(tasks,random.randint(smin,smax))
        cur.cost = random.randint(sminc, smaxc)
        suppliers[i] = cur
    return vals,requester,suppliers 

def Initialization_normal_distribution(host_id, task_num, avgv, sigma,  minv, maxv, start_idx, providers_num, min_a, max_a, min_c, max_c):
    X = stats.truncnorm(
        (minv - avgv) / sigma, (maxv - avgv) / sigma, loc=avgv, scale=sigma
    )
    vals = X.rvs(task_num)
    idxs = list(range(1, task_num+1))
    requester = ReqHT()
    requester.tasks = {idx:val for idx, val in zip(idxs, vals)}
    requester.idx = host_id
    providers = {}
    for i in range(start_idx, providers_num+start_idx):
        if i == host_id:
            continue 
        cur = SupHT(i)
        cur.ability = random.sample(idxs, random.randint(min_a, max_a))
        # curX = stats.truncnorm((minv - avgv) / sigma, (maxv - avgv) / sigma, loc=avgv, scale=sigma)
        cur.cost = random.randint(min_c, max_c)
        providers[i] = cur 
    return vals, requester, providers

class DiffCRAHT:
    def __init__(self,tasks,graph,suppliers_num,requester,suppliers,vals) -> None:
        '''
        tasks: total task num
        graph market: nx graph class instance 
        requester: reqht class instance
        suppliers: supht class instance
        suppliers_num: total supplier num
        vals: reserved cost for tasks 
        '''
        self.tasks = list(range(1,tasks+1)) # heterogeneous tasks, number from 1
        self.tnum = tasks 
        self.vals = vals
        self.requester = requester
        self.suppliers = suppliers
        self.sup_nums = suppliers_num
        self.winners = set()
        self.payments = dict()
        self.graph = graph
        self.submarkets = collections.defaultdict(set)
        self.max_dist = -float('inf')
        self.left_tasks = set()
        self.total_sc = 0
        self.budgets = sum(self.vals)

    def NetworkedMarketDivision(self):
        for key in self.suppliers.keys():
            dist = nx.shortest_path_length(self.graph,self.requester.idx,key)
            # print(dist)
            self.submarkets[dist].add(key)
            self.max_dist = max(self.max_dist,dist)
    
    def ShowSubmarkets(self):
        return self.submarkets
    
    def cals_mv(self,ts,ms):
        res = []
        for m in ms:
            ct = ms[m].ability
            et = set(ts) & set(ct)
            # print('et',ts,ct)
            v = sum([self.requester.tasks[d] for d in et])
            # print('v-cost',v,ms[m].cost)
            mvt = v - ms[m].cost 
            res.append([ms[m].idx,mvt])
        return res 
    
    def OneLevelAllocation(self,ts,market):
        # iteration stops when there is no agent has non-negative marginal value for the requester 
        selected = []
        # payment = []
        cur_m = copy.deepcopy(market)
        # print(market)
        # print('cur_m',cur_m)
        lt = ts[:]
        while lt and cur_m:
            # update mv_list
            # print('cur_market:',cur_m) 
            mv_list = sorted(self.cals_mv(lt,cur_m),key=lambda x:[-x[1],x[0]])
            # print('show marginal values:',mv_list)
            if not mv_list:
                break 
            if mv_list[0][1] <= 0: # if the ranking top supplier's mv < 0 -> break
                break 
            cur = mv_list[0][0] # choose the top supplier with the highest mv 
            # print('here',cur)
            selected.append(cur) 
            cur_t = cur_m[cur].ability
            cur_m.pop(cur)
            for t in cur_t:
                if t in lt:
                    lt.remove(t)
        return selected
    
    def OneLevelPayment(self,ws,lt,market):
        payment = {}
        for w in ws:
            mw = copy.deepcopy(market)
            mw.pop(w)
            pw = -float('inf')
            ct = lt[:]
            wprime = []
            while ct and mw:
                # w's value under ct 
                vw = sum([self.requester.tasks[d] for d in (set(ct)&set(market[w].ability))])
                # print('here vw is:',vw)
                # mv_list = self.cals_mv(ct,mw)
                mv_list = sorted(self.cals_mv(ct,mw),key=lambda x:[-x[1],x[0]])
                # print('here mvs:',mv_list)
                cur = mv_list[0]
                if cur[1] <= 0:
                    break 
                # print('payment here:',vw,cur[1])
                pw = max(pw,vw - cur[1]) # update payment value 
                mw.pop(cur[0])
                wprime.append(cur[0])
                for t in market[cur[0]].ability:
                    if t in ct:
                        ct.remove(t)
            if not wprime:
                pw = sum([self.requester.tasks[d] for d in (set(ct)&set(market[w].ability))])
            if not mw:
                pw = max(pw,sum([self.requester.tasks[d] for d in (set(ct)&set(market[w].ability))]))
            payment[w] = pw 
        # print('here payment is:',payment)
        return payment 

    def AllocationAndPayment(self, type='global'):
        # self.Initialization()
        self.NetworkedMarketDivision()
        tasks = self.tasks
        if type == 'global':
            for d in tqdm.tqdm(range(1,self.max_dist+1)):
                print('Market Size: ', len(self.submarkets[d]))
                # level by level market s
                # cur_m = [self.suppliers[x] for x in self.submarkets[d]]
                cur_m = {x:self.suppliers[x] for x in self.submarkets[d]}
                if not tasks: # no task left 
                    break 
                # find winners in the current level 
                cur_winners = self.OneLevelAllocation(tasks,cur_m)
                # print('cur winners:',cur_winners)
                for cw in cur_winners:
                    self.winners.add(cw)
                cur_payments = self.OneLevelPayment(cur_winners,tasks,cur_m)
                for k,v in cur_payments.items():
                    self.payments[k] = v 
                # update left task set 
                finished_t = set()
                for x in cur_winners:
                    for t in self.suppliers[x].ability:
                        finished_t.add(t)
                for ft in finished_t:
                    if ft in tasks:
                        tasks.remove(ft)
        elif type == 'local':
            for d in range(1,2):
                # consider only the local market 
                cur_m = {x:self.suppliers[x] for x in self.submarkets[d]}
                if not tasks:
                    break 
                cur_winners = self.OneLevelAllocation(tasks,cur_m)
                for cw in cur_winners:
                    self.winners.add(cw)
                cur_payments = self.OneLevelPayment(cur_winners,tasks,cur_m)
                for k,v in cur_payments.items():
                    self.payments[k] = v 
                finished_t = set()
                for x in cur_winners:
                    for t in self.suppliers[x].ability:
                        finished_t.add(t)
                for ft in finished_t:
                    if ft in tasks:
                        tasks.remove(ft)               
        if tasks:
            self.left_tasks = set(tasks)
            self.winners.add('virtual')
            self.payments['virtual'] = 0
            for t in tasks:
                self.payments['virtual'] += self.requester.tasks[t]
    
    def Implementation(self,type):
        self.AllocationAndPayment(type)
    
    def AllocationResult(self):
        return self.winners
    
    def PaymentResult(self):
        return self.payments 
    
    def OutsourceItems(self):
        return self.tnum - len(self.left_tasks)
    
    def TotalSocialCost(self):
        self.total_sc = 0
        for w in self.winners:
            if w == 'virtual':
                self.total_sc += self.payments['virtual']
            else:
                self.total_sc += self.suppliers[w].cost 
        return self.total_sc 
    
    def TotalPayment(self):
        return sum(self.payments.values())
    
    def TotalBudget(self):
        return self.budgets
    
    def BudgetFeasible(self):
        total_payment = self.TotalPayment()
        return True if self.budgets >= total_payment else False 