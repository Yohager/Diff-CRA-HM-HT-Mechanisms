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
import collections
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from attributes import RequesterHM, Provider

def _FixCompetitors(graph, target):
    g = copy.deepcopy(graph)
    # return the set of all competitors' idxes 
    res = set()
    all_nodes = set(g.nodes)
    g.remove_node(target)
    if not nx.is_connected(g):
        # if unconnected, then there exists dominant relations
        left_nodes = nx.node_connected_component(g, 0)
        # for node in all_nodes:
        #     if node not in left_nodes:
        #         res.add(node)
        for node in left_nodes:
            res.add(node)
        res = res - {0}
    else:
        res = all_nodes - {0, target}
    del g
    return res

def New_RandomizeValuation_HM(rp, pn, host_id, start_id):
    '''
    :param rp: reserved price for homo task 
    :param pn: suppliers number
    :param host_id: crowdsourcing organizer 
    :param start_id: 0 or 1 depends on the specific networks 
    :return: req profile, suppliers' profile 
    '''
    req = RequesterHM(rp, host_id)
    ps = {}
    for i in range(start_id, pn + start_id):
        if i == host_id:
            continue  
        ri = max(1, round(random.gauss(5,2)))
        pi = max(1, random.gauss(5,2))
        p = Provider(i, pi, ri)
        ps[i] = p
    return req, ps


def _SocialCostMinimum(p, k, target):
    '''
    :param p: providers' type profile 
    :param k: left tasks number 
    :param target: check if can be the winner
    :return: bool var, true for winner while false for loser 
    '''
    winners = []
    n = len(p)
    # p[i][0]: idx, p[i][1]: unit-cost, p[i][2]: amount
    p.sort(key=lambda x: x[1])
    # print('rank p:', p)
    idx = 0
    while idx < n and k >= 0:
        # if it is not the final person or there exists some unallocated tasks 
        k -= p[idx][2]
        winners.append(p[idx][0])
        idx += 1
    # if target in winners:
    #     print('yes')
    # print('here are winners: ', winners)
    return True if target in winners else False


def _CalculateSocialCost(p, total_num):
    p.sort(key=lambda x: x[1])
    i = 0
    num = total_num
    sc = {}
    while i < len(p) and num > 0:
        if num > p[i][2]:
            sc[p[i][0]] = [p[i][1], p[i][2]]
            num -= p[i][2]
        else:
            sc[p[i][0]] = [p[i][1], num]
            break
        i += 1
    # print('allocation opt:', sc)
    return sc


def _SocialCostIncrease(profile, target, total_num):
    '''
    :param profile: for all p in profile: p[0]: id; p[1]: unit cost; p[2]: amount
    :param target: one winner idx
    :return: social cost
    '''
    profile.sort(key=lambda x: x[1])
    profile_without_target = [p for p in profile if p[0] != target]
    profile_without_target.sort(key=lambda x: x[1])
    allocation_opt = _CalculateSocialCost(profile, total_num)
    allocation_without_target = _CalculateSocialCost(profile_without_target, total_num)
    if target in allocation_opt:
        sc_opt = sum(x * y for x, y in allocation_opt.values()) - allocation_opt[target][0] * allocation_opt[target][1]
    else:
        sc_opt = sum(x * y for x, y in allocation_opt.values())
    sc_without_target = sum(x * y for x, y in allocation_without_target.values())
    # print('social cost increase:', target, sc_without_target - sc_opt)
    return sc_without_target - sc_opt


# VCG extension in local market 
class NDVCG:
    def __init__(self, requester, providers, tasks):
        self.tasks_num = tasks
        self.requester = requester
        self.providers = providers
        self.winners = {}
        self.payments = {}
        self.total_social_cost = 0
        self.budgets = self.requester.reserved_p * self.tasks_num

    def AllocationAndPayment(self):
        profile = [['virtual', self.requester.reserved_p, self.tasks_num]]
        for p in self.providers:
            profile.append([self.providers[p].idx,
                            self.providers[p].cost,
                            self.providers[p].tasks])
        profile.sort(key=lambda y: y[1])
        # print('Here profile is:',profile)
        # print(profile)
        # rank unit cost in ascending order 
        num = self.tasks_num
        vcg_allocation = _CalculateSocialCost(profile, num)
        for w in vcg_allocation:
            self.winners[w] = vcg_allocation[w][1]
        # for x in self.winners:
        #     self.payments[x] = _SocialCostIncrease(profile, x, num)
        for provider in self.providers:
            cur_payment = _SocialCostIncrease(profile, self.providers[provider].idx, num)
            # print('provider, cur_payment', provider, cur_payment)
            if cur_payment != 0:
                self.payments[self.providers[provider].idx] = cur_payment

    def AllocationResult(self):
        return self.winners

    def PaymentResult(self):
        return self.payments

    def TotalSocialCost(self):
        self.total_social_cost = 0
        for winner in self.winners:
            if winner == 'virtual':
                self.total_social_cost += self.requester.reserved_p * self.winners[winner]
            else:
                self.total_social_cost += self.winners[winner] * self.providers[winner].cost
        return self.total_social_cost

    def TotalPayment(self):
        total_p = sum(self.payments.values())
        if 'virtual' in self.winners:
            total_p += self.winners['virtual'] * self.requester.reserved_p
        return total_p

    def BudgetFeasible(self):
        total_cost = sum(self.payments.values())
        return True if self.budgets >= total_cost else False


# VCG extension in global market
class DVCG:
    def __init__(self, requester, providers, tasks, graph):
        self.requester = requester
        self.providers = providers
        self.winners = {}
        self.payments = {}
        self.tasks_num = tasks
        self.dominate_set = {}
        self.graph = graph
        self.cr_tree = nx.Graph()
        self.total_social_cost = 0
        self.budgets = self.requester.reserved_p * self.tasks_num

    def _ConstructDS(self):
        principal = 0
        node_set = set(self.graph.nodes)
        agent_set = node_set - {principal}
        # critical_dict = {contestant: set() for contestant in agent_set}
        dominating_dict = {contestant: set() for contestant in agent_set}

        leaf_set = set()

        for node in agent_set:
            sub_graphs = self.graph.subgraph([i for i in node_set if i != node])
            connected_parts = list(nx.connected_components(sub_graphs))
            if len(connected_parts) != 1:
                for part in connected_parts:
                    if principal not in part:
                        dominating_dict[node] = dominating_dict[node] | part
        self.dominate_set = dominating_dict

    def AllocationAndPayment(self):
        self._ConstructDS()
        profile = []
        for p in self.providers:
            profile.append([self.providers[p].idx,
                            self.providers[p].cost,
                            self.providers[p].tasks])
        profile.append(['virtual', self.requester.reserved_p, self.tasks_num])
        profile.sort(key=lambda y: y[1])
        # global allocation 
        i = 0
        num = self.tasks_num
        dvcg_allocation = _CalculateSocialCost(profile, num)
        for w in dvcg_allocation:
            self.winners[w] = dvcg_allocation[w][1]

        for winner in self.winners:
            if winner != 'virtual':
                self.total_social_cost += self.winners[winner] * self.providers[winner].cost
            else:
                self.total_social_cost += self.winners[winner] * self.requester.reserved_p

        for provider in self.providers:
            d_p = {provider}
            if provider in self.dominate_set:
                for node in self.dominate_set[provider]:
                    d_p.add(node)
            profile_p = [z for z in profile if z[0] not in d_p]
            cur_num = self.tasks_num
            cur_allocation = _CalculateSocialCost(profile_p, cur_num)
            cur_sc = sum(x * y for x, y in cur_allocation.values())
            cur_payment = 0
            if provider in self.winners:
                cur_payment = cur_sc - (self.total_social_cost - self.winners[provider] * self.providers[provider].cost)
            else:
                cur_payment = cur_sc - self.total_social_cost
            if cur_payment != 0: self.payments[provider] = cur_payment

    def AllocationResult(self):
        return self.winners

    def PaymentResult(self):
        return self.payments

    def TotalSocialCost(self):
        return self.total_social_cost

    def TotalPayment(self):
        return sum(self.payments.values())

    def BudgetFeasible(self):
        total_cost = sum(self.payments.values())
        return True if self.budgets >= total_cost else False




class DiffCRAHM:
    def __init__(self,tasks,requester,providers,graph) -> None:
        self.tasks_num = tasks
        self.requester = requester 
        self.providers = providers 
        self.winners = dict()
        self.payments = dict()
        self.graph = graph 
        self.cr_tree = None 
        self.competitors = {}
        self.submarkets = collections.defaultdict(set)
        self.max_dist = 0
        self.total_social_cost = 0

    def NetworkedMarketDivision(self):
        for p in self.providers:
            dist = nx.shortest_path_length(self.graph,0,p)
            self.submarkets[dist].add(p)
        self.max_dist = max(self.submarkets.keys())
    
    def ShowSubmarkets(self):
        return self.submarkets
    
    def ShowProviders(self):
        return [[p.idx,p.cost,p.tasks] for p in self.providers.values()]


    def AllocationAndPayment(self):
        self.NetworkedMarketDivision()
        tau = self.tasks_num
        for d in range(1,self.max_dist+1):
            if tau == 0:
                break 
            # print('tau',tau)
            sm_ast = [s for s in self.submarkets[d] if self.providers[s].cost < self.requester.reserved_p]
            # print('current market participants:',sm_ast)
            sm_total = sum([self.providers[a].tasks for a in sm_ast])
            if tau >= sm_total:
                # print(sm_total)
                for p in sm_ast:
                    self.winners[p] = self.providers[p].tasks 
                    self.payments[p] = self.providers[p].tasks * self.requester.reserved_p 
                tau -= sm_total
            else:
                # oversupply in current market 
                # sm_ast
                # print('current market and left num:',sm_ast,tau)
                costs = [[s,self.providers[s].cost,self.providers[s].tasks] for s in sm_ast]
                costs.append(['virtual',self.requester.reserved_p,self.tasks_num])
                winners_sm_ast = _CalculateSocialCost(costs,tau)
                cnt = tau
                for w in winners_sm_ast:
                    self.winners[w] = winners_sm_ast[w][1]
                    tau -= self.winners[w]
                    self.payments[w] = _SocialCostIncrease(costs,w,cnt)
        if tau > 0:
            self.winners['virtual'] = tau 
            self.payments['virtual'] = self.requester.reserved_p * tau 
                
    
    def AllocationResult(self):
        return self.winners 
    
    def PaymentResult(self):
        return self.payments 
    
    def OutsourceItems(self):
        cnt = 0
        for num in self.winners.values():
            cnt += num
        return cnt 

    def TotalSocialCost(self):
        self.total_social_cost = 0
        for w in self.winners:
            if w == 'virtual':
                self.total_social_cost += self.requester.reserved_p * self.winners[w]
            else:
                self.total_social_cost += self.winners[w] * self.providers[w].cost
        return self.total_social_cost

    def TotalPayment(self):
        return sum(self.payments.values())

    def BudgetFeasible(self):
        total_payment = sum(self.payments.values())
        return True if self.budgets >= total_payment else False