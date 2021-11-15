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
reserved_price = 10
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


def _RandomizeValuation_HM(rp, num):
    '''
    :param num: 生成多少个节点以及对应的属性
    :return: 给出一个requester类实例，给出一个list中包含num数量的providers的实例
    '''
    # 先生成requester类实例
    req = RequesterHM(rp)
    ps = {}
    for i in range(1, num + 1):
        ri = random.randint(1, TASK_NUM // 3)
        pi = random.randint(1, 10)
        p = Provider(i, pi, ri)
        ps[i] = p
    return req, ps


'''
定义几个class
class RequesterHM: DCM-HM机制的requester
class RequesterHT: DCM-HT机制的requester
class Provider: DCM-HM和DCM-HT机制的providers
'''


class RequesterHM:
    def __init__(self, price):
        self.idx = 0
        self.tasks_num = TASK_NUM
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


def _FixCompetitors(graph, target):
    g = copy.deepcopy(graph)
    # 返回值为所有的competitors的idx的集合
    res = set()
    all_nodes = set(g.nodes)
    g.remove_node(target)
    if not nx.is_connected(g):
        # 出现不连通的情况则表示当前这个节点存在支配的节点
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


def _SocialCostMinimum(p, k, target):
    '''
    :param p: providers的type集合
    :param k: 剩余的tasks的数量
    :param target: 判断是否成为winner
    :return: 是否成为winner
    '''
    winners = []
    n = len(p)
    # p[i][0]: idx, p[i][1]: unit-cost, p[i][2]: amount
    p.sort(key=lambda x: x[1])
    # print('rank p:', p)
    idx = 0
    while idx < n and k >= 0:
        # 当前没有到最后一个人或者还有task没有分配完成
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
    print('social cost increase:', target, sc_without_target - sc_opt)
    return sc_without_target - sc_opt


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
        # print(profile)
        # 按照unit cost从小到大排序的到分配的顺序
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
        # 全局进行分配
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


class DiffusionCrowdsensingMechanismHM:
    def __init__(self, requester, providers, tasks, graph):
        """
        :param requester: Requester对象实例
        :param providers: Providers对象实例的字典
        :param tasks: 总共的tasks的数量
        :param graph: 信息传播图，以requester为根节点，包含所有的providers
        """
        self.tasks_num = tasks
        self.requester = requester
        self.providers = providers
        self.winners = {}
        self.payments = {}
        self.graph = graph
        self.cr_tree = None
        # self.root = None
        self.competitors = {}
        self.total_social_cost = 0
        self.budgets = self.requester.reserved_p * self.tasks_num

    def _AddVirtualNode(self):
        # n = len(self.providers)
        self.cr_tree.add_node('virtual')
        self.cr_tree.add_edge(0, 'virtual')

    def _ConstructDiffusionTree(self):
        # self._graph_to_cr_tree()
        # self._AddVirtualNode()
        # self.cr_tree = self.graph
        self._graph_to_cr_tree()

    def _graph_to_cr_tree(self):
        """
        To creat the cr_tree of a referral graph,
        where agents are represented by integers and '0' represents the principal (change in PP file).
        """
        principal = 0
        network = self.graph
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
        self.cr_tree = cr_tree.to_undirected()

    def _CalculateDistance(self):
        for provider in self.providers:
            self.providers[provider].distance = nx.shortest_path_length(self.graph, 0, self.providers[provider].idx)

    def AllocationAndPayment(self):
        """
        firstly order all nodes with distance to the root node
        secondly create the critical diffusion tree
        thirdly add virtual node into the critical diffusion tree
        for each provider, decide whether she could be one winner
        """
        self._CalculateDistance()
        self._ConstructDiffusionTree()
        distances = []
        for provider in self.providers:
            distances.append([self.providers[provider].idx,
                              self.providers[provider].distance,
                              self.providers[provider].cost,
                              self.providers[provider].tasks])
        distances.sort(key=lambda x: x[1])
        # print(distances)
        i = 0
        task_num = self.tasks_num
        # _ShowGraph(self.graph)
        # _ShowGraph(self.cr_tree)
        while i < len(self.providers) and task_num > 0:
            cur_id = distances[i][0]
            # 如果单位cost大于reserved price直接退出，将剩余的所有的tasks分配给virtual node
            if self.providers[cur_id].cost > self.requester.reserved_p:
                self.winners['virtual'] = task_num
                break
            # 明确每一个节点的竞争对手
            # 调用_FixCompetitors()函数，返回所有竞争者的集合
            competitors = _FixCompetitors(self.cr_tree, cur_id)
            for w in self.winners.keys():
                if w in competitors:
                    competitors.remove(w)
            profile = [[self.providers[cur_id].idx, self.providers[cur_id].cost, self.providers[cur_id].tasks]]
            for c in competitors:
                profile.append([self.providers[c].idx, self.providers[c].cost, self.providers[c].tasks])
            # 需要在competitors中添加一个virtual node用于控制budget
            profile.append(['virtual', self.requester.reserved_p, self.tasks_num])
            print(task_num)
            print('here profile:', profile)
            # if cur_id in _CalculateSocialCost(profile, task_num):
            # if task_num >= self.providers[cur_id].tasks:
            #     self.winners[cur_id] = self.providers[cur_id].tasks
            #     # 计算对应当前winner的payment
            # else:
            #     self.winners[cur_id] = task_num
            # self.payments[cur_id] = _SocialCostIncrease(profile, cur_id, task_num)
            #
            # task_num -= self.winners[cur_id]
            sc_cur = _CalculateSocialCost(profile, task_num)
            if cur_id in sc_cur:
                self.winners[cur_id] = sc_cur[cur_id][1]
                self.payments[cur_id] = _SocialCostIncrease(profile, cur_id, task_num)
                task_num -= sc_cur[cur_id][1]
            i += 1

    def AllocationResults(self):
        return self.winners

    def PaymentResults(self):
        return self.payments

    def TotalSocialCost(self):
        for w in self.winners:
            self.total_social_cost += self.winners[w] * self.providers[w].cost
        return self.total_social_cost

    def TotalPayment(self):
        return sum(self.payments.values())

    def BudgetFeasible(self):
        total_payment = sum(self.payments.values())
        return True if self.budgets >= total_payment else False


# class DiffusionCrowdsensingMechanismHT:
#     def __init__(self):
#         self.winners = {}
#         self.payments = {}
#
#     def AllocationRule(self):
#         pass

if __name__ == "__main__":
    # 随机生成一棵树
    rand_tree = _RandomizeTree(PROVIDERS_NUM + 1,101)
    requester_neighbors = list(nx.all_neighbors(rand_tree, 0))
    _ShowGraph(rand_tree)
    # 随机生成requester类实例和providers的类实例
    requester_hm, providers_hm = _RandomizeValuation_HM(reserved_price, PROVIDERS_NUM)
    # 创建一个DCM-HM的实例
    dcm_hm = DiffusionCrowdsensingMechanismHM(requester_hm, providers_hm, TASK_NUM, rand_tree)
    dcm_hm.AllocationAndPayment()
    print('dcm-hm allocation:{allocation} and payment:{payment}'.format(allocation=dcm_hm.AllocationResults(),
                                                                        payment=dcm_hm.PaymentResults()))

    # No diffusion VCG result
    providers_nd_vcg = {}
    for p in providers_hm:
        if p in requester_neighbors:
            providers_nd_vcg[p] = providers_hm[p]

    requester_nd_vcg = requester_hm
    nd_vcg = NDVCG(requester_nd_vcg, providers_nd_vcg, TASK_NUM)
    nd_vcg.AllocationAndPayment()
    print('no-diffusion-vcg allocation:{allocation} and payment:{payment}'.format(allocation=nd_vcg.AllocationResult(),
                                                                                  payment=nd_vcg.PaymentResult()))
    requester_diffusion_vcg = requester_hm
    providers_diffusion_vcg = providers_hm
    diffusion_vcg = DVCG(requester_diffusion_vcg, providers_diffusion_vcg, TASK_NUM, rand_tree)
    diffusion_vcg.AllocationAndPayment()
    print('diffusion-vcg allocation:{allocation} and payment:{payment}'.format(
        allocation=diffusion_vcg.AllocationResult(),
        payment=diffusion_vcg.PaymentResult()))
    print('===' * 20)
    print('no-diffusion-vcg total payment: {nd_vcg_tp}, '
          'diffusion-vcg total payment: {d_vcg_tp}, '
          'dcm-hm total payment: {dcm_hm_tp}'.format(
        nd_vcg_tp=nd_vcg.TotalPayment(),
        d_vcg_tp=diffusion_vcg.TotalPayment(),
        dcm_hm_tp=dcm_hm.TotalPayment()))
    print('no-diffusion-vcg total social cost: {nd_vcg_tsc}, '
          'diffusion-vcg total social cost: {d_vcg_tsc}, '
          'dcm-hm total social cost: {dcm_hm_tsc}'.format(
        nd_vcg_tsc=nd_vcg.TotalSocialCost(),
        d_vcg_tsc=diffusion_vcg.TotalSocialCost(),
        dcm_hm_tsc=dcm_hm.TotalSocialCost()))
