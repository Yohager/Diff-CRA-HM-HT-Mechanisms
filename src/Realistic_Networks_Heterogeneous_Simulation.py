import imp
import os 
import networkx as nx 
import random 
import time 
from Network_Load import read_network_data
from Core_Crowd_HT import DiffCRAHT 
from attributes import RequesterHM, Provider
from Core_Crowd_HT import Initialization, Initialization_normal_distribution


def social_network_simulation(graph, task_num):
    # requester 生成一个长度为task_num的vector 
    total_num = graph.number_of_nodes() 
    start_id = 0 if 0 in graph.nodes else 1 
    host_id = random.randint(start_id, total_num-1+start_id)
    avgv = 8 
    sigma = 2 
    minv, maxv = 1, 15 
    min_a, max_a = 2, 10 
    min_c, max_c = 5, 30
    vals, requester, providers = Initialization_normal_distribution(host_id, \
                                task_num, avgv, sigma, minv, maxv, start_id,\
                                total_num, min_a, max_a, min_c, max_c)
    # vals, requester, providers = Initialization_normal_distribution(host_id, task_num, avgv, sigma, minv, maxv, min_a, max_a, min_c, max_c)
    M_local = DiffCRAHT(task_num,graph, total_num - 1, requester, providers, vals)
    M_global = DiffCRAHT(task_num,graph, total_num - 1, requester, providers, vals)
    start_time = time.time()
    M_local.Implementation('local')
    M_global.Implementation('global')
    end_time = time.time()
    print('Total Budget: ', sum(vals))
    print('Total Social Cost: ', M_local.TotalSocialCost(), M_global.TotalSocialCost())
    print('Total Payment: ', M_local.TotalPayment(), M_global.TotalPayment())
    print('Total Time Cost: ', end_time - start_time)


if __name__ == "__main__":
    ROOT = "../datasets/"
    files_paths = list(os.walk(ROOT))[0][-1]
    ts_files = [20000, 50000]
    for f in files_paths:
        print('current file name is:', f)
        for ts in ts_files:
            print('*'*10 + f'current task num: {ts}' + '*'*10)
            social_network_simulation(ROOT + f, ts)