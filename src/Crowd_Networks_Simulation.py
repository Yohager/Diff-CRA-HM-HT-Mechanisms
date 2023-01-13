import os 
import networkx as nx 
import random 
import time 
from Network_Load import read_network_data
from Core_Crowd_HM import NDVCG, DVCG, DiffCRAHM, New_RandomizeValuation_HM
from attributes import RequesterHM, Provider
from plot_figs import plot_fig_homo, plot_fig_hete
from Core_Crowd_HT import DiffCRAHT, Initialization, Initialization_normal_distribution

def RepresentativeNetworksSimulation_Homo(iterations):
    sn = 80
    types = [1,2,3]
    rp = 10
    data = []
    # probs = [0.01*x for x in range(5,31)]
    tasks = [x for x in range(100,501,20)]
    for t in types:
        cur_sc, cur_pay = [], []
        avg_1,avg_2,avg_3 = [],[],[]
        avg_p1,avg_p2,avg_p3 = [],[],[]
        max_1,max_2,max_3 = [],[],[]
        max_p1,max_p2,max_p3 = [],[],[]
        min_1,min_2,min_3 = [],[],[]
        min_p1,min_p2,min_p3 = [],[],[]
        for ts in tasks:
            avg_sc = [0,0,0]
            max_sc = [0,0,0]
            min_sc = [float('inf'),float('inf'),float('inf')]
            avg_pay = [0,0,0]
            max_pay = [0,0,0]
            min_pay = [float('inf'),float('inf'),float('inf')]
            for _ in range(iterations):
                rand_graph = None 
                if t == 1:
                    # rand_graph = _RandomGraph(sn,prob)
                    rand_graph = nx.connected_watts_strogatz_graph(sn+1, 4, 0.2)
                    while not nx.is_connected(rand_graph):
                        rand_graph = nx.connected_watts_strogatz_graph(sn+1, 4, 0.2)
                elif t == 2:
                    # rand_graph = _RandomCompleteGraph(sn)
                    rand_graph = nx.scale_free_graph(sn+1)
                    while not nx.is_connected(rand_graph.to_undirected()):
                        rand_graph = nx.scale_free_graph(sn+1)
                    rand_graph = rand_graph.to_undirected()
                else:
                    # rand_graph = _RandomizeTree(sn)
                    rand_graph = nx.erdos_renyi_graph(sn+1,0.05)
                    while not nx.is_connected(rand_graph):
                        rand_graph = nx.erdos_renyi_graph(sn+1,0.05)
                requester_hm, providers_hm = New_RandomizeValuation_HM(rp,sn,0,0)
                requester_neighbors = list(nx.all_neighbors(rand_graph,0)) # all of the requester's neighbors construct the first level market 
                providers_vcg = {p:providers_hm[p] for p in providers_hm if p in requester_neighbors}
                nd_vcg = NDVCG(requester_hm,providers_vcg,ts) # create the instance of local market with vcg 
                netram = DiffCRAHM(ts,requester_hm,providers_hm,rand_graph)
                dvcg = DVCG(requester_hm,providers_hm,ts,rand_graph)
                nd_vcg.AllocationAndPayment()
                netram.AllocationAndPayment()
                dvcg.AllocationAndPayment()
                avg_sc[0] += nd_vcg.TotalSocialCost()
                avg_sc[1] += netram.TotalSocialCost()
                avg_sc[2] += dvcg.TotalSocialCost()
                max_sc[0] = max(max_sc[0],nd_vcg.TotalSocialCost())
                max_sc[1] = max(max_sc[1],netram.TotalSocialCost())
                max_sc[2] = max(max_sc[2],dvcg.TotalSocialCost())
                min_sc[0] = min(min_sc[0],nd_vcg.TotalSocialCost())
                min_sc[1] = min(min_sc[1],netram.TotalSocialCost())
                min_sc[2] = min(min_sc[2],dvcg.TotalSocialCost())
                avg_pay[0] += nd_vcg.TotalPayment()
                avg_pay[1] += netram.TotalPayment()
                avg_pay[2] += dvcg.TotalPayment()
                max_pay[0] = max(max_pay[0],nd_vcg.TotalPayment())
                max_pay[1] = max(max_pay[1],netram.TotalPayment())
                max_pay[2] = max(max_pay[2],dvcg.TotalPayment())
                min_pay[0] = min(min_pay[0],nd_vcg.TotalPayment())
                min_pay[1] = min(min_pay[1],netram.TotalPayment())
                min_pay[2] = min(min_pay[2],dvcg.TotalPayment())    
            for i in range(3):
                avg_sc[i] = avg_sc[i] / iterations
                avg_pay[i] = avg_pay[i] / iterations 
            for idx,x in enumerate([avg_1,avg_2,avg_3]):
                x.append(avg_sc[idx])
            for idx,x in enumerate([avg_p1,avg_p2,avg_p3]):
                x.append(avg_pay[idx])          
            for idx,x in enumerate([min_1,min_2,min_3]):
                x.append(min_sc[idx])
            for idx,x in enumerate([min_p1,min_p2,min_p3]):
                x.append(min_pay[idx])
            for idx,x in enumerate([max_1,max_2,max_3]):
                x.append(max_sc[idx])
            for idx,x in enumerate([max_p1,max_p2,max_p3]):
                x.append(max_pay[idx])
        cur_sc.append([avg_1,min_1,max_1])        
        cur_sc.append([avg_2,min_2,max_2])
        cur_sc.append([avg_3,min_3,max_3])
        cur_pay.append([avg_p1,min_p1,max_p1])
        cur_pay.append([avg_p2,min_p2,max_p2])
        cur_pay.append([avg_p3,min_p3,max_p3])
        data.append([cur_sc,cur_pay])
    return tasks,data 


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
    budget = sum(vals)
    print('Total Budget: ', sum(vals))
    print('Total Social Cost: ', M_local.TotalSocialCost(), M_global.TotalSocialCost())
    print('Total Payment: ', M_local.TotalPayment(), M_global.TotalPayment())
    print('Total Time Cost: ', end_time - start_time)
    return budget, M_local.TotalSocialCost(), M_global.TotalSocialCost(), M_local.TotalPayment(), M_global.TotalPayment()

def RepresentativeNetworksSimulation_Hete(iterations):
    sn = 100 
    tasks = list(range(50, 301, 20))
    data = []
    budgs = []
    types = [1,2,3]
    for t in types:
        cur_sc = [] #cur_sc[0]: mechanism1, cur_sc[1]: mechanism2 
        cur_pay = []
        # for cur_sc[i]: cur_sc[i][0]: avg, cur_sc[i][1]: min, cur_sc[i][2]: max 
        avg_1,avg_2,avg_3,avg_4 = [],[],[],[]
        min_1,min_2,min_3,min_4 = [],[],[],[]
        max_1,max_2,max_3,max_4 = [],[],[],[]
        budgets = []
        for ts in tasks:
            avg_sc = [0,0]
            max_sc = [0,0]
            min_sc = [float('inf'),float('inf')]
            avg_pay = [0,0]
            max_pay = [0,0]
            min_pay = [float('inf'),float('inf')]  
            bud = 0                  
            for _ in range(iterations):
                rand_graph = None 
                if t == 1:
                    # rand_graph = _RandomGraph(sn,prob)
                    rand_graph = nx.connected_watts_strogatz_graph(sn+1, 4, 0.2)
                    while not nx.is_connected(rand_graph):
                        rand_graph = nx.connected_watts_strogatz_graph(sn+1, 4, 0.2)
                elif t == 2:
                    # rand_graph = _RandomCompleteGraph(sn)
                    rand_graph = nx.scale_free_graph(sn+1)
                    while not nx.is_connected(rand_graph.to_undirected()):
                        rand_graph = nx.scale_free_graph(sn+1)
                    rand_graph = rand_graph.to_undirected()
                else:
                    # rand_graph = _RandomizeTree(sn)
                    rand_graph = nx.erdos_renyi_graph(sn+1,0.05)
                    while not nx.is_connected(rand_graph):
                        rand_graph = nx.erdos_renyi_graph(sn+1,0.05)
                # requester_hm, providers_hm = New_RandomizeValuation_HM(reserved_price,sn,0,0)
                # requester_neighbors = list(nx.all_neighbors(rand_graph,0)) # all of the requester's neighbors construct the first level market 
                budget, local_sc, global_sc, local_pay, global_pay = \
                    social_network_simulation(rand_graph, ts)
                # print('here!:',budget, local_sc, global_sc, local_pay, global_pay)
                bud += budget
                avg_sc[0] += local_sc
                avg_sc[1] += global_sc 
                max_sc[0] = max(max_sc[0], local_sc)
                max_sc[1] = max(max_sc[1], global_sc)
                min_sc[0] = min(min_sc[0], local_sc)
                min_sc[1] = min(min_sc[1], global_sc)
                avg_pay[0] += local_pay
                avg_pay[1] += global_pay 
                max_pay[0] = max(max_pay[0], local_pay)
                max_pay[1] = max(max_pay[1], global_pay)
                min_pay[0] = min(min_pay[0], local_pay)
                min_pay[1] = min(min_pay[1], global_pay)
            avg_sc[0] /= iterations
            avg_sc[1] /= iterations
            avg_pay[0] /= iterations
            avg_pay[1] /= iterations
            bud /= iterations
            avg_1.append(avg_sc[0])
            avg_2.append(avg_sc[1])
            min_1.append(min_sc[0])
            min_2.append(min_sc[1])
            max_1.append(max_sc[0])
            max_2.append(max_sc[1])
            avg_3.append(avg_pay[0])
            avg_4.append(avg_pay[1])
            min_3.append(min_pay[0])
            min_4.append(min_pay[1])
            max_3.append(max_pay[0])
            max_4.append(max_pay[1])   
            budgets.append(bud)         
        cur_sc.append([avg_1,min_1,max_1])
        cur_sc.append([avg_2,min_2,max_2])
        cur_pay.append([avg_3,min_3,max_3])
        cur_pay.append([avg_4,min_4,max_4])
        data.append([cur_sc,cur_pay])
        budgs.append(budgets)   
    print('here:', data)         
    return tasks, data, budgs

if __name__ == "__main__":
    iterations = 10
    print('*'*10 + 'Homogeneous Scenario' + '*'*10)
    x_label_t = "the amount of tasks"
    title = "crowd-iot-homo-compare-three-models"
    axtitle_t = [
        "Watts_Strogatz Model",
        "Scale_Free Model",
        "Erdos_Renyi Model"
    ]
    xvals,data = RepresentativeNetworksSimulation_Homo(iterations)
    plot_fig_homo(xvals, data, x_label_t, axtitle_t, title)
    print('*'*30)
    print('*'*10 + 'Heterogeneous Scenario' + '*'*10)
    x_label_t = "the amount of tasks"
    title = "crowd-iot-hete-compare-three-models"
    axtitle_t = [
        "Watts_Strogatz Model",
        "Scale_Free Model",
        "Erdos_Renyi Model"
    ]
    xvals, data, budgs = RepresentativeNetworksSimulation_Homo(iterations)
    plot_fig_hete(xvals, data, budgs, x_label_t, axtitle_t, title)

    