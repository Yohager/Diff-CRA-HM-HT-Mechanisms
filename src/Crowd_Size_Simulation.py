import os 
import networkx as nx 
import random 
import time 
from Network_Load import read_network_data
from Core_Crowd_HM import NDVCG, DVCG, DiffCRAHM, New_RandomizeValuation_HM
from attributes import RequesterHM, Provider
from plot_figs import plot_fig_homo, plot_fig_hete
from Core_Crowd_HT import DiffCRAHT, Initialization, Initialization_normal_distribution


'''
compare markets with different sizes of suppliers over scale-free graph models 
'''

def SupplierSizeSimulation_Homo(iterations):
    sup_n = [40,80,120]
    rp = random.randint(5,10)
    data = []
    # probs = [0.01*x for x in range(5,31)]
    tasks = [x for x in range(10,501,20)]
    for sn in sup_n:
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
                rand_graph = nx.scale_free_graph(sn+1, 0.85, 0.1, 0.05)
                rand_graph = rand_graph.to_undirected()
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

def SupplierSizeSimulation_Hete(times):
    # tasks = [100,200,500]
    minv,maxv = 1,10
    sup_n = [40,80,120]
    # prob = 0.1
    # probs = [0.01*x for x in range(5,31)]
    # sup_n = [x for x in range(10,101,5)]
    sa_minv,sa_maxv = 2,10
    sc_minv,sc_maxv = 5,20  
    tasks = [x for x in range(10,500,20)]
    data = []
    budgs = []
    # for ts,sn in zip(tasks,sup_n): 
    for sn in sup_n:
        # (100,100) / (1000,1000) / (2000,1000)
        cur_sc = [] #cur_sc[0]: mechanism1, cur_sc[1]: mechanism2 
        cur_pay = []
        # for cur_sc[i]: cur_sc[i][0]: avg, cur_sc[i][1]: min, cur_sc[i][2]: max 
        avg_1,avg_2,avg_3,avg_4 = [],[],[],[]
        min_1,min_2,min_3,min_4 = [],[],[],[]
        max_1,max_2,max_3,max_4 = [],[],[],[]
        bud = []
        for ts in tasks:
            # run times and cals the avg,max,min val
            avg_sc = [0,0]
            max_sc = [0,0]
            min_sc = [float('inf'),float('inf')]
            avg_pay = [0,0]
            max_pay = [0,0]
            min_pay = [float('inf'),float('inf')]
            budget = 0
            for _ in range(times):
                # g = _RandomGraph(sn,prob) # generate random graph 
                g = nx.scale_free_graph(sn+1, 0.85, 0.1, 0.05)
                g = g.to_undirected()
                # vals,req,sups = Initialization(ts,minv,maxv,sn,sa_minv,sa_maxv,sc_minv,sc_maxv)
                vals, req, sups = Initialization_normal_distribution(0,ts,5,2,minv, maxv, 0, sn, sa_minv, sa_maxv, sc_minv, sc_maxv)
                M_local = DiffCRAHT(ts,g,sn,req,sups,vals)
                M_global = DiffCRAHT(ts,g,sn,req,sups,vals)
                M_local.Implementation('local')
                M_global.Implementation('global')
                avg_sc[0] += M_local.TotalSocialCost()
                avg_sc[1] += M_global.TotalSocialCost()
                max_sc[0] = max(max_sc[0],M_local.TotalSocialCost())
                max_sc[1] = max(max_sc[1],M_global.TotalSocialCost())
                min_sc[0] = min(min_sc[0],M_local.TotalSocialCost())
                min_sc[1] = min(min_sc[1],M_global.TotalSocialCost())
                avg_pay[0] += M_local.TotalPayment()
                avg_pay[1] += M_global.TotalPayment()
                max_pay[0] = max(max_pay[0],M_local.TotalPayment())
                max_pay[1] = max(max_pay[1],M_global.TotalPayment())
                min_pay[0] = min(min_pay[0],M_local.TotalPayment())
                min_pay[1] = min(min_pay[1],M_global.TotalPayment())
                budget += M_global.TotalBudget()
            print('here the average social cost:',avg_sc)
            avg_sc[0] = avg_sc[0] / times 
            avg_sc[1] = avg_sc[1] / times 
            avg_pay[0] = avg_pay[0] / times 
            avg_pay[1] = avg_pay[1] / times
            budget = budget / times 
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
            bud.append(budget)         
        cur_sc.append([avg_1,min_1,max_1])
        cur_sc.append([avg_2,min_2,max_2])
        cur_pay.append([avg_3,min_3,max_3])
        cur_pay.append([avg_4,min_4,max_4])
        data.append([cur_sc,cur_pay])
        budgs.append(bud)
    return tasks,data,budgs 

if __name__ == "__main__":
    iterations = 10
    print('*'*10 + 'Homogeneous Scenario' + '*'*10)
    x_label_t = "the amount of tasks"
    title = "crowd-iot-hm-tasks-scale"
    axtitle_t = [
        'Providers Number: 40',
        'Providers Number: 80',
        'Providers Number: 120'
    ]
    xvals,data = SupplierSizeSimulation_Homo(iterations)
    plot_fig_homo(xvals,data,x_label_t,axtitle_t, title)

    print('*'*30)
    
    print('*'*10 + 'Heterogeneous Scenario' + '*'*10)
    x_label_t = "the amount of tasks"
    title = "crowd-iot-ht-tasks-scale"
    axtitle_t = [
        'Providers Number: 40',
        'Providers Number: 80',
        'Providers Number: 120'
    ]
    xvals,data,budgs = SupplierSizeSimulation_Hete(iterations)
    plot_fig_hete(xvals, data, budgs, x_label_t,axtitle_t, title)