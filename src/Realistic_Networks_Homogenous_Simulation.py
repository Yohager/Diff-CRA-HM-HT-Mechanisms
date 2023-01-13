import os 
import networkx as nx 
import random 
import time 
from Network_Load import read_network_data
from Core_Crowd_HM import NDVCG, DVCG, DiffCRAHM
from attributes import RequesterHM, Provider
from Core_Crowd_HM import New_RandomizeValuation_HM


def simulation_network(file_path, ts):
    g = read_network_data(file_path) # load the dataset 
    pn = len(g.nodes()) if 0 not in g.nodes() else len(g.nodes()) - 1
    # rp = random.randint(5, 10) # reservered price 
    rp = 8 
    req_hm, sup_hm =New_RandomizeValuation_HM(rp, pn, ts)
    requester_neighbors = list(nx.all_neighbors(g,1)) # all of the requester's neighbors construct the first level market 
    providers_vcg = {p:sup_hm[p] for p in sup_hm if p in requester_neighbors}
    start_time = time.time()
    nd_vcg = NDVCG(req_hm,providers_vcg,ts) # create the instance of local market with vcg 
    diff_cra_hm = DiffCRAHM(ts,req_hm,sup_hm,g)
    dvcg = DVCG(req_hm,sup_hm,ts,g)
    nd_vcg.AllocationAndPayment()
    diff_cra_hm.AllocationAndPayment()
    dvcg.AllocationAndPayment()
    end_time = time.time()
    print('total time cost: ', end_time - start_time)
    print(nd_vcg.TotalSocialCost(), diff_cra_hm.TotalSocialCost(),dvcg.TotalSocialCost())
    print(nd_vcg.TotalPayment(), diff_cra_hm.TotalPayment(),dvcg.TotalPayment())


if __name__ == "__main__":
    ROOT = "../datasets/"
    files_paths = list(os.walk(ROOT))[0][-1]
    ts_files = [20000, 50000]
    for f in files_paths:
        print('current file name is:', f)
        for ts in ts_files:
            print('*'*10 + f'current task num: {ts}' + '*'*10)
            simulation_network(ROOT + f, ts)