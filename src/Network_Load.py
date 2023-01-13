import networkx as nx 


'''
read social network dataset 
'''

def read_network_data(path):
    edges = []
    with open(path) as f:
        for line in f.readlines():
            edges.append(list(map(int,line.split())))
    return nx.Graph(edges)