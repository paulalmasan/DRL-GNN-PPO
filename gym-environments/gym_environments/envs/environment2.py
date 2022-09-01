import gym
import numpy as np
import networkx as nx
import random
import sys
from gym import error, spaces, utils
from random import choice
import pylab
import json 
import gc
import matplotlib.pyplot as plt

DISTANCE_WEIGHT_NAME = "capacity"  # The name for the distance edge attribute.

def create_nsfnet_graph():

    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    #nx.draw(Gbase, with_labels=True)
    #plt.show()
    #plt.clf()

    return Gbase

def create_geant_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12,19), (12,21),
         (14, 15), (15, 16), (16, 17), (17,18), (18,21), (19, 23), (21,22), (22, 23)])

    # nx.draw(Gbase, with_labels=True)
    # plt.show()
    # plt.clf()

    return Gbase

def create_gbn_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    Gbase.add_edges_from(
        [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
         (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
         (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

    # nx.draw(Gbase, with_labels=True)
    # plt.show()
    # plt.clf()

    return Gbase

def generate_nx_graph(topology):
    """Generate graphs for training with the same topology but different weights.

  Args:
    rand: A random seed (np.RandomState instance).

  Returns:
    graphs: The list of input graphs.
    targets: The list of output targets (i.e. sum of edges).
  """
    if topology == 0:
        G = create_nsfnet_graph()
    elif topology == 1:
        G = create_geant_graph()
    else:
        G = create_gbn_graph()

    incId = 1
    # Put all distance weights into edge attributes.
    for i, j in G.edges():
        G.get_edge_data(i, j)['edgeId'] = incId
        G.get_edge_data(i, j)['betweenness'] = 0
        G.get_edge_data(i, j)['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        G.get_edge_data(i, j)[DISTANCE_WEIGHT_NAME] = float(200)
        G.get_edge_data(i, j)['bw_allocated'] = 0
        incId = incId + 1

    return G

class Env2(gym.Env):
    """
    Environment used to evaluate the RAND and SAP and compare with PPO agent
    """
    def __init__(self):
        self.graph = None
        self.initial_state = None
        self.reward = None
        self.max_demand = None

        self.K = 4
        self.nodes = None
        self.ordered_edges = None
        self.edgesDict = None
        self.numNodes = None
        self.numEdges = None

        self.state = None
        self.allPaths = dict()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def increment_shortest_path(self):
        # Iterate over all node1,node2 pairs from the graph
        for n1 in self.graph:
            for n2 in self.graph:
                if (n1 != n2):
                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                        currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
                        i = 0
                        j = 1

                        while (j < len(currentPath)):
                            self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] = \
                                self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] + 1
                            i = i + 1
                            j = j + 1

                        path = path + 1

    def num_shortest_path(self, topology):
        self.diameter = nx.diameter(self.graph)
        if topology == 5:
            self.allPaths = json.load(open('/home/paul/Documents/GNN/GNNtest/reinflearn/tf2.0Code/GBN50nodesShortPaths.json'))
            self.increment_shortest_path()
        
        else:
            # Iterate over all node1,node2 pairs from the graph
            for n1 in self.graph:
                for n2 in self.graph:
                    if (n1 != n2):
                        # Check if we added the element of the matrix
                        if str(n1)+':'+str(n2) not in self.allPaths:
                            self.allPaths[str(n1)+':'+str(n2)] = []
                        
                        # First we compute the shortest paths taking into account the diameter
                        [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter*2)]

                        # We take all the paths from n1 to n2 and we order them according to the path length
                        # sorted() ordena los paths de menor a mayor numero de
                        # saltos y los que tienen los mismos saltos te los ordena por indice
                        self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))

                        path = 0
                        while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                            currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
                            i = 0
                            j = 1

                            # Iterate over pairs of nodes and allocate linkDemand
                            while (j < len(currentPath)):
                                self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] = \
                                    self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] + 1
                                i = i + 1
                                j = j + 1

                            path = path + 1

                        # Remove paths not needed
                        del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                        gc.collect()
    
    def generate_environment(self, topology, listofdemands):
        # The nx graph will only be used to convert graph from edges to nodes
        self.graph = generate_nx_graph(topology)

        # Compute number of shortest paths per link. This will be used for the betweenness
        self.num_shortest_path(topology)

        self.max_demand = np.amax(listofdemands)

        self.edgesDict = dict()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
        self.ordered_edges = sorted(some_edges_1)

        self.graph_state = np.zeros((self.numEdges, 2))

        position = 0
        for i, j in self.ordered_edges:
            # Normalize link betweenness
            self.edgesDict[str(i)+':'+str(j)] = position
            self.edgesDict[str(j)+':'+str(i)] = position
            self.graph_state[position][0] = 200.0
            position = position + 1

        self.initial_state = np.copy(self.graph_state)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))
        return

    def make_step(self, negCap, demand):
        self.episode_over = True
        self.reward = 0

        # CHECK LINKS CAPACITY >=0
        if negCap:
            return self.reward/self.max_demand, self.episode_over

        # Reward is the allocated demand or 0 otherwise (end of episode)
        self.reward = demand/self.max_demand
        self.episode_over = False

        return self.reward, self.episode_over
    
    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        self.graph_state = np.copy(self.initial_state)

        return self.graph_state