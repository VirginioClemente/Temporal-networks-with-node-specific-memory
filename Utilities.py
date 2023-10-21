import numpy as np
import networkx as nx
import re
from operator import itemgetter

import math as m
import scipy.optimize as spo
from scipy.optimize import least_squares
import scipy
import os


def to_1(n):
    if n >= 1 and n != 0.:
        return 1
    else:
        return 0


def matrix_to_1(B):
    H = np.zeros(np.shape(B))
    for i in range(len(B)):
        H[i] = [to_1(el) for el in B[i]]

    return H

# Classe con cui definiamo la rete da dare in input al modello

class TemporalNetwork():
    def __init__(self, no_nodes, no_steps):
        self.no_nodes = no_nodes
        self.no_steps = no_steps

        self.verbose = False


class TemporalNetworkLoaderSynt(TemporalNetwork):

    def __init__(self, data_synt, start, stop, symmetric=True):
        self.data_synt = data_synt
        self.symmetric = symmetric
        self.start = start
        self.stop = stop
        self.__load()

    def __load(self):
        self.data = []
        self.nodes = set()

        self.no_steps = 0

        for el in self.data_synt:
            # h = filename.split("_")
            block_id = el[0]
            current_time = el[0]

            self.curr_id = block_id
            self.no_steps += 1

            # dati = el[1]
            curr_net = el[1]  # nx.from_numpy_matrix(np.matrix(dati), create_using=nx.DiGraph)
            if self.symmetric:
                curr_net = curr_net.to_undirected()
            self.nodes.update(curr_net.nodes())
            self.data.append([block_id, current_time, curr_net])
            self.data = sorted(self.data,
                               key=itemgetter(1))  # Ho messo in ordine crescente(rispetto al tempo) le reti lette.
        self.data = self.data[self.start:self.stop]
        # self.data = self.data[:]
        self.no_nodes = max(self.nodes) + 1

    def __iter__(self):
        for x in self.data:
            yield x


def AIC(likelihood, N_parameters):
    return (2*N_parameters) - (2*likelihood)