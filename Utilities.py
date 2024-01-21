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
        # Initialization with synthetic data and configuration parameters.
        """
        data_synt: This is a collection of data representing different states or snapshots of a temporal network.
        start and stop: These parameters define the range of time steps or snapshots of the network to be considered.
        """
        self.data_synt = data_synt
        self.symmetric = symmetric
        self.start = start
        self.stop = stop
        self.__load()

    def __load(self):
        # Initializing properties to store network data and statistics.
        self.data = []
        self.nodes = set()

        self.no_steps = 0

        # Processing each element in the synthetic data.
        for el in self.data_synt:
            block_id = el[0]
            current_time = el[0]

            self.curr_id = block_id
            self.no_steps += 1

             # Loading the network data for the current time step.
            curr_net = el[1]  
            if self.symmetric:
                curr_net = curr_net.to_undirected()
            self.nodes.update(curr_net.nodes())
            self.data.append([block_id, current_time, curr_net])
            self.data = sorted(self.data,
                               key=itemgetter(1))   # Sorting the data by time and slicing based on start and stop parameters.
        self.data = self.data[self.start:self.stop]
        self.no_nodes = max(self.nodes) + 1

    def __iter__(self):
        # Making the class iterable, yielding each network snapshot in order.
        for x in self.data:
            yield x


def AIC(likelihood, N_parameters):
    return (2*N_parameters) - (2*likelihood)
