import numpy as np
import networkx as nx
import re
from operator import itemgetter

import math as m
import scipy.optimize as spo
from scipy.optimize import least_squares
import scipy
import os



# Temporal Configuration Model
class NaiveNetFitness:

    def __init__(self, tgraph, symmetric=True):
        self.verbose = False
        self.temp_graph = tgraph
        self.no_steps = len(self.temp_graph.data)
        self.no_nodes = len(self.temp_graph.data[0][2].nodes())

        self.fitness = np.ones(2 * self.no_nodes, dtype=np.float64)
        self.max_likelihood = -1e20
        self.symmetric = symmetric

        self.__fill_vecs()

    def __fill_vecs(self):
        self.vec_k = np.zeros(self.no_nodes)
        self.vec_h = np.zeros(self.no_nodes)

        for i_step in range(self.no_steps):
            block_id, current_time, g = self.temp_graph.data[i_step]
            for [n1, n2] in g.edges():
                self.vec_k[int(n1)] += 1.
                self.vec_k[int(n2)] += 1.

        self.vec_k /= float(self.no_steps)

        for i_step in range(self.no_steps):  # Tolto - 1
            block_id, current_time, g0 = self.temp_graph.data[i_step]
            block_id, current_time, g1 = self.temp_graph.data[(i_step + 1) % self.no_steps]
            for [n1, n2] in g0.edges():

                if g1.has_edge(n1, n2):
                    self.vec_h[int(n1)] += 1.
                    self.vec_h[int(n2)] += 1.

        self.vec_h /= float(self.no_steps)

    def fun_likelihood(self, x):
        alpha = x[:self.no_nodes]

        term1 = 0.
        term2 = 0.
        for i_node in range(self.no_nodes):
            term1 += self.vec_k[i_node] * np.log(alpha[i_node])

        for i_node in range(self.no_nodes):
            for j_node in range(i_node):
                term2 += np.log(1 + alpha[i_node] * alpha[j_node])

        likelihood = term1 - term2

        return - likelihood

    # Here I solve the system of equations
    def solve_eq(self):

        def equations_to_solve(p, k):
            n_nodes = len(k)
            p = np.array(p)
            num_x_nonzero_nodes = np.count_nonzero(k)
            x_nonzero = p[0:num_x_nonzero_nodes]
            x = np.zeros(n_nodes)
            x[k != 0] = x_nonzero

            # Expected degrees
            k_exp = np.zeros(x.shape[0])

            for i in range(self.no_nodes):
                for j in range(self.no_nodes):
                    if i == j:
                        continue
                    k_exp[i] += (x[i] * x[j]) / (1 + x[i] * x[j])

            k_nonzero = k[k != 0]
            k_exp_nonzero = k_exp[k != 0]

            f1 = k_nonzero - k_exp_nonzero

            return np.asarray(f1)

        def numerically_solve_TN(vec_k):
            """
            Solves the TPCM numerically with least squares.
            """

            n_nodes = self.no_nodes

            # Rough estimate of initial values
            k = vec_k

            x_initial_values = np.random.rand(
                self.no_nodes)  # plus one to prevent dividing by zero np.sqrt(np.sum(k) + 1) * np.ones(self.no_nodes)

            x_initial_values = x_initial_values[k != 0]

            initial_values = x_initial_values
            boundslu = tuple([0] * len(initial_values)), tuple([np.inf] * len(initial_values))
            x_solved = least_squares(fun=equations_to_solve,
                                     x0=initial_values,
                                     jac='2-point',
                                     args=(k,),
                                     bounds=boundslu,
                                     max_nfev=500,
                                     method='trf',
                                     loss='linear',
                                     tr_solver='exact',
                                     tr_options={},
                                     verbose=1,
                                     ftol=1e-5, xtol=1e-5, gtol=1e-5)

            print(x_solved.message)

            p = x_solved.x
            p = np.array(p)
            num_x_nonzero_nodes = np.count_nonzero(k)
            x_nonzero = p[0:num_x_nonzero_nodes]
            x = np.zeros(self.no_nodes)
            x[k != 0] = x_nonzero

            x_array = x

            return x_array

        self.fitness_x = numerically_solve_TN(self.vec_k)

    # Here I optimize loglikelihood
    def solve_likelihood(self):

        fitness_0 = np.random.random(self.no_nodes) + 20
        bnds = [(1e-9, None) for i in range(self.no_nodes)]
        fitness = spo.minimize(self.fun_likelihood, fitness_0, method='COBYLA', bounds=bnds, options={'maxiter': 30000})

        self.max_likelihood = -1e20
        print(fitness['message'])
        self.reduced_fitness = fitness.x

        self.fitness_x = self.reduced_fitness[:self.no_nodes]

        self.likelihood = fitness['fun']

        self.message = fitness['message']
