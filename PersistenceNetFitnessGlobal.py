import numpy as np
import networkx as nx
from operator import itemgetter

import scipy.optimize as spo

class PersistenceNetFitnessGlobal:

    def __init__(self, tgraph, symmetric=True):
        self.verbose = False
        self.temp_graph = tgraph
        self.no_steps = len(self.temp_graph.data)
        self.no_nodes = len(self.temp_graph.data[0][2].nodes())

        self.fitness = np.ones(self.no_nodes + 1)
        self.symmetric = symmetric

        self.B = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.J = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.lambda_minus = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.lambda_plus = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)

        self.__update_bjlambda(self.fitness)
        self.__fill_vecs()

    def __fill_vecs(self):
        self.vec_k = np.zeros(self.no_nodes)
        self.vec_h = np.zeros(self.no_nodes)
        self.average_h = np.zeros(1)
        self.vec_h_t = np.zeros((self.no_steps, self.no_nodes))

        for i_step in range(self.no_steps):
            block_id, current_time, g = self.temp_graph.data[i_step]
            for [n1, n2] in g.edges():
                self.vec_k[int(n1)] += 1.
                self.vec_k[int(n2)] += 1.

        self.vec_k /= float(self.no_steps)

        for i_step in range(self.no_steps):
            block_id, current_time, g0 = self.temp_graph.data[i_step]
            block_id, current_time, g1 = self.temp_graph.data[(i_step + 1) % self.no_steps]
            for [n1, n2] in g0.edges():

                if g1.has_edge(n1, n2):
                    self.vec_h[int(n1)] += 1.
                    self.vec_h[int(n2)] += 1.

            self.vec_h_t[i_step, :] = self.vec_h
        self.vec_h /= float(self.no_steps)
        self.average_h = np.sum(self.vec_h) / self.no_nodes

    def __update_bjlambda(self, fitness):
        alpha = fitness[:self.no_nodes]  
        beta = fitness[self.no_nodes:]  

        for i in range(self.no_nodes):
            for j in range(i, self.no_nodes):
                ai = alpha[i]
                aj = alpha[j]
                bi = beta[0]
                bj = beta[0]

                self.B[i, j] = - 0.5 * (ai + aj + bi + bj) / float(self.no_steps)
                self.B[j, i] = self.B[i, j]  
                self.J[i, j] = - 0.25 * (bi + bj) / float(self.no_steps)
                self.J[j, i] = self.J[i, j]  

        lambda_term1 = np.cosh(self.B, dtype=np.float128) * np.exp(self.J, dtype=np.float128)
        lambda_term2 = np.sqrt(
            np.exp(2 * self.J, dtype=np.float128) * np.sinh(self.B, dtype=np.float128) ** 2 + np.exp(-2 * self.J,
                                                                                                     dtype=np.float128),
            dtype=np.float128)

        self.lambda_plus = lambda_term1 + lambda_term2
        self.lambda_minus = lambda_term1 - lambda_term2

        return self.lambda_plus, self.lambda_minus

    def fun_likelihood(self, x):
        lp, lm = self.__update_bjlambda(x)
        H = 0.
        H_0 = 0.
        t_2 = 0.

        alpha = x[:self.no_nodes]
        beta = x[self.no_nodes:]

        for j in range(self.no_nodes):
            aj = alpha[j]
            bj = beta[0]
            H += aj * self.vec_k[j] + bj * self.average_h

            for i in range(j):
                ai = alpha[i]

                H_0 += (ai + aj) / 2 + bj / 2

                t_2 += np.log(lp[i, j] ** self.no_steps + lm[i, j] ** self.no_steps, dtype=np.float64)

        acc = -(H - H_0) - t_2

        return -acc

    def solve(self, x, y):

        initial_alpha = (x)
        initial_beta = (y)

        fitness_0 = np.concatenate(
            (initial_alpha, initial_beta))

        fitness = spo.minimize(fun=self.fun_likelihood, x0=fitness_0, method='L-BFGS-B', tol=1e-6,
                               options={'maxiter': 600})

        print(fitness.message)

        self.reduced_fitness = fitness.x

        self.fitness_alpha = self.reduced_fitness[:self.no_nodes]
        self.fitness_beta = self.reduced_fitness[self.no_nodes:]  # [0]

        self.fitness_x = np.exp(- self.fitness_alpha / self.no_steps)
        self.fitness_y = np.exp(- self.fitness_beta / self.no_steps)

        x = self.fitness_x
        y = self.fitness_y

        return fitness.fun
