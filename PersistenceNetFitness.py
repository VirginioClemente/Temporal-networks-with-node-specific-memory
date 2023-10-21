import numpy as np
import networkx as nx
import re

import scipy.optimize as spo
from scipy.optimize import least_squares
import scipy




# Vincolo sui nodi
class PersistenceNetFitness:

    def __init__(self, tgraph, symmetric=True):
        self.verbose = False
        self.temp_graph = tgraph
        self.no_steps = len(self.temp_graph.data)
        self.no_nodes = len(self.temp_graph.data[0][2].nodes())

        self.fitness = np.ones(2 * self.no_nodes, dtype=np.float64)

        self.symmetric = symmetric

        self.B = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.J = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.lambda_minus = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.lambda_plus = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)

        self.__update_bjlambda(self.fitness)
        self.__fill_vecs()

    def __fill_vecs(self):

        self.vec_k = np.zeros(self.no_nodes, dtype=np.float64)
        self.vec_h = np.zeros(self.no_nodes, dtype=np.float64)

        self.vec_h_t = np.zeros((self.no_steps, self.no_nodes))

        for i_step in range(self.no_steps):
            block_id, current_time, g = self.temp_graph.data[i_step]
            for [n1, n2] in g.edges():
                if n1 == n2: continue

                self.vec_k[int(n1)] += 1.
                self.vec_k[int(n2)] += 1.

        self.vec_k /= float(self.no_steps)

        for i_step in range(self.no_steps):  # Tolto - 1
            block_id, current_time, g0 = self.temp_graph.data[i_step]
            block_id, current_time, g1 = self.temp_graph.data[(i_step + 1) % self.no_steps]
            for [n1, n2] in g0.edges():
                if n1 == n2: continue
                if g1.has_edge(n1, n2):
                    self.vec_h[int(n1)] += 1.
                    self.vec_h[int(n2)] += 1.

            self.vec_h_t[i_step, :] = self.vec_h
        # Devi rimuovere la riga sottostante quando calcoli vec_h medio per fare il grafico.
        list_h = []
        h = []
        for i in range(len(self.vec_h_t)):
            if i > 0:
                h = self.vec_h_t[i, :] - self.vec_h_t[i - 1, :]
                list_h.append(h)

            if i == 0:
                h = self.vec_h_t[i, :]
                list_h.append(h)

        # self.vec_h_t = list_h

        self.vec_h /= float(self.no_steps)

    def __update_bjlambda(self, fitness):
        alpha = fitness[:self.no_nodes]
        beta = fitness[self.no_nodes:]

        for i in range(self.no_nodes):
            for j in range(i, self.no_nodes):
                ai = alpha[i]
                aj = alpha[j]
                bi = beta[i]
                bj = beta[j]

                self.B[i, j] = - 0.5 * (ai + aj + bi + bj) / float(self.no_steps)
                self.B[j, i] = self.B[i, j]
                self.J[i, j] = - 0.25 * (bi + bj) / float(self.no_steps)
                self.J[j, i] = self.J[i, j]

        lambda_term1 = np.cosh(self.B, dtype=np.float64) * np.exp(self.J, dtype=np.float64)
        lambda_term2 = np.sqrt(
            np.exp(2 * self.J, dtype=np.float64) * np.sinh(self.B, dtype=np.float64) ** 2 + np.exp(-2 * self.J,
                                                                                                   dtype=np.float64),
            dtype=np.float128)

        self.lambda_plus = lambda_term1 + lambda_term2
        self.lambda_minus = lambda_term1 - lambda_term2

        lambda_p = self.lambda_plus
        lambda_m = self.lambda_minus

        return lambda_p, lambda_m

    def fun_likelihood(self, x):
        lp, lm = self.__update_bjlambda(x)
        acc = 0.
        H = 0.
        H_0 = 0.
        t_2 = 0.

        alpha = x[:self.no_nodes]
        beta = x[self.no_nodes:]

        for j in range(self.no_nodes):
            aj = alpha[j]
            bj = beta[j]
            H += aj * self.vec_k[j] + bj * self.vec_h[j]

            for i in range(j):
                ai = alpha[i]
                bi = beta[i]

                H_0 += (ai + aj) / 2 + (bi + bj) / 4

                t_2 += np.log(np.power(lp[i, j], self.no_steps, dtype=np.float64) + np.power(lm[i, j], self.no_steps,
                                                                                             dtype=np.float64),
                              dtype=np.float64)  # lp[i,j]**self.no_steps  + lm[i,j]**self.no_steps

        acc = -(H - H_0) - t_2

        return -acc

    def derivates(self, i, j):

        T = self.no_steps
        eJ = np.exp(self.J[i, j], dtype=np.float64)
        sinhB = np.sinh(self.B[i, j], dtype=np.float64)
        coshB = np.cosh(self.B[i, j], dtype=np.float64)
        e_4J = np.exp(-4 * self.J[i, j], dtype=np.float64)
        root = np.sqrt(sinhB ** 2 + np.exp(-4 * self.J[i, j], dtype=np.float128), dtype=np.float64)

        D_Lambda_plus_D_Alfa = -1 / (2 * T) * eJ * sinhB + eJ / (2 * root) * (2 * sinhB * coshB) / (-2 * T)
        D_Lambda_minus_D_Alfa = -1 / (2 * T) * eJ * sinhB - eJ / (2 * root) * (2 * sinhB * coshB) / (-2 * T)
        D_Lambda_plus_D_Beta = (-1 / (2 * T)) * eJ * sinhB - (1 / (4 * T)) * eJ * coshB + (
                    (-1 / (4 * T)) * eJ * root + eJ / (2 * root) * (-1 / T * sinhB * coshB + 1 / T * e_4J))
        D_Lambda_minus_D_Beta = (-1 / (2 * T)) * eJ * sinhB - (1 / (4 * T)) * eJ * coshB - (
                    (-1 / (4 * T)) * eJ * root + eJ / (2 * root) * (-1 / T * sinhB * coshB + 1 / T * e_4J))

        return D_Lambda_plus_D_Alfa, D_Lambda_minus_D_Alfa, D_Lambda_plus_D_Beta, D_Lambda_minus_D_Beta

    def loglikelihood_prime_persisting(self, theta):

        k = self.vec_k
        h = self.vec_h
        n = self.no_nodes

        T = self.no_steps

        f = np.zeros(2 * n, dtype=np.float64)

        for i in range(n):
            t_sum = 0
            t_beta = 0
            t_alfa = 0
            t_sum_alfa = 0
            t_sum_beta = 0
            for j in range(n):
                if j != i:
                    D_Lambda_plus_D_Alfa, D_Lambda_minus_D_Alfa, D_Lambda_plus_D_Beta, D_Lambda_minus_D_Beta = self.derivates(
                        i, j)

                    t_sum_alfa += (T * (
                        np.power(self.lambda_plus[i, j], T - 1, dtype=np.float64)) * D_Lambda_plus_D_Alfa + T * (
                                       np.power(self.lambda_minus[i, j], T - 1,
                                                dtype=np.float64)) * D_Lambda_minus_D_Alfa) / (
                                          np.power(self.lambda_plus[i, j], T, dtype=np.float64) + np.power(
                                      self.lambda_minus[i, j], T, dtype=np.float64))

                    t_sum_beta += (T * (
                        np.power(self.lambda_plus[i, j], T - 1, dtype=np.float64)) * D_Lambda_plus_D_Beta + T * (
                                       np.power(self.lambda_minus[i, j], T - 1,
                                                dtype=np.float64)) * D_Lambda_minus_D_Beta) / (
                                          np.power(self.lambda_plus[i, j], T, dtype=np.float64) + np.power(
                                      self.lambda_minus[i, j], T, dtype=np.float64))

            f[i] = -(-t_sum_alfa - k[i] + (n - 1) / 2)

            f[i + n] = -(-t_sum_beta - h[i] + (n - 1) / 4)

        return f

    def solve(self, x, y):

        initial_alpha = (x)
        initial_beta = (y)

        fitness_0 = np.concatenate((initial_alpha, initial_beta))

        bnds = [(-1e+3, 1e+3) for i in range(self.no_nodes)] + [(-1e+3, 1e+3) for i in range(self.no_nodes)]
        # bnds = [(None, None) for i in range(self.no_nodes)] + [(None, None) for i in range(self.no_nodes)]

        fitness = spo.minimize(fun=self.fun_likelihood, bounds=bnds, x0=fitness_0,
                               jac=self.loglikelihood_prime_persisting, method='L-BFGS-B', tol=1e-5,
                               options={'maxiter': 600, 'eps': 1e-9, 'gtol': 1e-07})

        print(fitness.message)
        self.reduced_fitness = fitness.x

        self.fitness_alpha = self.reduced_fitness[:self.no_nodes]
        self.fitness_beta = self.reduced_fitness[self.no_nodes:]

        self.fitness_x = np.exp(- self.fitness_alpha / self.no_steps, dtype=np.float64)
        self.fitness_y = np.exp(- self.fitness_beta / self.no_steps, dtype=np.float64)

        x = self.fitness_x
        y = self.fitness_y

        return fitness.fun