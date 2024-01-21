import numpy as np
import networkx as nx
import re

import scipy.optimize as spo
from scipy.optimize import least_squares
import scipy

# Constraints at dyadic level
class PersistenceNetFitnesslinks:
    MIN_FITNESS = 1e-5  # -5

    def __init__(self, tgraph, symmetric=True):
        #self.verbose = False
        
        # Storing the instance of TemporalNetworkLoaderSynt, which contains the temporal graph data.
        self.temp_graph = tgraph
        
        # Determining the number of time steps and nodes in the temporal graph.
        self.no_steps = len(self.temp_graph.data)
        self.no_nodes = len(self.temp_graph.data[0][2].nodes())

        # Initializing a 'fitness' array,for calculation purposes.
        self.fitness = np.ones(self.no_nodes + int(self.no_nodes * (self.no_nodes - 1) / 2), dtype=np.float64)

        self.symmetric = symmetric

        # Initializing matrices B, J, lambda_minus, and lambda_plus, likely for use in calculations.
        self.B = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.J = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.lambda_minus = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.lambda_plus = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)
        self.ave_persisting_matrix = np.zeros((self.no_nodes, self.no_nodes), dtype=np.float64)

        # Calculating and storing network statistics relevant to the model.
        self.__fill_vecs()

        # Initializing B, J, lambda_minus, and lambda_plus using the fitness array.
        self.update_bjlambda(self.fitness)


    def average_persisting_degree_compute(self, matrici):
        average_matrix_mult = np.zeros(np.shape(matrici[0]))

        for time in range(len(matrici)):
            persisting_matrix = matrici[time] * matrici[(time + 1) % len(matrici)]

            average_matrix_mult += persisting_matrix

        return average_matrix_mult / len(matrici)


    def __fill_vecs(self):
        self.vec_k = np.zeros(self.no_nodes)
        self.vec_h = np.zeros(self.no_nodes)
        self.average_h = np.zeros(1)
        self.ave_persisting_aij = np.zeros(int(self.no_nodes * (self.no_nodes - 1) / 2))

        matrici = []

        for time in range(self.no_steps):
            G = self.temp_graph.data[time][2]

            matrice = nx.to_numpy_array(G)

            matrici.append(matrix_to_1(matrice))

        self.ave_persisting_matrix = self.average_persisting_degree_compute(matrici)

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
        ij = 0

        for i_step in range(self.no_nodes):
            for j_step in range(i_step):
                self.ave_persisting_aij[ij] = self.ave_persisting_matrix[i_step, j_step]
                ij += 1

    def update_bjlambda(self, fitness):
        alpha = fitness[:self.no_nodes]
        beta = fitness[self.no_nodes:]


        ij = 0
        for i in range(self.no_nodes):
            for j in range(i):
                ai = alpha[i]
                aj = alpha[j]
                if self.ave_persisting_aij[ij ]= =0:
                    beta[ij] = 0

                bij = beta[ij]

                self.B[i, j] = - 0.5 * (ai + aj + bij) / float(self.no_steps)
                self.B[j, i] = self.B[i, j]
                self.J[i, j] = - 0.25 * (bij) / float(self.no_steps)
                self.J[j, i] = self.J[i, j]

                ij += 1

        lambda_term1 = np.cosh(self.B, dtype=np.float64) * np.exp(self.J, dtype=np.float128)
        lambda_term2 = np.sqrt(
            np.exp(2 * self.J, dtype=np.float64) * np.sinh(self.B, dtype=np.float128) ** 2 + np.exp(-2 * self.J,
                                                                                                    dtype=np.float128),
            dtype=np.float128)

        self.lambda_plus = lambda_term1 + lambda_term2
        self.lambda_minus = lambda_term1 - lambda_term2

        return self.lambda_plus, self.lambda_minus


    def derivates(self ,i ,j):

        T = self.no_steps
        eJ = np.exp(self.J[i, j], dtype=np.float64)
        sinhB = np.sinh(self.B[i, j], dtype=np.float64)
        coshB = np.cosh(self.B[i, j], dtype=np.float64)
        e_4J = np.exp(-4 * self.J[i, j], dtype=np.float64)
        root = np.sqrt(sinhB ** 2 + np.exp(-4 * self.J[i, j], dtype=np.float128), dtype=np.float64)

        D_Lambda_plus_D_Alfa = -1 / (2 * T) * eJ * sinhB + eJ / (2 * root) * (2 * sinhB * coshB) / (-2 * T)
        D_Lambda_minus_D_Alfa = -1 / (2 * T) * eJ * sinhB - eJ / (2 * root) * (2 * sinhB * coshB) / (-2 * T)



        D_Lambda_plus_D_Beta = (-1 / (2 * T)) * eJ * sinhB - (1 / (4 * T)) * eJ * coshB + \
                    ((-1 / (4 * T)) * eJ * root + eJ / (2 * root) * (-1 / T * sinhB * coshB + 1 / T * e_4J))
        D_Lambda_minus_D_Beta = (-1 / (2 * T)) * eJ * sinhB - (1 / (4 * T)) * eJ * coshB - (
                    (-1 / (4 * T)) * eJ * root + eJ / (2 * root) * (-1 / T * sinhB * coshB + 1 / T * e_4J))

        return D_Lambda_plus_D_Alfa, D_Lambda_minus_D_Alfa, D_Lambda_plus_D_Beta, D_Lambda_minus_D_Beta

    def loglikelihood_prime_persisting(self, theta):

        k = self.vec_k
        h = self.ave_persisting_aij

        n = self.no_nodes

        T = self.no_steps

        f = np.zeros(len(k) + len(h), dtype=np.float64)
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

            f[i] = -(-t_sum_alfa - k[i] + (n - 1) / 2)

        ij = 0
        for j in range(n):
            # t_sum_beta = 0
            for i in range(j):
                D_Lambda_plus_D_Alfa, D_Lambda_minus_D_Alfa, D_Lambda_plus_D_Beta, D_Lambda_minus_D_Beta = self.derivates(
                    i, j)

                t_sum_beta = (T * (
                    np.power(self.lambda_plus[i, j], T - 1, dtype=np.float64)) * D_Lambda_plus_D_Beta + T * (
                                  np.power(self.lambda_minus[i, j], T - 1,
                                           dtype=np.float64)) * D_Lambda_minus_D_Beta) / (
                                     np.power(self.lambda_plus[i, j], T, dtype=np.float64) + np.power(
                                 self.lambda_minus[i, j], T, dtype=np.float64))

                f[ij + n] = -(-t_sum_beta - h[ij] + 1 / 4)
                ij += 1

        return f

    def fun_likelihood(self, x):
        lp, lm = self.update_bjlambda(x)
        acc = 0.
        H_1 = 0.
        H_2 = 0.
        H_0 = 0.
        t_2 = 0.

        alpha = x[:self.no_nodes]
        beta = x[self.no_nodes:]
        ij = 0
        for j in range(self.no_nodes):
            aj = alpha[j]

            H_1 += aj * self.vec_k[j]

            for i in range(j):
                ai = alpha[i]
                if self.ave_persisting_aij[ij] == 0:
                    beta[ij] = 0

                bij = beta[ij]

                H_2 += bij * self.ave_persisting_aij[ij]

                H_0 += (ai + aj) / 2 + bij / 4

                t_2 += np.log(lp[i, j] ** self.no_steps + lm[i, j] ** self.no_steps, dtype=np.float64)

                ij += 1

        acc = -(H_1 + H_2 - H_0) - t_2

        return -acc

    def solve(self, alfa_in, beta_in):

        initial_alpha = alfa_in  

        initial_beta = [0 if self.ave_persisting_aij[i] == 0 else beta_in[i] for i in
                        range(int(self.no_nodes * (self.no_nodes - 1) / 2))]  # beta_in

        fitness_0 = np.concatenate(
            (initial_alpha, initial_beta)) 

        bnds = [(None, None) for i in range(len(alfa_in))] + [
            (0, 0) if self.ave_persisting_aij[i] == 0 else (None, None) for i in
            range(int(self.no_nodes * (self.no_nodes - 1) / 2))]

        fitness = spo.minimize(fun=self.fun_likelihood, bounds=bnds, x0=fitness_0, method='L-BFGS-B', tol=1e-7,
                               jac=self.loglikelihood_prime_persisting,
                               options={'maxiter': 1000, 'eps': 1e-7, 'gtol': 1e-7})

        print(fitness.message)

        self.reduced_fitness = fitness.x

        self.fitness_alpha = self.reduced_fitness[:self.no_nodes]
        self.fitness_beta = self.reduced_fitness[self.no_nodes:]

        self.fitness_x = np.exp(- self.fitness_alpha / self.no_steps)
        self.fitness_y = np.exp(- self.fitness_beta / self.no_steps)

        x = self.fitness_x
        y = self.fitness_y

        return fitness.fun
