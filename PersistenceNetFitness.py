import numpy as np
import networkx as nx
import re

import scipy.optimize as spo
from scipy.optimize import least_squares
import scipy




# Constraints at the node level
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
        # This method is used to calculate the average degree and the average persistent degree for each node.

        # Initialize arrays to store degree (vec_k) and persistent degree (vec_h) for each node.
        self.vec_k = np.zeros(self.no_nodes, dtype=np.float64)
        self.vec_h = np.zeros(self.no_nodes, dtype=np.float64)
   
        # Initialize a 2D array to store the persistent degree at each time step.
        self.vec_h_t = np.zeros((self.no_steps, self.no_nodes))

        for i_step in range(self.no_steps):
            block_id, current_time, g = self.temp_graph.data[i_step]
            for [n1, n2] in g.edges():
                if n1 == n2: continue # Skip self-loops

                self.vec_k[int(n1)] += 1.
                self.vec_k[int(n2)] += 1.

        # Average the degree over the number of steps
        self.vec_k /= float(self.no_steps)

        # Calculate the persistent degree for each node.
        for i_step in range(self.no_steps):  
            block_id, current_time, g0 = self.temp_graph.data[i_step]
            block_id, current_time, g1 = self.temp_graph.data[(i_step + 1) % self.no_steps]
            for [n1, n2] in g0.edges():
                if n1 == n2: continue # Skip self-loops
                if g1.has_edge(n1, n2):
                    # Increment persistent degree if the edge persists in the next time step.
                    self.vec_h[int(n1)] += 1.
                    self.vec_h[int(n2)] += 1.

            # Store the persistent degree at this time step.
            self.vec_h_t[i_step, :] = self.vec_h
            
        # Average the persistent degree over the number of steps.
        self.vec_h /= float(self.no_steps)

    def __update_bjlambda(self, fitness):
        # This function constructs functional quantities for the calculation of likelihood.

        # Splitting the 'fitness' array into 'alpha' and 'beta' components.
        alpha = fitness[:self.no_nodes]
        beta = fitness[self.no_nodes:]
        
        # Iterating over all pairs of nodes to calculate the B and J matrices.
        for i in range(self.no_nodes):
            for j in range(i, self.no_nodes):
                ai = alpha[i]
                aj = alpha[j]
                bi = beta[i]
                bj = beta[j]

                # Calculating the B matrix elements.
                self.B[i, j] = - 0.5 * (ai + aj + bi + bj) / float(self.no_steps)
                self.B[j, i] = self.B[i, j]

                # Calculating the J matrix elements.
                self.J[i, j] = - 0.25 * (bi + bj) / float(self.no_steps)
                self.J[j, i] = self.J[i, j]

        lambda_term1 = np.cosh(self.B, dtype=np.float64) * np.exp(self.J, dtype=np.float64)
        lambda_term2 = np.sqrt(
            np.exp(2 * self.J, dtype=np.float64) * np.sinh(self.B, dtype=np.float64) ** 2 + np.exp(-2 * self.J,
                                                                                                   dtype=np.float64),
            dtype=np.float128)

        # Returning the calculated lambda_plus and lambda_minus for further use.
        self.lambda_plus = lambda_term1 + lambda_term2
        self.lambda_minus = lambda_term1 - lambda_term2

        return self.lambda_plus, self.lambda_minus

    def fun_likelihood(self, x):
        # Update the lambda_plus and lambda_minus matrices based on the current parameter set 'x'.
        lp, lm = self.__update_bjlambda(x)
        # Initializing accumulators for different parts of the likelihood calculation.
        acc = 0.  # Accumulator for the final log-likelihood.
        H = 0.    # Sum related to the nodes' degrees and persistent degrees.
        H_0 = 0.  # Sum related to the alpha and beta parameters.
        t_2 = 0.  # Sum related to the log of the lambda terms.
    
        # Splitting the parameter array 'x' into 'alpha' and 'beta' components.
        alpha = x[:self.no_nodes]
        beta = x[self.no_nodes:]
    
        # Calculating the H term using node degrees (vec_k) and persistent degrees (vec_h).
        for j in range(self.no_nodes):
            aj = alpha[j]
            bj = beta[j]
            H += aj * self.vec_k[j] + bj * self.vec_h[j]
    
            # Calculating the H_0 and t_2 terms.
            for i in range(j):
                ai = alpha[i]
                bi = beta[i]
    
                # Increment H_0 based on alpha and beta parameters.
                H_0 += (ai + aj) / 2 + (bi + bj) / 4
    
                # Increment t_2 based on the log of the lambda terms.
                t_2 += np.log(np.power(lp[i, j], self.no_steps, dtype=np.float64) +
                              np.power(lm[i, j], self.no_steps, dtype=np.float64), dtype=np.float64)
    
        # Final calculation of the log-likelihood.
        acc = -(H - H_0) - t_2
    
        # Returning the negative of the log-likelihood for optimization purposes.
        return -acc

    def derivates(self, i, j):
        #This function calculates the derivatives of lambda_plus and lambda_minus with respect to alpha and beta parameters for a given pair of nodes (i, j).
        
        # Initialization of various terms used in the derivative calculations.
        T = self.no_steps
        eJ = np.exp(self.J[i, j], dtype=np.float64)
        sinhB = np.sinh(self.B[i, j], dtype=np.float64)
        coshB = np.cosh(self.B[i, j], dtype=np.float64)
        e_4J = np.exp(-4 * self.J[i, j], dtype=np.float64)
        root = np.sqrt(sinhB ** 2 + np.exp(-4 * self.J[i, j], dtype=np.float128), dtype=np.float64)

        # Calculation of the derivatives of lambda_plus and lambda_minus w.r.t alpha.
        D_Lambda_plus_D_Alfa = -1 / (2 * T) * eJ * sinhB + eJ / (2 * root) * (2 * sinhB * coshB) / (-2 * T)
        D_Lambda_minus_D_Alfa = -1 / (2 * T) * eJ * sinhB - eJ / (2 * root) * (2 * sinhB * coshB) / (-2 * T)

        # Calculation of the derivatives of lambda_plus and lambda_minus w.r.t beta.
        D_Lambda_plus_D_Beta = (-1 / (2 * T)) * eJ * sinhB - (1 / (4 * T)) * eJ * coshB + (
                    (-1 / (4 * T)) * eJ * root + eJ / (2 * root) * (-1 / T * sinhB * coshB + 1 / T * e_4J))
        D_Lambda_minus_D_Beta = (-1 / (2 * T)) * eJ * sinhB - (1 / (4 * T)) * eJ * coshB - (
                    (-1 / (4 * T)) * eJ * root + eJ / (2 * root) * (-1 / T * sinhB * coshB + 1 / T * e_4J))

        return D_Lambda_plus_D_Alfa, D_Lambda_minus_D_Alfa, D_Lambda_plus_D_Beta, D_Lambda_minus_D_Beta

    def loglikelihood_prime_persisting(self, theta):
        #This function calculates the gradient of the log-likelihood function with respect to the parameters theta (composed of alpha and beta).
        
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
        # Initialization of initial alpha and beta values from arguments x and y.
        initial_alpha = (x)
        initial_beta = (y)

        # Combining alpha and beta into a single array for the optimization algorithm.
        fitness_0 = np.concatenate((initial_alpha, initial_beta))

        # Setting bounds for the optimization. Each parameter is constrained to be between -1000 and 1000.
        bnds = [(-1e+3, 1e+3) for i in range(self.no_nodes)] + [(-1e+3, 1e+3) for i in range(self.no_nodes)]
        # bnds = [(None, None) for i in range(self.no_nodes)] + [(None, None) for i in range(self.no_nodes)]

        # Performing the optimization using SciPy's minimize function.
        fitness = spo.minimize(fun=self.fun_likelihood, bounds=bnds, x0=fitness_0,
                               jac=self.loglikelihood_prime_persisting, method='L-BFGS-B', tol=1e-5,
                               options={'maxiter': 600, 'eps': 1e-9, 'gtol': 1e-07})

        print(fitness.message)
        # Storing the optimized parameters.
        self.reduced_fitness = fitness.x

        self.fitness_alpha = self.reduced_fitness[:self.no_nodes]
        self.fitness_beta = self.reduced_fitness[self.no_nodes:]

        # Calculating transformed fitness parameters
        self.fitness_x = np.exp(- self.fitness_alpha / self.no_steps, dtype=np.float64)
        self.fitness_y = np.exp(- self.fitness_beta / self.no_steps, dtype=np.float64)

        x = self.fitness_x
        y = self.fitness_y

        # Returning the function value at the optimized parameters.
        return fitness.fun
