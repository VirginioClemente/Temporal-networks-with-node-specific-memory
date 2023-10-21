import networkx as nx
import numpy as np
from networkx.utils.mapped_queue import MappedQueue


class community_detectino_with_memory():

    def __init__(self, networks, x_all, y_all):

        self.networks = networks
        self.x_all = x_all
        self.y_all = y_all

    def average_degree_compute(self, matrici):
        average_matrix_sum = np.zeros(np.shape(matrici[0]))

        for time in range(len(matrici)):
            average_matrix_sum += matrici[time]

        return average_matrix_sum / len(matrici)

    def average_persisting_degree_compute(self, matrici):
        average_matrix_mult = np.zeros(np.shape(matrici[0]))

        for time in range(len(matrici)):
            persisting_matrix = matrici[time] * matrici[(time + 1) % len(matrici)]

            average_matrix_mult += persisting_matrix

        return average_matrix_mult / len(matrici)

    def lambda_pm(self, fitness, no_steps):

        no_nodes = int(len(fitness) / 2)

        B = np.zeros((no_nodes, no_nodes), dtype=np.float64)
        J = np.zeros((no_nodes, no_nodes), dtype=np.float64)
        lambda_minus = np.zeros((no_nodes, no_nodes), dtype=np.float64)
        lambda_plus = np.zeros((no_nodes, no_nodes), dtype=np.float64)

        alfa_beta = [-no_steps * np.log(ii) for ii in fitness]

        alpha = alfa_beta[:no_nodes]
        beta = alfa_beta[no_nodes:]

        for i in range(no_nodes):
            for j in range(i, no_nodes):
                ai = alpha[i]
                aj = alpha[j]
                bi = beta[i]
                bj = beta[j]

                B[i, j] = - 0.5 * (ai + aj + bi + bj) / float(no_steps)
                B[j, i] = B[i, j]
                J[i, j] = - 0.25 * (bi + bj) / float(no_steps)
                J[j, i] = J[i, j]

        lambda_term1 = np.cosh(B) * np.exp(J)
        lambda_term2 = np.sqrt(np.exp(2 * J, ) * np.sinh(B) ** 2 + np.exp(-2 * J))

        lambda_plus = lambda_term1 + lambda_term2
        lambda_minus = lambda_term1 - lambda_term2

        return lambda_plus, lambda_minus

    def P_ij(self, networks, x, y, lambda_plus, lambda_minus, i, j):

        T = len(networks)

        root = np.sqrt(np.exp(-1 * (np.log(y[i]) + np.log(y[j]))) + np.sinh(
            1 / (2) * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j]))) ** 2)
        sinhB = np.sinh(1 / (2) * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))

        coshB = np.cosh(1 / (2) * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))

        e_2B = np.exp(-1 * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))
        e_B = np.exp(-1 / 2 * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))

        eJ = np.exp(1 / (4) * (np.log(y[i]) + np.log(y[j])))

        sigma_l_t = (sinhB) / (2 * root) + 1 / 2

        p_ij = eJ / (2 * (lambda_minus[i, j] ** T + lambda_plus[i, j] ** T)) * (
                    lambda_plus[i, j] ** (T - 1) * (sinhB + (sinhB * coshB) / root) + lambda_minus[i, j] ** (T - 1) * (
                        sinhB - (sinhB * coshB) / root)) + 1 / 2

        return p_ij

    def Q_ij(self, networks, x, y, lambda_plus, lambda_minus, i, j):

        T = len(networks)

        root = np.sqrt(np.exp(-1 * (np.log(y[i]) + np.log(y[j]))) + np.sinh(
            1 / (2) * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j]))) ** 2)
        sinhB = np.sinh(1 / (2) * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))

        coshB = np.cosh(1 / (2) * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))

        e_2B = np.exp(-1 * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))
        e_B = np.exp(-1 / 2 * (np.log(x[i]) + np.log(x[j]) + np.log(y[i]) + np.log(y[j])))

        eJ = np.exp(1 / (4) * (np.log(y[i]) + np.log(y[j])))

        sigma_l_t = (sinhB) / (2 * root) + 1 / 2

        q_ij = sigma_l_t ** 2 + (1 - sigma_l_t) * sigma_l_t * (
                    lambda_minus[i, j] * lambda_plus[i, j] ** (T - 1) + lambda_plus[i, j] * lambda_minus[i, j] ** (
                        T - 1)) / (lambda_minus[i, j] ** T + lambda_plus[i, j] ** T)

        return q_ij

    def fitness_vec(self, dict_fitness_degree, dict_fitness_persisting):
        fitness = []

        for key_degree in dict_fitness_degree:
            fitness.append(dict_fitness_degree[key_degree])

        for key_persisting in dict_fitness_persisting:
            fitness.append(dict_fitness_persisting[key_persisting])

        # print(fitness)
        return fitness[::]  # Prima era return fitness[::-1], controlla

    def penality_com(self, dict_fitness_degree, dict_fitness_persisting, networks, community):
        penality = 0.

        T = len(networks)
        fitness = self.fitness_vec(dict_fitness_degree, dict_fitness_persisting)

        alfa_beta = [-T * np.log(ii) for ii in fitness]
        lambda_plus, lambda_minus = self.lambda_pm(alfa_beta, T)

        x = fitness[:int(len(fitness) / 2)]
        y = fitness[int(len(fitness) / 2):]

        for node_1 in community:

            for node_2 in community:
                if node_1 == node_2:
                    continue

                q_ij = self.Q_ij(networks, x, y, lambda_plus, lambda_minus, node_1, node_2)

                p_ij = self.P_ij(networks, x, y, lambda_plus, lambda_minus, node_1, node_2)

                penality += p_ij + q_ij

        return penality

    def internal_evidence(self, average_degree, average_persisting_degree, community):
        gain = 0.
        for node_1 in community:

            for node_2 in community:
                if node_1 == node_2:
                    continue

                gain += self.average_degree[node_1, node_2] + self.average_persisting_degree[node_1, node_2]

        return gain

    def modularity_persisting(self, dict_fitness_degree, dict_fitness_persisting, networks, partition):

        result = 0.

        average_degree = self.average_degree_compute(networks)
        average_persisting_degree = self.average_persisting_degree_compute(networks)

        for community in partition:
            in_evidence = self.internal_evidence(average_degree, average_persisting_degree, community)

            penality = self.penality_com(dict_fitness_degree, dict_fitness_persisting, networks, community)

            result += in_evidence - penality
        return np.float32(result)

    def greedy_modularity_communities_memory(self, networks, x, y):

        # Count nodes and edges

        G = nx.from_numpy_matrix(networks[0])

        N = len(G.nodes())

        P_null = np.zeros((N, N))
        # Calculate average degrees and average persisting degrees
        average_degree = self.average_degree_compute(networks)
        average_persisting_degree = self.average_persisting_degree_compute(networks)

        fitness = np.concatenate((x, y))
        lambda_plus, lambda_minus = self.lambda_pm(fitness, len(networks))

        # Calcolo P_null
        for i in range(N):
            for j in range(N):
                P_null[i, j] = (self.Q_ij(networks, x, y, lambda_plus, lambda_minus, i, j) + self.P_ij(networks, x, y,
                                                                                                       lambda_plus,
                                                                                                       lambda_minus, i,
                                                                                                       j))

        # Map node labels to contiguous integers
        label_for_node = {i: v for i, v in enumerate(G.nodes())}
        node_for_label = {label_for_node[i]: i for i in range(N)}

        # Initialize community and merge lists
        communities = {i: frozenset([i]) for i in range(N)}
        merges = []


        # Initial modularity
        partition = [[label_for_node[x] for x in c] for c in communities.values()]
        # calcolo la modularità definita per pesisting degree e grado
        q_cnm = 0  # modularity_persisting(dict_fitness_degree, dict_fitness_persisting, networks, partition)

        dq_dict = {
            i: {
                j: np.float32(2 * (average_degree[i, j] + average_persisting_degree[i, j]) - 2 * P_null[i, j])
                for j in [node_for_label[u] for u in range(N)]
                if j != i
            }
            for i in range(N)
        }

        dq_heap = [
            MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(N)
        ]
        H = MappedQueue([dq_heap[i].h[0] for i in range(N) if len(dq_heap[i]) > 0])

        # print([H.pop() for i in range(len(H.h))])

        # Merge communities until we can't improve modularity
        while len(H) > 1:
            # Find best merge
            # Remove from heap of row maxes
            # Ties will be broken by choosing the pair with lowest min community id
            try:
                dq, i, j = H.pop()
            except IndexError:
                break
            dq = -dq
            # Remove best merge from row i heap
            dq_heap[i].pop()
            # Push new row max onto H
            if len(dq_heap[i]) > 0:
                H.push(dq_heap[i].h[0])
            # If this element was also at the root of row j, we need to remove the
            # duplicate entry from H
            if dq_heap[j].h[0] == (-dq, j, i):
                H.remove((-dq, j, i))
                # Remove best merge from row j heap
                dq_heap[j].remove((-dq, j, i))
                # Push new row max onto H
                if len(dq_heap[j]) > 0:
                    H.push(dq_heap[j].h[0])
            else:
                # Duplicate wasn't in H, just remove from row j heap

                dq_heap[j].remove((-dq, j, i))
            # Stop when change is non-positive

            if dq <= 0:
                break

            # Perform merge
            communities[j] = frozenset(communities[i] | communities[j])
            del communities[i]
            merges.append((i, j, dq))
            # New modularity
            q_cnm += dq

            # Get list of communities connected to merged communities
            i_set = set(dq_dict[i].keys())
            j_set = set(dq_dict[j].keys())
            all_set = (i_set | j_set) - {i, j}
            both_set = i_set & j_set
            # Merge i into j and update dQ

            for k in all_set:

                # Calculate new dq value
                # if k in both_set:
                dq_jk = dq_dict[j][k] + dq_dict[i][k]
                # elif k in j_set:
                # I due passaggi sotto, sembrano essere inutili
                # dq_jk = dq_dict[j][k] -  2*P_null[i,k]  #(Q_ij(networks,x,y,lambda_plus,lambda_minus,i,k) + P_ij(networks,x,y,lambda_plus,lambda_minus,i,k))
                # else:
                # k in i_set

                # dq_jk = dq_dict[i][k] -  2*P_null[j,k] #(Q_ij(networks,x,y,lambda_plus,lambda_minus,j,k) + P_ij(networks,x,y,lambda_plus,lambda_minus,j,k))
                # Update rows j and k
                for row, col in [(j, k), (k, j)]:
                    # Save old value for finding heap index
                    if k in j_set:
                        d_old = (-dq_dict[row][col], row, col)
                    else:
                        d_old = None
                    # Update dict for j,k only (i is removed below)
                    dq_dict[row][col] = dq_jk
                    # Save old max of per-row heap
                    if len(dq_heap[row]) > 0:
                        d_oldmax = dq_heap[row].h[0]
                    else:
                        d_oldmax = None
                    # Add/update heaps
                    d = (-dq_jk, row, col)
                    if d_old is None:
                        # We're creating a new nonzero element, add to heap
                        dq_heap[row].push(d)
                    else:
                        # Update existing element in per-row heap
                        dq_heap[row].update(d_old, d)
                    # Update heap of row maxes if necessary
                    if d_oldmax is None:
                        # No entries previously in this row, push new max
                        H.push(d)
                    else:
                        # We've updated an entry in this row, has the max changed?
                        if dq_heap[row].h[0] != d_oldmax:
                            H.update(d_oldmax, dq_heap[row].h[0])

            # Remove row/col i from matrix
            i_neighbors = dq_dict[i].keys()
            for k in i_neighbors:
                # Remove from dict
                dq_old = dq_dict[k][i]
                del dq_dict[k][i]
                # Remove from heaps if we haven't already
                if k != j:
                    # Remove both row and column
                    for row, col in [(k, i), (i, k)]:
                        # Check if replaced dq is row max
                        d_old = (-dq_old, row, col)
                        if dq_heap[row].h[0] == d_old:
                            # Update per-row heap and heap of row maxes
                            dq_heap[row].remove(d_old)
                            H.remove(d_old)
                            # Update row max
                            if len(dq_heap[row]) > 0:
                                H.push(dq_heap[row].h[0])
                        else:
                            # Only update per-row heap

                            dq_heap[row].remove(d_old)

            del dq_dict[i]
            # Mark row i as deleted, but keep placeholder
            dq_heap[i] = MappedQueue()
            # Merge i into j and update P_null
            # Il passaggio seguente è inutile
            P_null[j, :] += P_null[i, :]
            P_null[i, :] = 0
            P_null[:, j] += P_null[:, i]
            P_null[:, i] = 0

        communities = [
            frozenset([label_for_node[i] for i in c]) for c in communities.values()
        ]
        return sorted(communities, key=len, reverse=True)

    def communities(self):

        return self.greedy_modularity_communities_memory(self.networks, self.x_all, self.y_all)


class community_detectino_CM():

    def __init__(self, network, dict_fitness_degree):

        self.network = network
        self.dict_fitness_degree = dict_fitness_degree

    def P_ij_no_memory(self, network, x, i, j):

        p_ij = (x[i] * x[j]) / (1 + x[i] * x[j])

        return p_ij

    def fitness_vec_no_memory(self, dict_fitness_degree):
        fitness = []

        for key_degree in dict_fitness_degree:
            fitness.append(dict_fitness_degree[key_degree])

        return fitness[::]

    def penality_com_no_memory(self, dict_fitness_degree, network, community):
        penality = 0.

        fitness = self.fitness_vec_no_memory(dict_fitness_degree)

        x = fitness

        for node_1 in community:

            for node_2 in community:
                if node_1 == node_2:
                    continue

                p_ij = self.P_ij_no_memory(network, x, node_1, node_2)

                penality += p_ij

        return penality

    def internal_evidence_no_memory(self, network, community):
        gain = 0.
        for node_1 in community:

            for node_2 in community:
                if node_1 == node_2:
                    continue

                gain += network[node_1, node_2]

        return gain

    def modularity_no_memory(self, dict_fitness_degree, network, partition):
        result = 0.

        for community in partition:
            in_evidence = self.internal_evidence_no_memory(network, community)

            penality = self.penality_com_no_memory(dict_fitness_degree, network, community)

            result += in_evidence - penality
        return np.float32(result)

    def greedy_modularity_communities_CM(self, network, dict_fitness_degree_no_memory):
        # Count nodes and edges

        G = nx.from_numpy_matrix(network)

        N = len(G.nodes())

        P_null = np.zeros((N, N))

        # definisco i vettori delle fitness
        x = self.fitness_vec_no_memory(dict_fitness_degree_no_memory)

        # Calcolo P_null
        for i in range(N):
            for j in range(N):
                P_null[i, j] = (self.P_ij_no_memory(network, x, i, j))

        # Map node labels to contiguous integers
        label_for_node = {i: v for i, v in enumerate(G.nodes())}
        node_for_label = {label_for_node[i]: i for i in range(N)}

        # Initialize community and merge lists
        communities = {i: frozenset([i]) for i in range(N)}
        merges = []

        # Initial modularity
        partition = [[label_for_node[x] for x in c] for c in communities.values()]
        # calcolo la modularità definita per pesisting degree e grado
        q_cnm = 0  # modularity_no_memory(dict_fitness_degree_no_memory,  networks, partition)

        dq_dict = {
            i: {
                j: np.float32((2 * network[i, j]) - 2 * P_null[i, j])
                for j in [node_for_label[u] for u in range(N)]
                if j != i
            }
            for i in range(N)
        }

        dq_heap = [
            MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(N)
        ]
        H = MappedQueue([dq_heap[i].h[0] for i in range(N) if len(dq_heap[i]) > 0])

        # print([H.pop() for i in range(len(H.h))])

        # Merge communities until we can't improve modularity
        while len(H) > 1:
            # Find best merge
            # Remove from heap of row maxes
            # Ties will be broken by choosing the pair with lowest min community id
            try:
                dq, i, j = H.pop()
            except IndexError:
                break
            dq = -dq
            # Remove best merge from row i heap
            dq_heap[i].pop()
            # Push new row max onto H
            if len(dq_heap[i]) > 0:
                H.push(dq_heap[i].h[0])
            # If this element was also at the root of row j, we need to remove the
            # duplicate entry from H
            if dq_heap[j].h[0] == (-dq, j, i):
                H.remove((-dq, j, i))
                # Remove best merge from row j heap
                dq_heap[j].remove((-dq, j, i))
                # Push new row max onto H
                if len(dq_heap[j]) > 0:
                    H.push(dq_heap[j].h[0])
            else:
                # Duplicate wasn't in H, just remove from row j heap

                dq_heap[j].remove((-dq, j, i))
            # Stop when change is non-positive

            if dq <= 0:
                break

            # Perform merge
            communities[j] = frozenset(communities[i] | communities[j])
            del communities[i]
            merges.append((i, j, dq))
            # New modularity
            q_cnm += dq

            # Get list of communities connected to merged communities
            i_set = set(dq_dict[i].keys())
            j_set = set(dq_dict[j].keys())
            all_set = (i_set | j_set) - {i, j}
            both_set = i_set & j_set
            # Merge i into j and update dQ

            for k in all_set:

                # Calculate new dq value
                # if k in both_set: # Nel mio caso k è sempre in both_set
                dq_jk = dq_dict[j][k] + dq_dict[i][k]

                # Update rows j and k
                for row, col in [(j, k), (k, j)]:
                    # Save old value for finding heap index
                    if k in j_set:
                        d_old = (-dq_dict[row][col], row, col)
                    else:
                        d_old = None
                    # Update dict for j,k only (i is removed below)
                    dq_dict[row][col] = dq_jk
                    # Save old max of per-row heap
                    if len(dq_heap[row]) > 0:
                        d_oldmax = dq_heap[row].h[0]
                    else:
                        d_oldmax = None
                    # Add/update heaps
                    d = (-dq_jk, row, col)
                    if d_old is None:
                        # We're creating a new nonzero element, add to heap
                        dq_heap[row].push(d)
                    else:
                        # Update existing element in per-row heap
                        dq_heap[row].update(d_old, d)
                    # Update heap of row maxes if necessary
                    if d_oldmax is None:
                        # No entries previously in this row, push new max
                        H.push(d)
                    else:
                        # We've updated an entry in this row, has the max changed?
                        if dq_heap[row].h[0] != d_oldmax:
                            H.update(d_oldmax, dq_heap[row].h[0])

            # Remove row/col i from matrix
            i_neighbors = dq_dict[i].keys()
            for k in i_neighbors:
                # Remove from dict
                dq_old = dq_dict[k][i]
                del dq_dict[k][i]
                # Remove from heaps if we haven't already
                if k != j:
                    # Remove both row and column
                    for row, col in [(k, i), (i, k)]:
                        # Check if replaced dq is row max
                        d_old = (-dq_old, row, col)
                        if dq_heap[row].h[0] == d_old:
                            # Update per-row heap and heap of row maxes
                            dq_heap[row].remove(d_old)
                            H.remove(d_old)
                            # Update row max
                            if len(dq_heap[row]) > 0:
                                H.push(dq_heap[row].h[0])
                        else:
                            # Only update per-row heap

                            dq_heap[row].remove(d_old)

            del dq_dict[i]
            # Mark row i as deleted, but keep placeholder
            dq_heap[i] = MappedQueue()
            # Merge i into j and update P_null
            # Il passaggio seguente sembra essere inutile per l'algoritmo (per come l'ho scritto io)
            P_null[j, :] += P_null[i, :]
            P_null[i, :] = 0
            P_null[:, j] += P_null[:, i]
            P_null[:, i] = 0

        communities = [
            frozenset([label_for_node[i] for i in c]) for c in communities.values()
        ]
        return sorted(communities, key=len, reverse=True)

    def communities(self):

        return self.greedy_modularity_communities_CM(self.network, self.dict_fitness_degree)


import networkx as nx
import numpy as np
from networkx.utils.mapped_queue import MappedQueue


class community_detectino_no_memory():

    def __init__(self, networks, dict_fitness_degree):

        self.networks = networks
        self.dict_fitness_degree = dict_fitness_degree

    def average_degree_compute(self, matrici):
        average_matrix_sum = np.zeros(np.shape(matrici[0]))

        for time in range(len(matrici)):
            average_matrix_sum += matrici[time]

        return average_matrix_sum / len(matrici)

    def P_ij_no_memory(self, networks, x, i, j):
        T = len(networks)
        p_ij = (x[i] * x[j]) / (1 + x[i] * x[j])

        return p_ij

    def fitness_vec_no_memory(self, dict_fitness_degree):
        fitness = []

        for key_degree in dict_fitness_degree:
            fitness.append(dict_fitness_degree[key_degree])

        return fitness[::]

    def penality_com_no_memory(self, dict_fitness_degree, networks, community):
        penality = 0.

        T = len(networks)
        fitness = self.fitness_vec_no_memory(dict_fitness_degree)

        x = fitness

        for node_1 in community:

            for node_2 in community:
                if node_1 == node_2:
                    continue

                p_ij = self.P_ij_no_memory(networks, x, node_1, node_2)

                penality += p_ij

        return penality

    def internal_evidence_no_memory(self, average_degree, community):
        gain = 0.
        for node_1 in community:

            for node_2 in community:
                if node_1 == node_2:
                    continue

                gain += average_degree[node_1, node_2]

        return gain

    def modularity_no_memory(self, dict_fitness_degree, networks, partition):
        result = 0.

        average_degree = self.average_degree_compute(networks)

        for community in partition:
            in_evidence = self.internal_evidence_no_memory(average_degree, community)

            penality = self.penality_com_no_memory(dict_fitness_degree, networks, community)

            result += in_evidence - penality
        return np.float32(result)

    def greedy_modularity_communities_no_memory(self, networks, dict_fitness_degree_no_memory):
        # Count nodes and edges

        G = nx.from_numpy_matrix(networks[0])

        N = len(G.nodes())

        P_null = np.zeros((N, N))
        # Calculate average degrees and average persisting degrees
        average_degree = self.average_degree_compute(networks)

        # definisco i vettori delle fitness
        x = self.fitness_vec_no_memory(dict_fitness_degree_no_memory)

        # Calcolo P_null
        for i in range(N):
            for j in range(N):
                P_null[i, j] = (self.P_ij_no_memory(networks, x, i, j))

        # Map node labels to contiguous integers
        label_for_node = {i: v for i, v in enumerate(G.nodes())}
        node_for_label = {label_for_node[i]: i for i in range(N)}

        # Initialize community and merge lists
        communities = {i: frozenset([i]) for i in range(N)}
        merges = []

        # Initial modularity
        partition = [[label_for_node[x] for x in c] for c in communities.values()]
        # calcolo la modularità definita per pesisting degree e grado
        q_cnm = 0  # modularity_no_memory(dict_fitness_degree_no_memory,  networks, partition)

        dq_dict = {
            i: {
                j: np.float32((2 * average_degree[i, j]) - 2 * P_null[i, j])
                for j in [node_for_label[u] for u in range(N)]
                if j != i
            }
            for i in range(N)
        }

        dq_heap = [
            MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(N)
        ]
        H = MappedQueue([dq_heap[i].h[0] for i in range(N) if len(dq_heap[i]) > 0])

        # print([H.pop() for i in range(len(H.h))])

        # Merge communities until we can't improve modularity
        while len(H) > 1:
            # Find best merge
            # Remove from heap of row maxes
            # Ties will be broken by choosing the pair with lowest min community id
            try:
                dq, i, j = H.pop()
            except IndexError:
                break
            dq = -dq
            # Remove best merge from row i heap
            dq_heap[i].pop()
            # Push new row max onto H
            if len(dq_heap[i]) > 0:
                H.push(dq_heap[i].h[0])
            # If this element was also at the root of row j, we need to remove the
            # duplicate entry from H
            if dq_heap[j].h[0] == (-dq, j, i):
                H.remove((-dq, j, i))
                # Remove best merge from row j heap
                dq_heap[j].remove((-dq, j, i))
                # Push new row max onto H
                if len(dq_heap[j]) > 0:
                    H.push(dq_heap[j].h[0])
            else:
                # Duplicate wasn't in H, just remove from row j heap

                dq_heap[j].remove((-dq, j, i))
            # Stop when change is non-positive

            if dq <= 0:
                break

            # Perform merge
            communities[j] = frozenset(communities[i] | communities[j])
            del communities[i]
            merges.append((i, j, dq))
            # New modularity
            q_cnm += dq

            # Get list of communities connected to merged communities
            i_set = set(dq_dict[i].keys())
            j_set = set(dq_dict[j].keys())
            all_set = (i_set | j_set) - {i, j}
            both_set = i_set & j_set
            # Merge i into j and update dQ

            for k in all_set:

                # Calculate new dq value
                # if k in both_set: # Nel mio caso k è sempre in both_set
                dq_jk = dq_dict[j][k] + dq_dict[i][k]

                # Update rows j and k
                for row, col in [(j, k), (k, j)]:
                    # Save old value for finding heap index
                    if k in j_set:
                        d_old = (-dq_dict[row][col], row, col)
                    else:
                        d_old = None
                    # Update dict for j,k only (i is removed below)
                    dq_dict[row][col] = dq_jk
                    # Save old max of per-row heap
                    if len(dq_heap[row]) > 0:
                        d_oldmax = dq_heap[row].h[0]
                    else:
                        d_oldmax = None
                    # Add/update heaps
                    d = (-dq_jk, row, col)
                    if d_old is None:
                        # We're creating a new nonzero element, add to heap
                        dq_heap[row].push(d)
                    else:
                        # Update existing element in per-row heap
                        dq_heap[row].update(d_old, d)
                    # Update heap of row maxes if necessary
                    if d_oldmax is None:
                        # No entries previously in this row, push new max
                        H.push(d)
                    else:
                        # We've updated an entry in this row, has the max changed?
                        if dq_heap[row].h[0] != d_oldmax:
                            H.update(d_oldmax, dq_heap[row].h[0])

            # Remove row/col i from matrix
            i_neighbors = dq_dict[i].keys()
            for k in i_neighbors:
                # Remove from dict
                dq_old = dq_dict[k][i]
                del dq_dict[k][i]
                # Remove from heaps if we haven't already
                if k != j:
                    # Remove both row and column
                    for row, col in [(k, i), (i, k)]:
                        # Check if replaced dq is row max
                        d_old = (-dq_old, row, col)
                        if dq_heap[row].h[0] == d_old:
                            # Update per-row heap and heap of row maxes
                            dq_heap[row].remove(d_old)
                            H.remove(d_old)
                            # Update row max
                            if len(dq_heap[row]) > 0:
                                H.push(dq_heap[row].h[0])
                        else:
                            # Only update per-row heap

                            dq_heap[row].remove(d_old)

            del dq_dict[i]
            # Mark row i as deleted, but keep placeholder
            dq_heap[i] = MappedQueue()
            # Merge i into j and update P_null
            # Il passaggio seguente sembra essere inutile per l'algoritmo (per come l'ho scritto io)
            P_null[j, :] += P_null[i, :]
            P_null[i, :] = 0
            P_null[:, j] += P_null[:, i]
            P_null[:, i] = 0

        communities = [
            frozenset([label_for_node[i] for i in c]) for c in communities.values()
        ]
        return sorted(communities, key=len, reverse=True)

    def communities(self):

        return self.greedy_modularity_communities_no_memory(self.networks, self.dict_fitness_degree)
