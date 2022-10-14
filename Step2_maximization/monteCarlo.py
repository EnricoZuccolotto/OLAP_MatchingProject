import copy
import math

import numpy as np



class monteCarlo():
    def __init__(self):
        self.z=np.zeros(5)

        return
    # product->fist product the user will visit
    # weight associated to user
    # n number of repetitions
    def monteCarloRuns(self, n, product,weights):
        self.z = np.zeros(5)
        # using 1000 runs compared to using 50 runs i make an error of avg 6 percent
        # using 1000 runs compared to using 100 runs i make an error of avg 3 percent (RSS squared error)(experimented on 300 visits)

        for i in range(n):
            self.simulateEpisode(product,copy.deepcopy(weights))

        return self.z/n

    def simulateEpisode(self, product, usableWeights):
        prob_matrix = usableWeights.copy()
        active_nodes = np.zeros(5)
        active_nodes[product] = 1
        self.z[product] += 1
        newly_active_nodes = active_nodes
        t = 0
        while t < 5 and np.sum(newly_active_nodes) > 0:
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
            self.z[newly_active_nodes == 1] += 1
            active_nodes = np.array(active_nodes + newly_active_nodes)

            t += 1

        return
