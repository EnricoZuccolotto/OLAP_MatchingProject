import copy
import math

import numpy as np


class matchingBestDiscountCode():
    def __init__(self,prices,costs,n):
        self.costs=costs
        self.prices=prices
        self.z=np.zeros(5)
        self.numberOfRuns=n
        return
    # product->fist product the user will visit
    # weight associated to user
    # n number of repetitions
    def monteCarloRuns(self, n, product, weights, productsSeen):
        self.z = np.zeros(5)
        if productsSeen[product]==0:
            return 0

        weights= (weights.T * productsSeen).T
        # using 1000 runs compared to using 50 runs i make an error of avg 6 percent
        # using 1000 runs compared to using 100 runs i make an error of avg 3 percent (RSS squared error)(experimented on 300 visits)

        for i in range(n):
            self.simulateEpisode(product,copy.deepcopy(weights))



        rew= np.sum(self.z / n * (self.prices-self.costs))
        return rew

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




    def matcher(self,M,M0,user,weights,returnerWeights):

        w=np.zeros(6)
        for p in range(6):
            if p>4:
                # no discount case
                w[p]=self.noDiscountCase(weights,M0,user)
            else:
                # in case we have a discount
                w[p]=self.discountCase(returnerWeights[p],M,user,p)
        w=np.nan_to_num(w)

        return np.random.choice(np.where(w == max(w))[0])


    def noDiscountCase(self,weights,M0,user):
        reward=0
        for p in range(5):
            if user.probabilityFutureBehaviour[p]==0:
                r=0
            else:
                r=self.monteCarloRuns(self.numberOfRuns, p, copy.deepcopy(weights), user.probabilityFutureBehaviour)
            if r>0:
                reward+= r * M0[p]
        return reward

    def discountCase(self,returnerWeights,M,user,p):
        if user.probabilityFutureBehaviour[p]==0:
            reward=0
        else:
            reward= self.monteCarloRuns(self.numberOfRuns, p, copy.deepcopy(returnerWeights), user.probabilityFutureBehaviour) -self.prices[p]*user.probabilityFutureBehaviour[p]
        if reward!=0:
            reward=reward * M[p]

        return reward


    def matcherAggregated(self,M,M0,user,weights,returnerWeights):
        w=np.zeros(6)
        for p in range(6):
            if p>4:
                # no discount case
                w[p]=self.noDiscountCase(weights,M0,user)
            else:
                # in case we have a discount
                w[p]=self.discountCase(returnerWeights,M,user,p)
        w=np.nan_to_num(w)
        return np.random.choice(np.where(w == max(w))[0])



