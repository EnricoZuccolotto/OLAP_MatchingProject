import copy
import math

import numpy as np


class matchingBestDiscountCode():
    def __init__(self,prices,costs,n):
        self.costs=costs
        self.prices=prices
        self.z=np.zeros(5)
        self.activationProb_weights=np.zeros((5,5))
        self.activationProb_ReturnerWeights = np.zeros((5,5))
        self.numberOfRuns=n
        return
    # product->fist product the user will visit
    # weight associated to user
    # n number of repetitions
    def monteCarloRuns(self, n, product, weights):
        self.z=np.zeros(5)


        # weights = (weights.T * ([0 if p==0 else 1 for p in productsSeen])).T
        # using 1000 runs compared to using 50 runs i make an error of avg 6 percent
        # using 1000 runs compared to using 100 runs i make an error of avg 3 percent (RSS squared error)(experimented on 300 visits)

        for i in range(n):
            self.simulateEpisode(product,copy.deepcopy(weights))

        return self.z/n

    def simulateEpisode(self, product, usableWeights):
        prob_matrix = usableWeights.copy()
        active_nodes  = np.zeros(5)
        active_nodes [product] = 1
        self.z[product]+=1
        newly_active_nodes = active_nodes
        t = 0
        while t < 5 and np.sum(newly_active_nodes) > 0:
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
            self.z[newly_active_nodes==1] += 1
            active_nodes = np.array(active_nodes + newly_active_nodes)


            t += 1

        return




    def productsExplorableByUser(self, product, usableWeights,user):

        prob_matrix = (usableWeights.T * user.probabilityFutureBehaviour>0).T
        active_nodes = np.zeros(5)
        active_nodes[product] = 1
        newly_active_nodes = active_nodes
        t = 0
        while t < 5 and np.sum(newly_active_nodes) > 0:
            p = (prob_matrix.T * newly_active_nodes).T
            prob_matrix = prob_matrix * (p != 0)
            newly_active_nodes = (np.sum(p>0, axis=0) > 0) * (1 - active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)
            t += 1

        return active_nodes

    def updateActivationProb_weights(self,weights):
        self.activationProb_weights=[self.monteCarloRuns(self.numberOfRuns,p,weights) for p in range(5)]

    def updateActivationProb_returnerWeights(self,returnerWeights):
        self.activationProb_ReturnerWeights = [self.monteCarloRuns(self.numberOfRuns, p, returnerWeights[p]) for p in range(5)]

    def matcher(self,M,M0,user,weights,returnerWeights,estimatedQuantities):

        w=np.nan_to_num([self.noDiscountCase(M0,user,returnerWeights,estimatedQuantities) if p>4 else self.discountCase(M,user,p,self.productsExplorableByUser(p,weights,user),estimatedQuantities) for p in range(6)])

        return np.random.choice(np.where(w == max(w))[0])


    def noDiscountCase(self,M0,user,returnerWeights,estimatedQuantities):
        reward=0
        for p in range(5):
            if user.probabilityFutureBehaviour[p]==0:
                r=0
            else:
                visitableNodes=self.productsExplorableByUser(p, returnerWeights[p], user)
                r=np.sum(self.activationProb_weights[p]*(self.prices-self.costs) * user.probabilityFutureBehaviour*visitableNodes*estimatedQuantities)
            if r!=0:
                reward+= r * M0[p]
        return reward

    def discountCase(self,M,user,p,visitableNodes,estimatedQuantities):
        if user.probabilityFutureBehaviour[p] == 0:
            reward = 0
        else:
            reward  = np.sum((self.activationProb_ReturnerWeights[p]*(self.prices-self.costs) * user.probabilityFutureBehaviour*visitableNodes*estimatedQuantities)) - self.prices[p]*user.probabilityFutureBehaviour[p]
        if reward!=0:
            reward=reward * M[p]
        return reward

    def matcherAggregatedReturningUsers(self,weights,returnerWeights,M,M0,user,estimatedQuantities):
        w = np.zeros(6)
        for p in range(6):
            if p > 4:
                # no discount case
                w[p] = self.noDiscountCase(weights, M0, user,estimatedQuantities)
            else:
                # in case we have a discount
                w[p] = self.discountCase(returnerWeights, M, user, p,estimatedQuantities)
        w = np.nan_to_num(w)

        return np.random.choice(np.where(w == max(w))[0])
