import copy
import math

import numpy as np
from Step2_maximization.monteCarlo import monteCarlo


class matchingBestDiscountCode():
    def __init__(self,prices,costs,n):
        self.costs=costs
        self.prices=prices
        self.z=np.zeros(5)
        self.activationProb_weights=np.zeros((5,5))
        self.activationProb_ReturnerWeights = np.zeros((5,5))
        self.activationProb_weights_optimal = np.zeros((5, 5))
        self.activationProb_ReturnerWeights_optimal = np.zeros((5, 5))
        self.numberOfRuns=n
        self.mC=monteCarlo()
        return
    # product->fist product the user will visit
    # weight associated to user
    # n number of repetitions





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
        self.activationProb_weights=[self.mC.monteCarloRuns(self.numberOfRuns,p,weights) for p in range(5)]

    def updateActivationProb_returnerWeights(self,returnerWeights):
        self.activationProb_ReturnerWeights = [self.mC.monteCarloRuns(self.numberOfRuns, p, returnerWeights[p]) for p in range(5)]

    def updateActivationProb_weights_optimal(self, weights):
        self.activationProb_weights_optimal = [self.mC.monteCarloRuns(self.numberOfRuns, p, weights) for p in range(5)]

    def updateActivationProb_returnerWeights_optimal(self, returnerWeights):
        self.activationProb_ReturnerWeights_optimal = [self.mC.monteCarloRuns(self.numberOfRuns, p, returnerWeights[p]) for p in
                                               range(5)]

    def matcher(self,M,M0,user,weights,returnerWeights,estimatedQuantities):

        w=np.nan_to_num([self.noDiscountCase(M0,user,returnerWeights,estimatedQuantities,self.activationProb_weights) if p>4 else self.discountCase(M,user,p,self.productsExplorableByUser(p,weights,user),estimatedQuantities,self.activationProb_ReturnerWeights[p]) for p in range(6)])

        return np.random.choice(np.where(w == max(w))[0])

    def matcher_opt(self,M,M0,user,weights,returnerWeights,estimatedQuantities):

        w=np.nan_to_num([self.noDiscountCase(M0,user,returnerWeights,estimatedQuantities,self.activationProb_weights_optimal) if p>4 else self.discountCase(M,user,p,self.productsExplorableByUser(p,weights,user),estimatedQuantities,self.activationProb_ReturnerWeights_optimal[p]) for p in range(6)])

        return np.random.choice(np.where(w == max(w))[0])



    def noDiscountCase(self,M0,user,returnerWeights,estimatedQuantities,activation_prob):
        reward=0
        for p in range(5):
            if user.probabilityFutureBehaviour[p]==0:
                r=0
            else:
                visitableNodes=self.productsExplorableByUser(p, returnerWeights[p], user)
                r=np.sum(activation_prob[p]*(self.prices-self.costs) * user.probabilityFutureBehaviour*visitableNodes*estimatedQuantities)
            if r!=0:
                reward+= r * M0[p]
        return reward

    def discountCase(self,M,user,p,visitableNodes,estimatedQuantities,activation_prob):
        if user.probabilityFutureBehaviour[p] == 0:
            reward = 0
        else:
            reward  = np.sum((activation_prob*(self.prices-self.costs) * user.probabilityFutureBehaviour*visitableNodes*estimatedQuantities)) - self.prices[p]*user.probabilityFutureBehaviour[p]
        if reward!=0:
            reward=reward * M[p]
        return reward

