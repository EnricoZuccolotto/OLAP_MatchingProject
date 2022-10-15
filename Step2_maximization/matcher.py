import copy
from Step2_maximization.monteCarlo import monteCarlo
import numpy as np


class matchingBestDiscountCode():
    def __init__(self,prices,costs,n):
        self.costs=costs
        self.prices=prices
        self.z=np.zeros(5)
        self.numberOfRuns=n
        self.mC=monteCarlo()
        return


    def matcher(self,M,M0,user,weights,returnerWeights,estimatedQuantities):

        w=np.zeros(6)
        for p in range(6):
            if p>4:
                # no discount case
                w[p]=self.noDiscountCase(weights,M0,user,estimatedQuantities)
            else:
                # in case we have a discount
                w[p]=self.discountCase(returnerWeights[p],M,user,p,estimatedQuantities)
        w=np.nan_to_num(w)

        return np.random.choice(np.where(w == max(w))[0])


    def noDiscountCase(self,weights,M0,user,estimatedQuantities):
        reward=0
        for p in range(5):
            if user.probabilityFutureBehaviour[p]==0:
                r=0
            else:
                weights_new = (weights.T * user.probabilityFutureBehaviour).T
                z_p=self.mC.monteCarloRuns(self.numberOfRuns, p, copy.deepcopy(weights_new))
                r = np.sum(z_p * (self.prices - self.costs) * estimatedQuantities)
            if r>0:
                reward+= r * M0[p]
        return reward

    def discountCase(self,returnerWeights,M,user,p,estimatedQuantities):
        if user.probabilityFutureBehaviour[p]==0:
            reward=0
        else:
            weights_new = (returnerWeights.T * user.probabilityFutureBehaviour).T
            z_p= self.mC.monteCarloRuns(self.numberOfRuns, p, copy.deepcopy(weights_new))
            reward = np.sum(z_p * (self.prices - self.costs) * estimatedQuantities)-self.prices[p] * user.probabilityFutureBehaviour[p]
        if reward!=0:
            reward=reward * M[p]

        return reward


    def matcherAggregated(self,M,M0,user,weights,returnerWeights,estimatedQuantities):
        w=np.zeros(6)
        for p in range(6):
            if p>4:
                # no discount case
                w[p]=self.noDiscountCase(weights,M0,user,estimatedQuantities)
            else:
                # in case we have a discount
                w[p]=self.discountCase(returnerWeights,M,user,p,estimatedQuantities)
        w=np.nan_to_num(w)
        return np.random.choice(np.where(w == max(w))[0])



