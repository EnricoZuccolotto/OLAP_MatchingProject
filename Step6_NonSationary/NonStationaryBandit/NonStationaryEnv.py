from Step1_Environment.Environment import *
import numpy as np


class NonStationaryEnv(Environment):
    def __init__(self, alphas, weights, returnerWeights, M, M0,  prices, costs,  maxQuantity,horizon):
        super().__init__(alphas, weights, returnerWeights, M, M0,  prices, costs,  maxQuantity)
        self.t = 0
        n_phases = len(self.M)
        self.phase_size = horizon / n_phases

    # if user will return,return the product in which it will lands otherwise return None
    def returningLandingProduct(self, user):
        current_phase = int(self.t / self.phase_size)
        # if user has a discounted item will land for sure on the discounted page
        if user.discountedItem <5:
            if np.random.rand() < self.M[current_phase][user.firstLandingItem][user.discountedItem]:
                return user.discountedItem
        else:
            # No discounted item-->will land to page of product p with probability M0[firstLandingProduct][p],
            # otherwise it will not return

            product = np.random.choice([5, 0, 1, 2, 3, 4], p=np.append([1 - math.fsum(self.M0[current_phase][user.firstLandingItem])],
                                                                       self.M0[current_phase][user.firstLandingItem]))
            # Product.p0 it means the user wont return
            return product

        return 5

    def updateT(self):
        self.t+=1
