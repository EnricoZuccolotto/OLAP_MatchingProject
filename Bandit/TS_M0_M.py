from Step1_Environment.Products import buyableProducts
from Bandit.TS_learner import TS_learner
from Step1_Environment.Products import Products

class TS_M0_M():
    def __init__(self):
        self.learners={}
        for p in buyableProducts():
            self.learners[p]= TS_learner(5)
        return

    def update(self,firstLandingItem,discountedItem,landingProduct, reward):
        if discountedItem is not None:
            self.learners[firstLandingItem].update(discountedItem.value, reward)


    def pull_arm(self,firstLandingItem):
        index=self.learners[firstLandingItem].pull_arm()
        discountedItem=list(Products)[index]

        if discountedItem is Products.P0:
            return None
        else:
            return discountedItem