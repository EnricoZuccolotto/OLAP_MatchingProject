from Step1_Environment.Products import buyableProducts
from Bandit.UCBLearner import UCBLearner
from Step1_Environment.Products import Products

class UCB_M0_M():
    def __init__(self):
        self.learners={}
        for p in buyableProducts():
            self.learners[p]= UCBLearner(5)
        return

    def update(self,firstLandingItem,discountedItem,landingProduct, reward):

        self.learners[firstLandingItem].update(discountedItem.value, reward)


    def pull_arm(self,firstLandingItem):
        index=self.learners[firstLandingItem].pull_arm()
        discountedItem=list(Products)[index]

        if discountedItem is Products.P0:
            return None
        else: return discountedItem