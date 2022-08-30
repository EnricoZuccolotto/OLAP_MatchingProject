from Step1_Environment.Products import buyableProducts
from Bandit.UCBLearner_M0 import UCBLearner_M0
from Bandit.TS_LearnerM0 import TS_LearnerM0
from Step1_Environment.Products import Products
# if ucb equal to 1 use the UCB method to compute probabilities
# if ucb equal to 0 use the TS method
class Learner_M0_M():
    def __init__(self,ucb):
        self.learners={}
        for p in buyableProducts():
            if ucb:
                self.learners[p] = UCBLearner_M0(6)
            else:
                self.learners[p] = TS_LearnerM0(6)
        return

    def update(self,firstLandingItem,discountedItem,landingProduct, reward):
        if discountedItem is not None:
            self.learners[firstLandingItem].update(discountedItem.value, reward)
        else:
            if landingProduct is None:
                self.learners[firstLandingItem].updateM0(5, 6, reward)
            else:
                self.learners[firstLandingItem].updateM0(5,landingProduct.value, reward)

    def pull_arm(self,firstLandingItem):
        index=self.learners[firstLandingItem].pull_arm()
        discountedItem=list(Products)[index]

        if discountedItem is Products.P0:
            return None
        else:
            return discountedItem