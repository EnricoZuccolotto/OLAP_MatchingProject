
from Bandit.UCBLearner_M0 import UCBLearner_M0
from Bandit.TS_LearnerM0 import TS_LearnerM0

# if ucb equal to 1 use the UCB method to compute probabilities
# if ucb equal to 0 use the TS method
class Learner_M0_M():
    def __init__(self,ucb):
        self.learners=[]
        for p in range(5):
            if ucb:
                self.learners.append(UCBLearner_M0(6))
            else:
                self.learners.append(TS_LearnerM0(6))
        return

    def update(self,firstLandingItem,discountedItem,landingProduct, reward):
        if discountedItem <5:
            self.learners[firstLandingItem].update(discountedItem, reward)
        else:
            self.learners[firstLandingItem].updateM0(discountedItem,landingProduct, reward)

    def pull_arm(self,firstLandingItem):
        return self.learners[firstLandingItem].pull_arm()


    def pull_arm_M0(self,firstLandingItem):
        return self.learners[firstLandingItem].pull_arm_M0()
