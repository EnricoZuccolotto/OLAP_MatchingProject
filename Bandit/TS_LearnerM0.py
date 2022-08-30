
from Bandit.TS_learner import *

class TS_LearnerM0(TS_learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.M0_beta = np.ones((n_arms-1, 2))


    def updateM0(self, pulled_arm,landingProduct, reward):

        self.update_observations(pulled_arm, reward)
        for idx in range(self.number_arms-1):
            if landingProduct==idx:
                self.M0_beta[idx, 0] += reward
                self.M0_beta[idx, 1] += 1.0 - reward
            else:
                self.M0_beta[idx, 0] += 0
                self.M0_beta[idx, 1] += 1.0


