

from Bandit.UCBLearner import *

class UCBLearner_M0(UCBLearner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.M0 = np.zeros(n_arms-1)
        self.M0_width = np.array([np.inf for _ in range(n_arms-1)])
        self.rewards_per_product=[[] for _ in range(n_arms-1)]

    def updateM0(self, pulled_arm,landingProduct, reward):
        self.update(pulled_arm,reward)
        for idx in range(self.number_arms-1):
            if landingProduct==idx:
                self.rewards_per_product[landingProduct].append(reward)
            else:
                self.rewards_per_product[idx].append(0)
            n = len(self.rewards_per_product[idx])
            if n > 0:
                self.M0_width[idx] = np.sqrt(2 * np.log(self.t) / n)
                self.M0[idx] = np.mean(self.rewards_per_product[idx])
            else:
                self.M0_width[idx] = np.inf

    def pull_arm_M0(self):
        idx = self.M0 + self.M0_width
        return idx



