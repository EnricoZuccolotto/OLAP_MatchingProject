

from Step3_M0_M_Unknown.Bandit.Learner import *


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)  # we have 2 params for each arm: alpha and beta
        self.widths = np.array([np.inf for _ in range(n_arms)])

    def pull_arm(self):
        idx = self.means + self.widths
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        self.updateWidths()

    def updateWidths(self):
        for idx in range(self.number_arms):
            n = len(self.rewards_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2*np.log(self.t) / n)




            # 18  --> 1* & 0.1*
            # 19  --> 0.75* & 0.1*
            else:
                self.widths[idx] = np.inf
