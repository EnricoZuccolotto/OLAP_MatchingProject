from Bandit.Learner import *


class TS_learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta = np.ones((n_arms, 2))  # we have 2 params for each arm: alpha and beta

    def pull_arm(self):
        idx = np.random.beta(self.beta[:, 0], self.beta[:, 1])
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta[pulled_arm, 0] += reward
        self.beta[pulled_arm, 1] += 1.0 - reward

