from Bandit.UCBLearner_M0 import UCBLearner_M0
import numpy as np
from CUSUM import CUSUM



class CD_CUSUM_UCB(UCBLearner_M0):
    def __init__(self, n_arms, M=100, eps=0.05, h=20, alpha=0.01):
        super(CD_CUSUM_UCB, self).__init__(n_arms)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.change_detection_M0=[CUSUM(M, eps, h) for _ in range(n_arms)]
        self.detections_M0 = [[] for _ in range(n_arms)]
        self.aplha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.aplha):
            upper_conf = self.means + self.widths
            return upper_conf
        else:
            costs_random = np.random.rand(5)
            return costs_random

    def update(self, pulled_arm, reward):

        if self.change_detection[pulled_arm].update(reward):
            self.detections[pulled_arm].append(self.t)
            self.rewards_per_arm[pulled_arm] = []
            self.change_detection[pulled_arm].reset()

        self.rewards_per_arm[pulled_arm].append(reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.rewards_per_arm])
        for a in range(self.number_arms-1):
            n = len(self.rewards_per_arm[a])
            self.widths[a] = np.sqrt(2 * np.log(total_valid_samples) / (n*(total_valid_samples-1))) if n > 1 else np.inf

    def pull_arm_M0(self):
        if np.random.binomial(1, 1 - self.aplha):
            upper_conf = self.M0 + self.M0_width
            return upper_conf
        else:
            costs_random = np.random.rand(5)
            return costs_random

    def updateM0(self, pulled_arm,landingProduct, reward):
        self.t += 1
        self.update(pulled_arm, reward)
        for idx in range(self.number_arms-1):
            if landingProduct == idx:
                r=reward
            else:
                r=0
            if self.change_detection_M0[idx].update(r):
                self.detections_M0[idx].append(self.t)
                self.rewards_per_product[idx] = []
                self.change_detection_M0[idx].reset()

            self.rewards_per_product[idx].append(r)

        total_valid_samples = sum([len(x) for x in self.rewards_per_arm])
        for a in range(self.number_arms-1):
            n = len(self.rewards_per_product[a])
            self.M0[a] = np.mean(self.rewards_per_product[a])
            self.M0_width[a] = np.sqrt(2 * np.log(total_valid_samples) / (n*(total_valid_samples-1))) if n > 1 else np.inf
