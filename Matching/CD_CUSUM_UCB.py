from UCBSuperArmMatching import UCBSuperArmMatching
import numpy as np
from CUSUM import CUSUM
from scipy.optimize import linear_sum_assignment


class CD_CUSUM_UCB(UCBSuperArmMatching):
    def __init__(self, n_arms, n_cols, n_rows, M=100, eps=0.05, h=20, alpha=0.01):
        super(CD_CUSUM_UCB, self).__init__(n_arms, n_rows, n_cols)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arm = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.aplha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.aplha):
            upper_conf = self.means + self.widhts
            upper_conf[np.isinf(upper_conf)] = 1000
            row_idx, col_idx = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
            return row_idx, col_idx
        else:
            costs_random = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arms_flat, rewards):
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arm[pulled_arm] = []
                self.change_detection[pulled_arm].reset()

            self.update_observations(pulled_arm, reward)
            self.means[pulled_arm] = np.mean(self.valid_rewards_per_arm[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arm])
        for a in range(self.number_arms):
            n = len(self.valid_rewards_per_arm[a])
            self.widhts[a] = np.sqrt(2 * np.log(total_valid_samples) / n) if n > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
