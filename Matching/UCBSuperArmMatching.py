from Bandit.UCBLearner import UCBLearner
import numpy as np
from scipy.optimize import linear_sum_assignment


class UCBSuperArmMatching(UCBLearner):
    def __init__(self, n_arms, n_cols, n_rows):
        super(UCBSuperArmMatching, self).__init__(n_arms=n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        assert n_arms == n_cols * n_rows

    def pull_arm(self):
        upper_conf = self.means + self.widths
        upper_conf[np.isinf(upper_conf)] = 1000
        row_ind, col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
        return row_ind, col_ind

    def update(self, pulled_arms, rewards):
        self.t += 1
        pulled_arms_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for a in range(self.number_arms):
            n = len(self.rewards_per_arm[a])
            self.widths[a] = np.sqrt(2 * np.log(self.t) / n) if n > 0 else np.inf
        for pulled_arm, reward in zip(pulled_arms_flat, rewards):
            self.update_observations(pulled_arm, reward)
            self.means[pulled_arm] = self.means[pulled_arm] * (self.t - 1 + reward) / self.t
