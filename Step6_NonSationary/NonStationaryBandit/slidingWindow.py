import copy

from Step3_M0_M_Unknown.Bandit.UCBLearner_M0 import UCBLearner_M0
import numpy as np



class slidingWindow(UCBLearner_M0):
    def __init__(self, n_arms, window):
        super().__init__(n_arms)
        self.slidingWindow=window
        self.rewards_per_arm_at_time_t=[[] for _ in range(n_arms)]
        self.rewards_per_product_at_time_t = [[] for _ in range(n_arms-1)]


    def update(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)

        total_valid_samples = sum([len(x) for x in self.rewards_per_arm]) + sum(
            [len(np.concatenate(x,axis=None)) for x in self.rewards_per_arm_at_time_t])

        valid_sample_per_arm =np.concatenate((np.concatenate(self.rewards_per_arm_at_time_t[pulled_arm],axis=None),
                               self.rewards_per_arm[pulled_arm]),axis=None)
        self.means[pulled_arm] = np.mean(valid_sample_per_arm)
        for a in range(self.number_arms-1):
            n = len(np.concatenate((np.concatenate(self.rewards_per_arm_at_time_t[a],axis=None),
                               self.rewards_per_arm[a]),axis=None))
            self.widths[a] = np.sqrt(2 * np.log(total_valid_samples) / n) if n > 0 else np.inf


    def updateM0(self, pulled_arm,landingProduct, reward):

        self.update(pulled_arm, reward)
        for idx in range(self.number_arms-1):
            if landingProduct == idx:
                r=reward
            else:
                r=0
            self.rewards_per_product[idx].append(r)
        total_valid_samples = sum([len(x) for x in self.rewards_per_arm])+sum([len(np.concatenate(x,axis=None)) for x in self.rewards_per_arm_at_time_t])
        for a in range(self.number_arms-1):
            valid_sample_per_arm = np.concatenate((np.concatenate(self.rewards_per_product_at_time_t[a],axis=None),
                                   self.rewards_per_product[a]),axis=None)
            n = len(valid_sample_per_arm)
            self.M0[a] = np.mean(valid_sample_per_arm)
            self.M0_width[a] = 0.1*np.sqrt(2 * np.log(total_valid_samples) / n) if n > 0 else np.inf

    def time(self):
        for p in range(6):
            self.rewards_per_arm_at_time_t[p].append(copy.deepcopy(self.rewards_per_arm[p]))
            if p<5:
                self.rewards_per_product_at_time_t[p].append(copy.deepcopy(self.rewards_per_product[p]))


        self.rewards_per_arm = [[] for _ in range(self.number_arms)]
        self.rewards_per_product = [[] for _ in range(self.number_arms - 1)]
        self.t += 1

        if self.t>self.slidingWindow:
            for p in range(6):
                self.rewards_per_arm_at_time_t[p].pop(0)
                if p<5:
                    self.rewards_per_product_at_time_t[p].pop(0)




