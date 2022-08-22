# Learner class
#
# Defined by:
# * Number of arms it can pulls
# * Number of rounds
# * The list of the collected rewards

# The learner interacts with the environment by selecting the arm to pull at each round
# and observing the reward given by the environment
import numpy as np


class Learner():
    def __init__(self, number_arms):
        self.number_arms = number_arms
        self.t = 0  # number of rounds
        self.rewards_per_arm = x = [[] for i in range(number_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
