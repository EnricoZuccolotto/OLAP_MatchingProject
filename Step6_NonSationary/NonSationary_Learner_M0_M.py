from Step6_NonSationary.CD_UCB import CD_CUSUM_UCB
from Bandit.Learner_M0_M import Learner_M0_M


# if ucb equal to 1 use the UCB method to compute probabilities
# if ucb equal to 0 use the TS method
class NonStationary_Learner_M_M0(Learner_M0_M):
    def __init__(self):
        super(NonStationary_Learner_M_M0, self).__init__(0)
        self.learners=[CD_CUSUM_UCB(6) for _ in range(5)]
        return

