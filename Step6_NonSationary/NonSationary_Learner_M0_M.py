from Step6_NonSationary.CD_UCB import CD_CUSUM_UCB
from Bandit.Learner_M0_M import Learner_M0_M

class NonStationary_Learner_M_M0(Learner_M0_M):
    def __init__(self):
        super(NonStationary_Learner_M_M0, self).__init__(0)
        self.learners=[CD_CUSUM_UCB(6) for _ in range(5)]
        return


from Step6_NonSationary.slidingWindow import slidingWindow
from Bandit.Learner_M0_M import Learner_M0_M



class NonStationary_Sliding_Learner_M_M0(Learner_M0_M):
    def __init__(self, window):
        super().__init__(1)
        if window>0:
            self.learners=[slidingWindow(6,window) for _ in range(5)]
        else:
            self.learners = [CD_CUSUM_UCB(6) for _ in range(5)]
        return

    def time(self):
        for p in range(5):
            self.learners[p].time()

