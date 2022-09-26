import copy


import numpy as np
class weightsLearner():
    def __init__(self):
        self.n_nodes=5
        self.credits=np.zeros((self.n_nodes,self.n_nodes))

        self.occur_v_active=np.zeros((self.n_nodes,self.n_nodes))

        self.estimated_prob=np.array([np.ones(self.n_nodes) * 1.0 / (self.n_nodes - 1)]*5)


    def estimate_prob(self,episode, product):
        episode=np.array(episode)

        idx_w_active = np.argwhere(episode[:, product] == 1).reshape(-1)

        if len(idx_w_active) > 0 and idx_w_active > 0:
            active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)

            self.credits[product] += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)

        for v in range(self.n_nodes):
            if v != product:
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len(idx_v_active) > 0 and (len(idx_w_active) == 0 or idx_v_active < idx_w_active):
                    self.occur_v_active[product][v] += 1

        self.estimated_prob[product] = self.credits[product]/ self.occur_v_active[product]
        self.estimated_prob[product] = np.nan_to_num(self.estimated_prob[product])


    def updateEstimates(self,episode):
        for p in range(5):
            self.estimate_prob(episode,p)

    def returnWeights(self):

        return self.estimated_prob.T