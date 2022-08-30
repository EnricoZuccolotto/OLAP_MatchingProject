from Step1_Environment.Products import Products
from Step1_Environment.Products import buyableProducts
import numpy as np
class weightsLearner():
    def __init__(self):
        self.n_nodes=len(buyableProducts())
        self.credits={}
        self.estimated_prob={}
        self.occur_v_active={}
        for p in buyableProducts():
            self.estimated_prob[p]=np.ones(self.n_nodes) * 1.0 / (self.n_nodes - 1)
            self.credits[p] = np.zeros(self.n_nodes)
            self.occur_v_active[p] = np.zeros(self.n_nodes)

    def estimate_prob(self,episode, product):
        episode=np.array(episode)
        print(episode)
        idx_w_active = np.argwhere(episode[:, product.value] == 1).reshape(-1)
        print(str(product)+"   active:"+str(idx_w_active))
        if len(idx_w_active) > 0 and idx_w_active > 0:
            active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
            print(active_nodes_in_prev_step)
            self.credits[product] += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
            print(self.credits[product])
        for v in range(self.n_nodes):
            if v != product.value:
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len(idx_v_active) > 0 and (len(idx_w_active) == 0 or idx_v_active < idx_w_active):
                    self.occur_v_active[product][v] += 1
                    print(self.occur_v_active[product])
        self.estimated_prob[product] = self.credits[product]/ self.occur_v_active[product]
        self.estimated_prob[product] = np.nan_to_num(self.estimated_prob[product])


    def updateEstimates(self,episode):
        for p in buyableProducts():
            self.estimate_prob(episode,p)
