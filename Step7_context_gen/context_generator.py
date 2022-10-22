import copy

import numpy as np

from Step4_WeightsUnknown.weightsLearner import weightsLearner
from Step2_maximization.monteCarlo import monteCarlo

class context_generator():
    def __init__(self,margins,prices):
        self.margins=margins
        self.prices=prices
        self.context=[]
        self.mC=monteCarlo()


    def init(self):
        weightsLearn = weightsLearner()
        self.context.append([1 for _ in range(5)])
        return [weightsLearn for _ in range(5)]

    def test(self,data_set):
        learners = [weightsLearner() for _ in range(5)]
        # train each different learner
        self.train_learners(data_set, learners)
        return learners

    def new_context(self,data_set,quantities,oldLearners,pages):
        tot_samples = np.sum([len(data_set[i]) for i in range(5)])
        lowerBound=np.sqrt(-np.log(0.10)/(2*tot_samples))
        meanValuesOldContext = [
            (np.sum((self.mC.monteCarloRuns(1000, i, oldLearners[i].returnWeights()*(pages>0))-lowerBound) * self.margins * quantities)-self.prices[i]) / 2 for i in
            range(5)]



        # search for new context
        learners=[weightsLearner() for _ in range(5)]
        # train each different learner
        self.train_learners(data_set,learners)
        valueSingleFeature = np.array(
            [(np.sum(self.mC.monteCarloRuns(1000, i, learners[i].returnWeights()*(pages>0)) * self.margins * quantities)-self.prices[i]) / 2 for
             i
             in range(5)])

        # TODO: iterate
        while np.sum(self.context[0])>1:
            # select the most promising features

            mask=np.array(self.context[0])== 0
            print(valueSingleFeature[mask])
            valueSingleFeature[mask]=0
            mostPromisingFeature = np.random.choice(
                    np.where(valueSingleFeature == max(valueSingleFeature))[
                        0])
            print(mostPromisingFeature)
            # generate new context
            new_context =copy.deepcopy(self.context)
            new_context.append([0 for _ in range(5)])
            new_context[0][mostPromisingFeature]=0
            new_context[len(new_context)-1][mostPromisingFeature]=1
            # we generate new Learners and assign them correctly to the context
            nL=[weightsLearner() for _ in range(len(new_context))]
            newLearners=[nL[0] for _ in range(5)]
            for j in range(1,len(new_context)):
                for i in range(len(new_context[j])):
                    if new_context[j][i]==1:
                        newLearners[i]=nL[j]

            # we train the new generated learners
            self.train_learners(data_set,newLearners)
            tot_samples_per_arm = np.array([len(data_set[i]) for i in range(5)])
            samples=[np.sum(tot_samples_per_arm[np.array(new_context[i])>0]) for i in range(len(new_context))]
            lowerBounds=[np.sqrt(-np.log(0.10)/(2*samples[i])) for i in range(len(new_context))]
            # we evaluate them
            valuesSingleFeatureNew=[((np.sum((self.mC.monteCarloRuns(1000,i,newLearners[i].returnWeights()*(pages>0))-lowerBounds[nL.index(newLearners[i])])* self.margins * quantities)-self.prices[i])/2) for i in range(5)]


            if valuesSingleFeatureNew[mostPromisingFeature]>=meanValuesOldContext[mostPromisingFeature]:
                self.context=new_context
                learners=newLearners
                meanValuesOldContext=valuesSingleFeatureNew
            else:
                break




        # generate random numbers and decide how to try to divide
        # for each division compute the probabilities
        # multiply the probs per margins
        # using lower confidence bound
        # sum over the partions >= old context
        # return new weights learners
        return learners


    def train_learners(self,data_set,learners):

        for l in learners:
            single_set=data_set[learners.index(l)]
            for e in single_set:
                l.updateEstimates(e)