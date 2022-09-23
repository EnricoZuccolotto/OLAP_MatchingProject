# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import copy
import math
import gc
from Step2_maximization.matcher import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from CreditAssignment.weightsLearner import weightsLearner
from Bandit.Learner_M0_M import Learner_M0_M
import matplotlib.pyplot as plt
# given a user and the page from which the user will start to navigate our website,return the reward
# check if product is diff from P0
# if he didn't buy anything don't call method finalizePurchase and don't assign the discount
# add the user to the possible returner list
# set user.returner=True
import numpy as np

def printProb():
    if ucb:
        for p in range(5):
            print(learner.pull_arm(p))
            print(learner.pull_arm_M0(p))

    else:
        for p in range(6):
            print(p)
            for i in range(5):
                prob=learner.learners[p].beta[i][0]/(learner.learners[p].beta[i][0]+learner.learners[p].beta[i][1])
                print(prob)
            print("M0")
            for i in range(5):
                prob = learner.learners[p].M0_beta[i][0] / (
                            learner.learners[p].M0_beta[i][0] + learner.learners[p].M0_beta[i][1])
                print(prob)
def returningVisit(user1):
    landingProduct = env.returningLandingProduct(user1)
    reward = landingProduct<5*1
    learner.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m
if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
    prices = np.array([13, 10, 17, 3, 7])
    costs = np.array([5, 3, 7, 0.5, 2])

    # initialization of the alpha values
    alphas = np.array([0.2, 0.2, 0.2, 0.1, 0.2, 0.1])

    # initialization of the weights
    w = np.array([[0.7, 0.67, 0.5, 0.35, 0.55],
                  [0.69, 0.54, 0.55, 0.59, 0.55],
                  [0.75, 0.5, 0.58, 0.52, 0.64],
                  [0.7, 0.6, 0.62, 0.53, 0.63],
                  [0.73, 0.55, 0.65, 0.57, 0.59]])

    # initialization of the matrix M
    M = np.array([[0.3, 0.4, 0.38, 0.45, 0.2],
                  [0.2, 0.5, 0.34, 0.4, 0.1],
                  [0.25, 0.31, 0.15, 0.28, 0.44],
                  [0.19, 0.28, 0.42, 0.17, 0.35],
                  [0.35, 0.25, 0.29, 0.5, 0.47]])
    # initialization of the matrix M0
    M0 = np.array([[0.01, 0.05, 0.02, 0.03, 0.025],
                   [0.02, 0.025, 0.015, 0.05, 0.02],
                   [0.03, 0.01, 0.04, 0.05, 0.02],
                   [0.01, 0.017, 0.02, 0.015, 0.05],
                   [0.03, 0.02, 0.05, 0.035, 0.01]])

    # initialization of the weight for each class defined by discounted product
    returnerWeights = np.array([[[0, 0.65, 0.5, 0.4, 0.25],
                                 [0.6, 0.0, 0.35, 0.2, 0.5],
                                 [0.48, 0.5, 0.0, 0.3, 0.4],
                                 [0.49, 0.6, 0.45, 0, 0.55],
                                 [0.35, 0.45, 0.55, 0.65, 0]],
                                [[0, 0.45, 0.48, 0.35, 0.5],
                                 [0.55, 0, 0.45, 0.63, 0.38],
                                 [0.59, 0.47, 0, 0.58, 0.61],
                                 [0.41, 0.45, 0.6, 0, 0.5],
                                 [0.45, 0.39, 0.47, 0.55, 0]],
                                [[0, 0.43, 0.58, 0.55, 0.39],
                                 [0.42, 0, 0.57, 0.61, 0.6],
                                 [0.47, 0.53, 0, 0.58, 0.62],
                                 [0.41, 0.49, 0.58, 0, 0.6],
                                 [0.45, 0.5, 0.35, 0.6, 0]],
                                [[0, 0.5, 0.4, 0.37, 0.61],
                                 [0.39, 0, 0.47, 0.43, 0.53],
                                 [0.65, 0.55, 0, 0.47, 0.6],
                                 [0.5, 0.37, 0.47, 0, 0.44],
                                 [0.51, 0.57, 0.6, 0.63, 0]],
                                [[0, 0.45, 0.42, 0.37, 0.58],
                                 [0.57, 0, 0.5, 0.6, 0.57],
                                 [0.38, 0.45, 0.0, 0.47, 0.5],
                                 [0.52, 0.39, 0.49, 0, 0.5],
                                 [0.65, 0.54, 0.45, 0.59, 0]]]
                               )
    theta = 0.8
    # initialization of the 5 fixed webpages
    pages = np.array([[0, 1, theta, 0, 0],
                      [0, 0, 1, 0, theta],
                      [0, 0, 0, 1, theta],
                      [theta, 0, 0, 0, 1],
                      [1, 0, theta, 0, 0]])
    # initialization of the environment

    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,100)
    ucb = 1

    n_experiment = 5
    horizon = 180
    delay = 30
    rewards_per_exp = []
    regrets_per_exp=[]
    numberOfDailyVisit =150
    # matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    # matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights * pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = Environment(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs, 3)
        learner = Learner_M0_M(ucb)
        weightsLearn=weightsLearner()
        possibleReturnersAtTimeT = []
        instantRegret = []
        instantReward=[]


        for t in range(horizon):
            print(t)
            dailyMargins = [0]
            dailyOptimalMargins = [0]
            possibleReturningUser = []
            randomNumberNewVisits = max(0, int(np.random.normal(numberOfDailyVisit,15)))


            userVisitingToday=env.generateRandomUser(randomNumberNewVisits)

            # get the possible returning user from t-delay time
            if t-delay>=0:
                returnerUsers=possibleReturnersAtTimeT.pop(0)
                userVisitingToday=userVisitingToday+returnerUsers


            np.random.shuffle(userVisitingToday)

            for u in userVisitingToday:

                if u.returner:
                    optimalDiscountedItem = matchingBestDiscountCode.matcher( M[u.firstLandingItem],
                                                                                M0[u.firstLandingItem],u, w * pages,
                                                                        returnerWeights * pages)

                    oldDiscountedItem=int(copy.deepcopy(u.discountedItem))
                    margin = returningVisit(u)
                    weightsLearn.updateEstimates(u.episode)
                    dailyMargins.append(margin)



                    if int(oldDiscountedItem)!=int(optimalDiscountedItem):
                        u.discountedItem = optimalDiscountedItem
                        optimalMargin = returningVisit(u)
                        dailyOptimalMargins.append(optimalMargin)
                    else:

                        dailyOptimalMargins.append(margin)

                    del u,oldDiscountedItem,optimalDiscountedItem

                # if first visit just compute the margin
                else:
                    margin = env.userVisits(u,u.firstLandingItem)
                    if margin >0:
                        possibleReturningUser.append(u)
                        u.discountedItem = matchingBestDiscountCode.matcherAggregated(learner.pull_arm(u.firstLandingItem),
                                                                        learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                weightsLearn.returnWeights()*pages)

            possibleReturnersAtTimeT.append(possibleReturningUser)
            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))
            instantReward.append(math.fsum(dailyMargins))

            del userVisitingToday,margin,dailyMargins,dailyOptimalMargins
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegret = np.cumsum(instantRegret)
        printProb()
        regrets_per_exp.append(cumRegret)
        rewards_per_exp.append(instantReward)
        mean = np.mean(regrets_per_exp, axis=0)
        std = np.std(regrets_per_exp, axis=0) / np.sqrt(e+1)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("regret")
        plt.plot(mean)
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
        plt.savefig('fooo'+str(e)+'.png')
        plt.show()

        plt.figure(1)
        mean = np.mean(rewards_per_exp, axis=0)
        std = np.std(rewards_per_exp, axis=0) / np.sqrt(e + 1)
        plt.xlabel("t")
        plt.ylabel("reward")
        plt.plot(mean)
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
        plt.savefig('reward' + str(e) + '.png')
        plt.show()

        del instantRegret, possibleReturnersAtTimeT, learner, mean, std, env
        gc.collect()


