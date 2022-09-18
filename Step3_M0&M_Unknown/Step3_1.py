# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import copy
import math
import gc
from Step2_maximization.matchingBestDiscountCode import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
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
    reward = landingProduct<5
    learner.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m
if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
    prices = np.array([2,3,4,5,7])
    costs = np.array([0,0,0,0,0])

    # initialization of the alpha values
    alphas = np.array([0.2,0.2,0.2,0.1,0.2,0.1])



    # initialization of the weights
    w = np.array([[0.6, 0.5, 1, 0, 0],
                  [0, 0, 0.6, 0, 0.4],
                  [0, 0, 0, 0.5, 0.3],
                  [0.4, 0, 0, 0, 0.2],
                  [0.4, 0, 1, 0, 0]])


    # initialization of the matrix M
    M = np.array([[0.4, 0.5, 1, 0.4, 0],
                  [0, 0, 0.6, 0, 0.4],
                  [0, 0, 0, 0.5, 0.3],
                  [0.4, 0, 0, 0, 0.2],
                  [0.4, 0, 1, 0, 0]])
    # initialization of the matrix M0
    M0 = np.array([[0.1, 0.05, 0.1, 0, 0],
                  [0, 0, 0.06, 0, 0.04],
                  [0, 0, 0, 0.05, 0.03],
                  [0.04, 0, 0, 0, 0.02],
                  [0.04, 0, 0.01, 0, 0]])


    # initialization of the weight for each class defined by discounted product
    returnerWeights =np.array([[[0, 0.5, 1, 0, 0],
                                [0, 0, 0.6, 0, 0.4],
                                [0, 0, 0, 0.5, 0.3],
                                 [0.4, 0, 0, 0, 0.2],
                                 [0.4, 0, 1, 0, 0]],
                              [[0, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]],
                              [[0.7, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]],
                              [[0, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]],
                              [[0.6, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]]]
                              )
    theta=0.8
    # initialization of the 5 fixed webpages
    pages = np.array([[0, 1,theta, 0, 0],
                      [0, 0, 1, 0,theta],
                      [0, 0, 0, 1, theta],
                      [theta, 0, 0, 0, 1],
                      [1, 0, theta, 0, 0]])
    # initialization of the environment

    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,1000)
    ucb = 1

    n_experiment = 1
    horizon = 180
    delay = 1
    rewards_per_exp = []
    numberOfDailyVisit =300

    matchingBestDiscountCode.updateActivationProb_weights(w*pages)
    matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights*pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = Environment(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs, 3)
        learner = Learner_M0_M(ucb)
        possibleReturnersAtTimeT = []
        instantRegret = []



        for t in range(horizon):
            print(t)
            dailyMargins = [0]
            dailyOptimalMargins = [0]
            possibleReturningUser = []
            randomNumberNewVisits = max(10, int(np.random.normal(numberOfDailyVisit,15)))

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

                # if the user actually navigated our website
                # add it to possible returners and give the appropriate discount
                    if margin>=0:
                        if margin >0:
                            u.discountedItem = matchingBestDiscountCode.matcher(learner.pull_arm(u.firstLandingItem),
                                                                        learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                returnerWeights * pages)

            possibleReturnersAtTimeT.append(possibleReturningUser)
            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))

            del userVisitingToday,margin,dailyMargins,dailyOptimalMargins
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegret = np.cumsum(instantRegret)
        printProb()
        rewards_per_exp.append(cumRegret)
        mean = np.mean(rewards_per_exp, axis=0)
        std = np.std(rewards_per_exp, axis=0) / np.sqrt(e+1)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("regret")
        plt.plot(mean)
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
        plt.savefig('fooo'+str(e)+'.png')
        plt.show()

        del instantRegret, possibleReturnersAtTimeT, learner, mean, std, env
        gc.collect()


