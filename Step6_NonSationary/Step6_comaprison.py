
import copy
import math
import gc
from Step2_maximization.matcher import matchingBestDiscountCode
from Step6_NonSationary.NonStationaryBandit.NonStationaryEnv import NonStationaryEnv
from Step6_NonSationary.NonStationaryBandit.NonSationary_Learner_M0_M import NonStationary_Learner_M_M0
import matplotlib.pyplot as plt
import numpy as np

def returningVisit(user1,l):
    landingProduct = env.returningLandingProduct(user1)
    if landingProduct<5:
        reward=1
    else:
        reward=0
    l.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    l.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m



if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
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
    M = np.array([[[0.3, 0.4, 0.38, 0.45, 0.2],
                   [0.2, 0.5, 0.34, 0.4, 0.1],
                   [0.25, 0.31, 0.15, 0.28, 0.44],
                   [0.19, 0.28, 0.42, 0.17, 0.35],
                   [0.35, 0.25, 0.29, 0.5, 0.47]],
                  [[0.03, 0.04, 0.038, 0.045, 0.02],
                   [0.9, 0.9, 0.94, 0.9, 0.90],
                   [0.025, 0.031, 0.15, 0.28, 0.44],
                   [0.019, 0.028, 0.42, 0.017, 0.35],
                   [0.035, 0.025, 0.29, 0.5, 0.47]]]
                 )
    # initialization of the matrix M0
    M0 = np.array([[[0.01, 0.05, 0.02, 0.03, 0.025],
                    [0.02, 0.025, 0.015, 0.05, 0.02],
                    [0.03, 0.01, 0.04, 0.05, 0.02],
                    [0.01, 0.017, 0.02, 0.015, 0.05],
                    [0.03, 0.02, 0.05, 0.035, 0.01]],
                   [[0.01, 0.05, 0.02, 0.003, 0.025],
                    [0.02, 0.025, 0.015, 0.05, 0.02],
                    [0.03, 0.001, 0.04, 0.005, 0.002],
                    [0.01, 0.0017, 0.002, 0.015, 0.005],
                    [0.003, 0.02, 0.05, 0.0035, 0.01]]]
                  )

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
    quantities = [1, 2, 3, 4, 5]
    quantities_std = [1, 2, 1, 1, 1]
    # initialization of the environment

    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,1000)
    ucb = 1

    n_experiment = 20
    horizon = 365
    delay = 30
    regrets_per_exp_sliding = []
    rewards_per_exp_sliding = []
    regrets_per_exp_CD = []
    rewards_per_exp_CD = []
    numberOfDailyVisit =150
    # matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    # matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights * pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = NonStationaryEnv(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs,quantities,quantities_std,horizon)
        slidingLearner = NonStationary_Learner_M_M0(2 * int(np.sqrt(horizon)))
        CDLearner=NonStationary_Learner_M_M0(0)

        possibleReturnersAtTimeT = []
        instantRegretSliding = []

        instantRewardSliding=[]
        instantRegretCD = []

        instantRewardCD = []

        for t in range(horizon):
            print('time:'+str(t))
            margin=-1
            dailyMarginsSliding = [0]
            dailyMarginsCD = [0]
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
                    phase=int((t-delay) / env.phase_size)
                    optimalDiscountedItem = matchingBestDiscountCode.matcher( M[phase][u.firstLandingItem],
                                                                                M0[phase][u.firstLandingItem],u, w * pages,
                                                                        returnerWeights * pages,quantities)

                    oldDiscountedItemSliding=int(copy.deepcopy(u.discountedItem[0]))
                    oldDiscountedItemCD = int(copy.deepcopy(u.discountedItem[1]))
                    # visit sliding learner
                    u.discountedItem=copy.deepcopy(oldDiscountedItemSliding)
                    marginSliding = returningVisit(u,slidingLearner)
                    # visit CD learner
                    u.discountedItem = copy.deepcopy(oldDiscountedItemCD)
                    marginCD = returningVisit(u, CDLearner)
                    # optimal visit
                    u.discountedItem = optimalDiscountedItem
                    landingProduct = env.returningLandingProduct(u)
                    optimalMargin = env.userVisits(u, landingProduct)
                    # append optimal margin for both
                    dailyOptimalMargins.append(optimalMargin)

                    # if all 3 equals
                    if optimalDiscountedItem==oldDiscountedItemCD and optimalDiscountedItem==oldDiscountedItemSliding:
                        dailyMarginsCD.append(optimalMargin)
                        dailyMarginsSliding.append(optimalMargin)
                    # if all 3 different
                    if optimalDiscountedItem!=oldDiscountedItemSliding and optimalDiscountedItem!=oldDiscountedItemCD and oldDiscountedItemSliding!=oldDiscountedItemCD:
                        dailyMarginsCD.append(marginCD)
                        dailyMarginsSliding.append(marginSliding)
                    # if both different from optimal but equal
                    if optimalDiscountedItem != oldDiscountedItemCD and oldDiscountedItemSliding == oldDiscountedItemCD:
                        dailyMarginsCD.append(marginCD)
                        dailyMarginsSliding.append(marginCD)
                    # 1 equal optimal the other is different
                    if optimalDiscountedItem==oldDiscountedItemSliding and optimalDiscountedItem!=oldDiscountedItemCD:
                        dailyMarginsCD.append(marginCD)
                        dailyMarginsSliding.append(optimalMargin)
                    # 1 equal optimal the other is different
                    if optimalDiscountedItem==oldDiscountedItemCD and optimalDiscountedItem!=oldDiscountedItemSliding:
                        dailyMarginsCD.append(optimalMargin)
                        dailyMarginsSliding.append(marginSliding)



                    del u,oldDiscountedItemSliding,oldDiscountedItemCD,optimalDiscountedItem,phase

                # if first visit just compute the margin
                else:
                    margin = env.userVisits(u,u.firstLandingItem)

                    if margin >0:
                        dailyMarginsSliding.append(margin)
                        dailyMarginsCD.append(margin)
                        dailyOptimalMargins.append(margin)
                        possibleReturningUser.append(u)
                        u.discountedItem =[matchingBestDiscountCode.matcher(slidingLearner.pull_arm(u.firstLandingItem),
                                                                                slidingLearner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                returnerWeights * pages,means),matchingBestDiscountCode.matcher(CDLearner.pull_arm(u.firstLandingItem),
                                                                                CDLearner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                returnerWeights * pages,quantities)]

            possibleReturnersAtTimeT.append(possibleReturningUser)
            env.updateT()
            slidingLearner.time()
            instantRegretSliding.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMarginsSliding))
            instantRewardSliding.append(math.fsum(dailyMarginsSliding))
            instantRegretCD.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMarginsCD))
            instantRewardCD.append(math.fsum(dailyMarginsCD))
            del userVisitingToday,margin,dailyOptimalMargins,dailyMarginsSliding,dailyMarginsCD
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegretSliding = np.cumsum(instantRegretSliding)
        rewards_per_exp_sliding.append(instantRewardSliding)
        regrets_per_exp_sliding.append(cumRegretSliding)
        mean = np.mean(regrets_per_exp_sliding, axis=0)
        std = np.std(regrets_per_exp_sliding, axis=0) / np.sqrt(e + 1)
        cumRegretCD = np.cumsum(instantRegretCD)
        rewards_per_exp_CD.append(instantRewardCD)
        regrets_per_exp_CD.append(cumRegretCD)
        meanCD = np.mean(regrets_per_exp_CD, axis=0)
        stdCD = np.std(regrets_per_exp_CD, axis=0) / np.sqrt(e + 1)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("regret")
        plt.plot(mean,color='green',label='Slider')
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4,color='green')
        plt.plot(meanCD, color='blue',label='Change detection')
        plt.fill_between(range(horizon), meanCD - stdCD, meanCD + stdCD, alpha=0.4,color='blue')
        plt.legend()
        plt.savefig('fooo.png')
        plt.clf()



        plt.figure(1)
        mean = np.mean(rewards_per_exp_sliding, axis=0)
        std = np.std(rewards_per_exp_sliding, axis=0) / np.sqrt(e + 1)
        meanCD = np.mean(rewards_per_exp_CD, axis=0)
        stdCD = np.std(rewards_per_exp_CD, axis=0) / np.sqrt(e + 1)
        plt.xlabel("t")
        plt.ylabel("reward")
        plt.plot(mean,color='green',label='Slider')
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4,color='green')
        plt.plot(meanCD, color='blue', label='Change detection')
        plt.fill_between(range(horizon), meanCD - stdCD, meanCD + stdCD, alpha=0.4,color='blue')
        plt.legend()
        plt.savefig('reward.png')
        plt.clf()



        del instantRegretSliding, possibleReturnersAtTimeT, slidingLearner,CDLearner,instantRegretCD, mean, std,meanCD,stdCD, env
        gc.collect()


