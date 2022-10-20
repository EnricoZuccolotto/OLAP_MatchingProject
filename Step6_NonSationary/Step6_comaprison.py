
import copy
import math
import gc
from Step2_maximization.matcher import matchingBestDiscountCode
from Step6_NonSationary.NonStationaryBandit.NonStationaryEnv import NonStationaryEnv
from Step6_NonSationary.NonStationaryBandit.NonSationary_Learner_M0_M import NonStationary_Learner_M_M0
import matplotlib.pyplot as plt
import numpy as np
import json

def returningVisit(user1,l):
    current_phase = min(1, int((env.t - delay) / env.phase_size))
    landingProduct = env.returningLandingProduct(user1,current_phase)
    reward = 1 if landingProduct < 5 else 0
    l.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    l.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m



if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
    f = open("Step1_Environment\config1.json")
    config = json.loads(f.read())
    prices = np.array(config["prices"])
    costs = np.array(config["costs"])

    # initialization of the alpha values
    alphas = np.array(config["alphas"])

    # initialization of the weights
    w = np.array(config["weights"])

    # initialization of the matrix M
    M = np.array(config["M_CD"])
    # initialization of the matrix M0
    M0 = np.array(config["M0_CD"])

    # initialization of the weight for each class defined by discounted product
    returnerWeights = np.array(config["returnerWeights"])

    # initialization of the 5 fixed webpages
    pages = np.array(config["pages"])

    quantities = np.array(config["quantities"])
    quantities_std = np.array(config["quantities_std"])
    f.close()
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
                    landingProduct = env.returningLandingProduct(u,phase)
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
                                                                                returnerWeights * pages,quantities),matchingBestDiscountCode.matcher(CDLearner.pull_arm(u.firstLandingItem),
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


