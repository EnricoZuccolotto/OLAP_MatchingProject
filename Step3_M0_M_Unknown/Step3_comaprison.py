
import copy
import math
import gc
from Step2_maximization.matcher import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from Step3_M0_M_Unknown.Bandit.Learner_M0_M import Learner_M0_M
import matplotlib.pyplot as plt
import numpy as np
import json

def returningVisit(user1,l):
    landingProduct = env.returningLandingProduct(user1)
    reward = 1 if landingProduct < 5 else 0
    l.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    l.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m



if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
    # initialization of the prices and costs
    f = open("Step1_Environment\config1.json")
    config = json.loads(f.read())
    print(config)
    prices = np.array(config["prices"])
    costs = np.array(config["costs"])

    # initialization of the alpha values
    alphas = np.array(config["alphas"])

    # initialization of the weights
    w = np.array(config["weights"])

    # initialization of the matrix M
    M = np.array(config["M"])
    # initialization of the matrix M0
    M0 = np.array(config["M0"])

    # initialization of the weight for each class defined by discounted product
    returnerWeights = np.array(config["returnerWeights"])

    # initialization of the 5 fixed webpages
    pages = np.array(config["pages"])

    quantities = np.array(config["quantities"])
    quantities_std = np.array(config["quantities_std"])
    f.close()
    # initialization of the environment

    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,100)
    ucb = 1

    n_experiment = 10
    horizon = 365
    delay = 30
    regrets_per_exp_TS = []
    rewards_per_exp_TS = []
    regrets_per_exp_UCB = []
    rewards_per_exp_UCB = []
    numberOfDailyVisit =150
    # matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    # matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights * pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = Environment(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs,quantities,quantities_std)
        TS_learner = Learner_M0_M(0)
        UCB_learner = Learner_M0_M(1)

        possibleReturnersAtTimeT = []


        instantRegretTS = []

        instantRewardTS=[]
        instantRegretUCB = []

        instantRewardUCB = []

        for t in range(horizon):
            print('time:'+str(t))
            margin=-1
            dailyMarginsTS = [0]
            dailyMarginsUCB = [0]
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
                                                                        returnerWeights * pages,quantities)

                    oldDiscountedItemSliding=int(copy.deepcopy(u.discountedItem[0]))
                    oldDiscountedItemCD = int(copy.deepcopy(u.discountedItem[1]))
                    # visit sliding learner
                    u.discountedItem=copy.deepcopy(oldDiscountedItemSliding)
                    marginTS = returningVisit(u, TS_learner)
                    # visit CD learner
                    u.discountedItem = copy.deepcopy(oldDiscountedItemCD)
                    marginUCB = returningVisit(u, UCB_learner)
                    # optimal visit
                    u.discountedItem = optimalDiscountedItem
                    landingProduct = env.returningLandingProduct(u)
                    optimalMargin = env.userVisits(u, landingProduct)
                    # append optimal margin for both
                    dailyOptimalMargins.append(optimalMargin)

                    # if all 3 equals
                    if optimalDiscountedItem==oldDiscountedItemCD and optimalDiscountedItem==oldDiscountedItemSliding:
                        dailyMarginsUCB.append(optimalMargin)
                        dailyMarginsTS.append(optimalMargin)
                    # if all 3 different
                    if optimalDiscountedItem!=oldDiscountedItemSliding and optimalDiscountedItem!=oldDiscountedItemCD and oldDiscountedItemSliding!=oldDiscountedItemCD:
                        dailyMarginsUCB.append(marginUCB)
                        dailyMarginsTS.append(marginTS)
                    # if both different from optimal but equal
                    if optimalDiscountedItem != oldDiscountedItemCD and oldDiscountedItemSliding == oldDiscountedItemCD:
                        dailyMarginsUCB.append(marginUCB)
                        dailyMarginsTS.append(marginUCB)
                    # 1 equal optimal the other is different
                    if optimalDiscountedItem==oldDiscountedItemSliding and optimalDiscountedItem!=oldDiscountedItemCD:
                        dailyMarginsUCB.append(marginUCB)
                        dailyMarginsTS.append(optimalMargin)
                    # 1 equal optimal the other is different
                    if optimalDiscountedItem==oldDiscountedItemCD and optimalDiscountedItem!=oldDiscountedItemSliding:
                        dailyMarginsUCB.append(optimalMargin)
                        dailyMarginsTS.append(marginTS)



                    del u,oldDiscountedItemSliding,oldDiscountedItemCD,optimalDiscountedItem

                # if first visit just compute the margin
                else:
                    margin = env.userVisits(u,u.firstLandingItem)
                    if margin >0:
                        dailyMarginsTS.append(margin)
                        dailyMarginsUCB.append(margin)
                        dailyOptimalMargins.append(margin)
                        possibleReturningUser.append(u)
                        u.discountedItem =[matchingBestDiscountCode.matcher(TS_learner.pull_arm(u.firstLandingItem),
                                                                            TS_learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                            returnerWeights * pages,quantities),matchingBestDiscountCode.matcher(UCB_learner.pull_arm(u.firstLandingItem),
                                                                                                                                      UCB_learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                                                                      returnerWeights * pages,quantities)]

            possibleReturnersAtTimeT.append(possibleReturningUser)
            instantRegretTS.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMarginsTS))
            instantRewardTS.append(math.fsum(dailyMarginsTS))
            instantRegretUCB.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMarginsUCB))
            instantRewardUCB.append(math.fsum(dailyMarginsUCB))
            del userVisitingToday,margin,dailyOptimalMargins,dailyMarginsTS,dailyMarginsUCB
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegretSliding = np.cumsum(instantRegretTS)
        rewards_per_exp_TS.append(instantRewardTS)
        regrets_per_exp_TS.append(cumRegretSliding)
        mean = np.mean(regrets_per_exp_TS, axis=0)
        std = np.std(regrets_per_exp_TS, axis=0) / np.sqrt(e + 1)
        cumRegretCD = np.cumsum(instantRegretUCB)
        rewards_per_exp_UCB.append(instantRewardUCB)
        regrets_per_exp_UCB.append(cumRegretCD)
        meanCD = np.mean(regrets_per_exp_UCB, axis=0)
        stdCD = np.std(regrets_per_exp_UCB, axis=0) / np.sqrt(e + 1)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("regret")
        plt.plot(mean,color='green',label='TS')
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4,color='green')
        plt.plot(meanCD, color='blue',label='UCB')
        plt.fill_between(range(horizon), meanCD - stdCD, meanCD + stdCD, alpha=0.4,color='blue')
        plt.savefig('fooo'+str(e)+'.png')
        plt.legend()
        plt.clf()

        plt.figure(1)
        mean = np.mean(rewards_per_exp_TS, axis=0)
        std = np.std(rewards_per_exp_TS, axis=0) / np.sqrt(e + 1)
        meanCD = np.mean(rewards_per_exp_UCB, axis=0)
        stdCD = np.std(rewards_per_exp_UCB, axis=0) / np.sqrt(e + 1)
        plt.xlabel("t")
        plt.ylabel("reward")
        plt.plot(mean,color='green',label='TS')
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4,color='green')
        plt.plot(meanCD, color='blue', label='UCB')
        plt.fill_between(range(horizon), meanCD - stdCD, meanCD + stdCD, alpha=0.4,color='blue')
        plt.savefig('reward' + str(e) + '.png')
        plt.legend()
        plt.clf()

        del instantRegretTS, possibleReturnersAtTimeT, TS_learner,UCB_learner,instantRegretUCB, mean, std,meanCD,stdCD, env
        gc.collect()


