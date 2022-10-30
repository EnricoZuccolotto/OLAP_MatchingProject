
import copy
import math
import gc
from Step2_maximization.heuristic import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from Step4_WeightsUnknown.weightsLearner import weightsLearner
from Step3_M0_M_Unknown.Bandit.Learner_M0_M import Learner_M0_M
import matplotlib.pyplot as plt
import numpy as np
import json

def returningVisit(user1):
    landingProduct = env.returningLandingProduct(user1)
    reward = 1 if landingProduct < 5 else 0
    learner.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    if landingProduct<5 and oldDiscountedItem<5:
        weightsLearn.updateEstimates(user1.episode)
    return m
if __name__ == '__main__':
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

    n_experiment = 100
    horizon = 180
    delay = 30
    rewards_per_exp = []
    regrets_per_exp=[]
    numberOfDailyVisit =150
    matchingBestDiscountCode.updateActivationProb_weights_optimal(w * pages)
    matchingBestDiscountCode.updateActivationProb_returnerWeights_optimal(returnerWeights * pages)
    matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = Environment(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs,quantities,quantities_std)
        learner = Learner_M0_M(ucb)
        weightsLearn=weightsLearner()
        possibleReturnersAtTimeT = []
        instantRegret = []
        instantReward=[]
        returnerWeightsEstimated = [weightsLearn.returnWeights() * (pages > 0) for i in range(5)]


        for t in range(horizon):
            # print(t)
            margin=-1
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
                    optimalDiscountedItem = matchingBestDiscountCode.matcher_opt( M[u.firstLandingItem],
                                                                                M0[u.firstLandingItem],u, w * pages,
                                                                        returnerWeights * pages,quantities)

                    oldDiscountedItem=int(copy.deepcopy(u.discountedItem))
                    margin = returningVisit(u)
                    dailyMargins.append(margin)



                    if int(oldDiscountedItem)!=int(optimalDiscountedItem):
                        u.discountedItem = optimalDiscountedItem
                        landingProduct = env.returningLandingProduct(u)
                        optimalMargin = env.userVisits(u, landingProduct)
                        dailyOptimalMargins.append(optimalMargin)
                    else:

                        dailyOptimalMargins.append(margin)

                    del u,oldDiscountedItem,optimalDiscountedItem

                # if first visit just compute the margin
                else:
                    margin = env.userVisits(u,u.firstLandingItem)

                    if margin >0:
                        dailyMargins.append(margin)
                        dailyOptimalMargins.append(margin)
                        possibleReturningUser.append(u)
                        returnerWeightsEstimated = [weightsLearn.returnWeights() * (pages > 0) for i in range(5)]
                        u.discountedItem = matchingBestDiscountCode.matcher(learner.pull_arm(u.firstLandingItem),
                                                                        learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                returnerWeightsEstimated,quantities)


            matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeightsEstimated)
            possibleReturnersAtTimeT.append(possibleReturningUser)
            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))
            instantReward.append(math.fsum(dailyMargins))

            del userVisitingToday,margin,dailyMargins,dailyOptimalMargins
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegret = np.cumsum(instantRegret)
        regrets_per_exp.append(cumRegret)
        rewards_per_exp.append(instantReward)
        del instantRegret, possibleReturnersAtTimeT, learner, env
        gc.collect()
    s = str('UCB') if ucb else str('TS')
    mean = np.mean(regrets_per_exp, axis=0)
    std = np.std(regrets_per_exp, axis=0) / np.sqrt(n_experiment)
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.savefig('regret_step4_' + s + '.png')
    plt.show()

    plt.figure(1)
    mean = np.mean(rewards_per_exp, axis=0)
    std = np.std(rewards_per_exp, axis=0) / np.sqrt(n_experiment)
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.savefig('rewards_step4_' + s + '.png')
    plt.show()




