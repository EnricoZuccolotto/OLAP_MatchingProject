
import copy
import math
import gc
from Step2_maximization.matchingBestDiscountCode import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from Bandit.Learner_M0_M import Learner_M0_M
import matplotlib.pyplot as plt
import numpy as np
import json

def printProb():
    if ucb:
        for p in range(5):
            print(learner.pull_arm(p))
            print([len(x) for x in learner.learners[p].rewards_per_arm])
            print(learner.pull_arm_M0(p))
            print([len(x) for x in learner.learners[p].rewards_per_product])

    else:
        for p in range(5):
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
    reward = 1 if landingProduct < 5 else 0
    learner.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m
if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
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

    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,1000)
    ucb = 1

    n_experiment = 100
    horizon = 180
    delay = 30
    regrets_per_exp = []
    rewards_per_exp = []
    numberOfDailyVisit =150
    matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights * pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = Environment(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs, quantities,quantities_std)
        learner = Learner_M0_M(ucb)
        possibleReturnersAtTimeT = []
        instantRegret = []
        instantReward=[]

        for t in range(horizon):
            print(t)
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
                    optimalDiscountedItem = matchingBestDiscountCode.matcher( M[u.firstLandingItem],
                                                                                M0[u.firstLandingItem],u, w * pages,
                                                                        returnerWeights * pages,quantities)

                    oldDiscountedItem=int(copy.deepcopy(u.discountedItem))
                    margin = returningVisit(u)
                    dailyMargins.append(margin)



                    if int(oldDiscountedItem)!=int(optimalDiscountedItem):
                        u.discountedItem = optimalDiscountedItem
                        landingProduct = env.returningLandingProduct(u)
                        optimalMargin= env.userVisits(u, landingProduct)
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
                        u.discountedItem = matchingBestDiscountCode.matcher(learner.pull_arm(u.firstLandingItem),
                                                                        learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                                returnerWeights * pages,quantities)

            possibleReturnersAtTimeT.append(possibleReturningUser)

            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))
            instantReward.append(math.fsum(dailyMargins))
            del userVisitingToday,margin,dailyOptimalMargins,dailyMargins
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegret = np.cumsum(instantRegret)
        printProb()
        rewards_per_exp.append(instantReward)
        regrets_per_exp.append(cumRegret)
        mean = np.mean(regrets_per_exp, axis=0)
        std = np.std(regrets_per_exp, axis=0) / np.sqrt(e + 1)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("regret")
        plt.plot(mean)
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
        plt.savefig('fo'+str(e)+'.png')
        plt.clf()

        plt.figure(1)
        mean = np.mean(rewards_per_exp, axis=0)
        std = np.std(rewards_per_exp, axis=0) / np.sqrt(e + 1)
        plt.xlabel("t")
        plt.ylabel("reward")
        plt.plot(mean)
        plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
        plt.savefig('rewardso' + str(e) + '.png')
        plt.clf()


        del instantRegret, possibleReturnersAtTimeT, learner, mean, std, env
        gc.collect()


