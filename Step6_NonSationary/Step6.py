
import copy
import math
import gc
from Step2_maximization.heuristic import matchingBestDiscountCode
from Step6_NonSationary.NonStationaryBandit.NonStationaryEnv import NonStationaryEnv
import json
from Step6_NonSationary.NonStationaryBandit.NonSationary_Learner_M0_M import NonStationary_Learner_M_M0
import matplotlib.pyplot as plt
import numpy as np

def returningVisit(user1):
    current_phase = min(1,int((env.t-delay)/ env.phase_size))
    landingProduct = env.returningLandingProductN(user1,current_phase)
    reward= 1 if landingProduct < 5 else 0
    learner.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
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

    # define the experiment parameters
    n_experiment = 5
    horizon = 365
    delay = 30
    numberOfDailyVisit = 150
    # init the matcher
    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,100)
    # if window 0 use change detection otherwise use sliding learner
    window=0

    regrets_per_exp = []
    rewards_per_exp = []

    # init matcher activation prob
    matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights * pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = NonStationaryEnv(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs,quantities,quantities_std,280)
        learner = NonStationary_Learner_M_M0(window)

        possibleReturnersAtTimeT = []
        instantRegret = []
        instantReward=[]

        for t in range(horizon):
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
                    phase=min(1,int((t-delay) / env.phase_size))
                    optimalDiscountedItem = matchingBestDiscountCode.matcher( M[phase][u.firstLandingItem],
                                                                                M0[phase][u.firstLandingItem],u, w * pages,
                                                                        returnerWeights * pages,quantities)

                    oldDiscountedItem=int(copy.deepcopy(u.discountedItem))
                    margin = returningVisit(u)

                    dailyMargins.append(margin)



                    if int(oldDiscountedItem)!=int(optimalDiscountedItem):
                        u.discountedItem = optimalDiscountedItem
                        landingProduct = env.returningLandingProductN(u,phase)
                        optimalMargin = env.userVisits(u, landingProduct)
                        dailyOptimalMargins.append(optimalMargin)
                    else:

                        dailyOptimalMargins.append(margin)

                    del u,oldDiscountedItem,optimalDiscountedItem,phase

                # if first visit just compute the margin
                else:
                    margin = env.userVisits(u,u.firstLandingItem)

                    if margin >0:
                        dailyMargins.append(margin)
                        dailyOptimalMargins.append(margin)
                        possibleReturningUser.append(u)
                        u.discountedItem = matchingBestDiscountCode.matcher(learner.pull_arm(u.firstLandingItem),
                                                                            learner.pull_arm_M0(u.firstLandingItem), u, w * pages,
                                                                            returnerWeights * pages, quantities)

            possibleReturnersAtTimeT.append(possibleReturningUser)
            env.updateT()
            learner.time()
            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))
            instantReward.append(math.fsum(dailyMargins))
            del userVisitingToday,margin,dailyOptimalMargins,dailyMargins
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegret = np.cumsum(instantRegret)
        rewards_per_exp.append(instantReward)
        regrets_per_exp.append(cumRegret)
        del instantRegret, possibleReturnersAtTimeT, learner, env
        gc.collect()


    s=str('CD') if window==0 else str('SL')
    mean = np.mean(regrets_per_exp, axis=0)
    std = np.std(regrets_per_exp, axis=0) / np.sqrt(n_experiment)
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.savefig('regret_step6_'+s+'.png')
    plt.show()

    plt.figure(1)
    mean = np.mean(rewards_per_exp, axis=0)
    std = np.std(rewards_per_exp, axis=0) / np.sqrt(n_experiment)
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.savefig('rewards_step6_'+s+'.png')
    plt.clf()





