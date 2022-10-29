
import gc
from Step2_maximization.heuristic import matchingBestDiscountCode
from Step2_maximization.matcher import matchingBestDiscountCode as matcher
from Step1_Environment.Environment import *
import matplotlib.pyplot as plt
import numpy as np
import json


def returningVisit(user1):
    landingProduct = env.returningLandingProduct(user1)
    m = env.userVisits(user1, landingProduct)
    return m
if __name__ == '__main__':
     # initialization of the prices and costs
    f=open("Step1_Environment\config1.json")
    config=json.loads(f.read())
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
    returnerWeights =np.array(config["returnerWeights"])

    # initialization of the 5 fixed webpages
    pages = np.array(config["pages"])

    quantities=np.array(config["quantities"])
    quantities_std=np.array(config["quantities_std"])
    f.close()
    # initialization of the environment

    matchingBestDiscountCode=matchingBestDiscountCode( prices, costs,1000)
    matcher=matcher(prices, costs,1000)
    ucb = 1

    n_experiment = 5
    horizon = 180
    delay = 30
    rewards_per_exp = []
    exact_matches_exp=[]
    numberOfDailyVisit =150
    matchingBestDiscountCode.updateActivationProb_weights(w * pages)
    matchingBestDiscountCode.updateActivationProb_returnerWeights(returnerWeights * pages)

    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp ' + str(e))
        env = Environment(alphas, w * pages, returnerWeights * pages, M, M0, prices, costs,quantities,quantities_std)
        possibleReturnersAtTimeT = []
        instantRegret = []
        exactMatch=[]


        for t in range(horizon):
            print(t)
            dailyMargins = [0]
            dailyOptimalMargins = [0]
            possibleReturningUser = []
            randomNumberNewVisits = max(0, int(np.random.normal(numberOfDailyVisit,15)))
            count=0
            tot=0
            userVisitingToday=env.generateRandomUser(randomNumberNewVisits)

            # get the possible returning user from t-delay time
            if t-delay>=0:
                returnerUsers=possibleReturnersAtTimeT.pop(0)
                userVisitingToday=userVisitingToday+returnerUsers


            np.random.shuffle(userVisitingToday)

            for u in userVisitingToday:

                if u.returner:
                    optimalDiscountedItem = matcher.matcher( M[u.firstLandingItem],M0[u.firstLandingItem],u, w * pages,returnerWeights * pages,quantities)
                    tot+=1
                    oldDiscountedItem=int(copy.deepcopy(u.discountedItem))
                    margin = returningVisit(u)
                    dailyMargins.append(margin)



                    if int(oldDiscountedItem)!=int(optimalDiscountedItem):
                        u.discountedItem = optimalDiscountedItem
                        optimalMargin = returningVisit(u)
                        dailyOptimalMargins.append(optimalMargin)
                    else:
                        count+=1
                        dailyOptimalMargins.append(margin)

                    del u,oldDiscountedItem,optimalDiscountedItem

                # if first visit just compute the margin
                else:
                    margin = env.userVisits(u,u.firstLandingItem)
                    if margin >0:
                        possibleReturningUser.append(u)
                        u.discountedItem = matchingBestDiscountCode.matcher(M[u.firstLandingItem],
                                                                                M0[u.firstLandingItem], u, w * pages,
                                                                                returnerWeights * pages,quantities)

            possibleReturnersAtTimeT.append(possibleReturningUser)
            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))
            if tot>0:
                exactMatch.append(count/tot)
            del userVisitingToday,margin,dailyMargins,dailyOptimalMargins
            if t - delay >= 0:
                del returnerUsers
            gc.collect()



        cumRegret = np.cumsum(instantRegret)
        rewards_per_exp.append(cumRegret)
        exact_matches_exp.append(exactMatch)
        del instantRegret, possibleReturnersAtTimeT, env
        gc.collect()
    mean = np.mean(rewards_per_exp, axis=0)
    std = np.std(rewards_per_exp, axis=0) / np.sqrt(n_experiment)
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.savefig('regret_step2.png')
    plt.show()

    mean = np.mean(exact_matches_exp, axis=0)
    std = np.std(exact_matches_exp, axis=0) / np.sqrt(n_experiment)
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("exact_matches")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.savefig('matches_step2.png')
    plt.show()




