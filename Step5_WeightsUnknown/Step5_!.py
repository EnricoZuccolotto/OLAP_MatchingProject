# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import copy
import math
from Bandit.Learner_M0_M import Learner_M0_M
from Step2_maximization.matchingBestDiscountCode import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from CreditAssignment.weightsLearner import weightsLearner
from Step1_Environment.Products import Products
import matplotlib.pyplot as plt
# given a user and the page from which the user will start to navigate our website,return the reward
# check if product is diff from P0
# if he didn't buy anything don't call method finalizePurchase and don't assign the discount
# add the user to the possible returner list
# set user.returner=True
import numpy as np

def printProb():
    if ucb:
        for p in list(Products):
            if p is not Products.P0:
                print(p)
                print(learner.learners[p].means)
                print(learner.learners[p].M0)
    else:
        for p in list(Products):
            if p is not Products.P0:
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
    reward = env.round(landingProduct)
    learner.update(user1.firstLandingItem, user1.discountedItem, landingProduct, reward)
    m = env.userVisits(user1, landingProduct)
    return m
if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
    prices = {Products.P1: 7,
              Products.P2: 4,
              Products.P3: 5,
              Products.P4: 3,
              Products.P5: 2}
    costs = {Products.P1: 0,
             Products.P2: 0,
             Products.P3: 0,
             Products.P4: 0,
             Products.P5: 0}

    # initialization of the alpha values
    alphas = {Products.P1: 0.2,
              Products.P2: 0.2,
              Products.P3: 0.2,
              Products.P4: 0.2,
              Products.P5: 0.1,
              Products.P0: 0.1}

    # initialization of the weights
    weights = {Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
               Products.P2: {Products.P1:0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
               Products.P3: {Products.P1: 0.2, Products.P2: 0.4, Products.P3: 0, Products.P4: 0.1, Products.P5: 0.9},
               Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
               Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
               }
    # initialization of the matrix M
    M = {Products.P1: {Products.P1: 0.7, Products.P2: 1, Products.P3:0.3, Products.P4: 0.4, Products.P5: 0.2},
         Products.P2: {Products.P1: 1, Products.P2: 0.5, Products.P3: 0.6, Products.P4: 0.4, Products.P5: 0.1},
         Products.P3: {Products.P1: 0.8, Products.P2: 0.3, Products.P3: 0, Products.P4: 1, Products.P5: 0.9},
         Products.P4: {Products.P1: 0.1, Products.P2: 0.1, Products.P3: 1, Products.P4: 0, Products.P5: 0.4},
         Products.P5: {Products.P1: 0.6, Products.P2: 0.7, Products.P3: 0.8, Products.P4: 1, Products.P5: 0},
         }
    # initialization of the matrix M0
    M0 = {Products.P1: {Products.P1: 0.001, Products.P2: 0.005, Products.P3: 0.02, Products.P4: 0.005, Products.P5: 0.005},
          Products.P2: {Products.P1: 0.02, Products.P2: 0.025, Products.P3: 0.015, Products.P4: 0.005, Products.P5: 0.002},
          Products.P3: {Products.P1: 0.003, Products.P2: 0.01, Products.P3: 0.007, Products.P4: 0.005, Products.P5: 0.005},
          Products.P4: {Products.P1: 0.001, Products.P2: 0.001, Products.P3: 0, Products.P4: 0.015, Products.P5: 0.005},
          Products.P5: {Products.P1: 0.0, Products.P2: 0.0, Products.P3: 0.005, Products.P4: 0.003, Products.P5: 0.002},
          }
    # initialization of the 5 fixed webpages
    pages = {Products.P1: [Products.P1, Products.P2, Products.P3],
             Products.P2: [Products.P2, Products.P3, Products.P5],
             Products.P3: [Products.P3, Products.P4, Products.P5],
             Products.P4: [Products.P4, Products.P5, Products.P1],
             Products.P5: [Products.P5, Products.P1, Products.P3], }

    # initialization of the weight for each class defined by discounted product
    returnerWeights = {}
    returnerWeights[Products.P1] = {Products.P1: {Products.P1: 0, Products.P2: 0.65, Products.P3: 0.77, Products.P4: 0.32, Products.P5: 0.43},
               Products.P2: {Products.P1:0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
                 Products.P3: {Products.P1: 0.2, Products.P2: 0.4, Products.P3: 0, Products.P4: 0.1, Products.P5: 0.9},
               Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
               Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
               }
    returnerWeights[Products.P2] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
        Products.P2: {Products.P1: 0.64, Products.P2: 0, Products.P3: 0.43, Products.P4: 0.54, Products.P5: 0.88},
        Products.P3: {Products.P1: 0.2, Products.P2: 0.4, Products.P3: 0, Products.P4: 0.1, Products.P5: 0.9},
        Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
        Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
    }
    returnerWeights[Products.P3] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
        Products.P2: {Products.P1: 0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
        Products.P3: {Products.P1: 0.32, Products.P2: 0.54, Products.P3: 0, Products.P4: 0.24, Products.P5: 0.9},
        Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
        Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
    }
    returnerWeights[Products.P4] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
        Products.P2: {Products.P1: 0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
        Products.P3: {Products.P1: 0.32, Products.P2: 0.54, Products.P3: 0, Products.P4: 0.24, Products.P5: 0.9},
        Products.P4: {Products.P1: 0.75, Products.P2: 0.76, Products.P3: 0.54, Products.P4: 0, Products.P5: 0.53},
        Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
    }
    returnerWeights[Products.P5] = {Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
               Products.P2: {Products.P1:0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
               Products.P3: {Products.P1: 0.2, Products.P2: 0.4, Products.P3: 0, Products.P4: 0.1, Products.P5: 0.9},
               Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
               Products.P5: {Products.P1: 0.87, Products.P2: 0.43, Products.P3: 0.516, Products.P4: 0.436, Products.P5: 0},
               }
    theta=0.8
    # initialization of the environment
    env = Environment(alphas, weights, returnerWeights, M, M0, theta, prices, costs, pages, 3)
    matchingBestDiscountCode = matchingBestDiscountCode(theta, pages, prices, costs,50)
    ucb=1
    learner= Learner_M0_M(ucb)
    learnerWeights=weightsLearner()
    n_experiment= 100
    horizon = 365
    delay = 30
    rewards_per_exp=[]
    numberOfDailyVisit = 100
    # user that visited our website at time t
    # <list(users)>
    for e in range(n_experiment):
        print('exp '+str(e))
        possibleReturnersAtTimeT = []
        instantRegret = []
        # TODO: difference between clairvoyant solution and our solution
        for t in range(horizon):
            print(t)
            dailyMargins = [0]
            dailyOptimalMargins = [0]
            possibleReturningUser = []
            randomNumberNewVisits = max(0,int(np.random.normal(numberOfDailyVisit, numberOfDailyVisit / 4)))

            # generate a random number of new users
            userVisitingToday=env.generateRandomUser(randomNumberNewVisits)

            # get the possible returning user from t-delay time
            if t - delay >= 0:
                returnerUsers = possibleReturnersAtTimeT.pop(0)
                userVisitingToday = userVisitingToday + returnerUsers

            np.random.shuffle(userVisitingToday)

            for u in userVisitingToday:
                # if he is a possible returner compute the landing product
                # reward 1 if he returned 0 otherwise
                # update the distribution of M and M0
                # compute the margin and apply the discount
                if u.returner:

                    discountedItem=copy.deepcopy(u.discountedItem)
                    probFutureBehaviour=copy.deepcopy(u.probabilityFutureBehaviour)


                    margin=returningVisit(u)


                    dailyMargins.append(margin)


                    u.probabilityFutureBehaviour=probFutureBehaviour
                    u.discountedItem=matchingBestDiscountCode.matcher(weights, returnerWeights,M[u.firstLandingItem],M0[u.firstLandingItem], u)


                    if u.discountedItem is not discountedItem:
                        optimalMargin=returningVisit(u)
                        dailyOptimalMargins.append(optimalMargin)
                    else:
                        dailyOptimalMargins.append(margin)

                # if first visit just compute the margin
                # if the user actually navigated our website
                # add it to possible returners and give the appropriate discount

                else:
                    margin = env.userVisits(u, u.firstLandingItem)

                    learnerWeights.updateEstimates(u.episode)
                    if margin>=0:
                        if margin>0:
                            u.discountedItem = matchingBestDiscountCode.matcher( learnerWeights.returnWeights(),returnerWeights,
                                                                                learner.pull_arm(u.firstLandingItem),
                                                                                learner.pull_arm_M0(u.firstLandingItem), u)


                        possibleReturningUser.append(u)

            possibleReturnersAtTimeT.append(possibleReturningUser)
            instantRegret.append(math.fsum(dailyOptimalMargins) - math.fsum(dailyMargins))
        cumRegret = np.cumsum(instantRegret)
        rewards_per_exp.append(cumRegret)
    printProb()


    mean = np.mean(rewards_per_exp, axis=0)
    std = np.std(rewards_per_exp, axis=0) / np.sqrt(n_experiment)
    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.plot(mean)
    plt.fill_between(range(horizon), mean - std, mean + std, alpha=0.4)
    plt.show()

