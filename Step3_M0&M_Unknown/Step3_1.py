# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
from Bandit.Learner_M0_M import Learner_M0_M
from Step2_maximization.matchingBestDiscountCode import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from Step1_Environment.User import User
from Step1_Environment.Products import Products
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
    margin = env.userVisits(user1, landingProduct)
    return margin
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
               Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
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
    M0 = {Products.P1: {Products.P1: 0.1, Products.P2: 0.05, Products.P3: 0.2, Products.P4: 0.05, Products.P5: 0.05},
          Products.P2: {Products.P1: 0.2, Products.P2: 0.25, Products.P3: 0.15, Products.P4: 0.05, Products.P5: 0.02},
          Products.P3: {Products.P1: 0.03, Products.P2: 0.1, Products.P3: 0.07, Products.P4: 0.05, Products.P5: 0.05},
          Products.P4: {Products.P1: 0.01, Products.P2: 0.01, Products.P3: 0, Products.P4: 0.15, Products.P5: 0.05},
          Products.P5: {Products.P1: 0.0, Products.P2: 0.0, Products.P3: 0.05, Products.P4: 0.03, Products.P5: 0.02},
          }
    # initialization of the 5 fixed webpages
    pages = {Products.P1: [Products.P1, Products.P2, Products.P3],
             Products.P2: [Products.P2, Products.P3, Products.P5],
             Products.P3: [Products.P3, Products.P4, Products.P5],
             Products.P4: [Products.P4, Products.P5, Products.P1],
             Products.P5: [Products.P5, Products.P1, Products.P3], }

    # initialization of the weight for each class defined by discounted product
    returnerWeights = {}
    returnerWeights[Products.P1] = {
        Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
        Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
        Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
    }
    returnerWeights[Products.P2] = {
        Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
        Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
        Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
    }
    returnerWeights[Products.P3] = {
        Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
        Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
        Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
    }
    returnerWeights[Products.P4] = {
        Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
        Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
        Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
    }
    returnerWeights[Products.P5] = {
        Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
        Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
        Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
        Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
    }
    theta=0.8
    # initialization of the environment
    env = Environment(alphas, weights, returnerWeights, M, M0, theta, prices, costs, pages, 3)
    matchingBestDiscountCode = matchingBestDiscountCode(theta, pages, prices, costs)
    ucb=1
    learner= Learner_M0_M(ucb)
    horizon = 100
    delay = 30
    margins = []
    optimalMargins=[]
    numberOfDailyVisit = 100
    # user that visited our website at time t
    # <list(users)>
    possibleReturnersAtTimeT = []
    # TODO: difference between clairvoyant solution and our solution
    for t in range(horizon):
        print(t)
        userVisitingToday = []
        dailyMargins = [0]
        dailyOptimalMargins = [0]
        possibleReturningUser = []
        randomNumberNewVisits = int(np.random.normal(numberOfDailyVisit, numberOfDailyVisit / 4))

        # generate a random number of new users
        for i in range(randomNumberNewVisits):
            userVisitingToday.append(env.generateUser())

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
                discountedItem=u.discountedItem
                margin=returningVisit(u)
                dailyMargins.append(margin)
                optimalDiscountedItem=matchingBestDiscountCode.matcher(weights, returnerWeights,M[u.firstLandingItem],M0[u.firstLandingItem], u)
                if optimalDiscountedItem is not discountedItem:
                    user2=User(u.reservationPrice,u.firstLandingItem)
                    user2.discountedItem=optimalDiscountedItem
                    user2.probabilityFutureBehaviour=u.probabilityFutureBehaviour
                    optimalMargin=returningVisit(user2)
                    if optimalMargin is not None:
                        dailyOptimalMargins.append(optimalMargin)
                else:
                    if margin is not None:
                        dailyOptimalMargins.append(margin)
            # if first visit just compute the margin
            # if the user actually navigated our website
            # add it to possible returners and give the appropriate discount

            else:
                margin = env.userVisits(u, u.firstLandingItem)
                if margin is not None:
                    dailyOptimalMargins.append(margin)
                    dailyMargins.append(margin)
                    if margin > 0:
                        u.discountedItem = matchingBestDiscountCode.matcher(weights, returnerWeights,
                                                                            learner.pull_arm(u.firstLandingItem),
                                                                            learner.pull_arm_M0(u.firstLandingItem), u)
                    possibleReturningUser.append(u)

        possibleReturnersAtTimeT.append(possibleReturningUser)
        margins.append(math.fsum(dailyMargins))
        optimalMargins.append(math.fsum(dailyOptimalMargins))

    printProb()
    print(margins)
    print(optimalMargins)

