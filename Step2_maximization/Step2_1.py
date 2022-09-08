# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
from Step2_maximization.matchingBestDiscountCode import matchingBestDiscountCode
from Step1_Environment.Environment import Environment
from Step1_Environment.Products import Products
# given a user and the page from which the user will start to navigate our website,return the reward
# check if product is diff from P0
# if he didn't buy anything don't call method finalizePurchase and don't assign the discount
# add the user to the possible returner list
# set user.returner=True
import numpy as np


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

    # initialization of the weights
    weights = {Products.P1: {Products.P1: 0, Products.P2: 0.8, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
         Products.P2: {Products.P1: 0.65, Products.P2: 0, Products.P3: 0.8, Products.P4: 0.4, Products.P5: 0.2},
         Products.P3: {Products.P1: 0.7, Products.P2: 0.75, Products.P3: 0, Products.P4: 0.3, Products.P5: 0.4},
         Products.P4: {Products.P1: 0.3, Products.P2: 0.4, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.8},
         Products.P5: {Products.P1: 0.4, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.85, Products.P5: 0},
         }

    # initialization of the matrix M
    M = {Products.P1: {Products.P1: 0.7, Products.P2: 1, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.2},
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
        Products.P1: {Products.P1:0, Products.P2: 0.8, Products.P3: 0.5, Products.P4: 1, Products.P5: 1},
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
    matchingBestDiscountCode=matchingBestDiscountCode(theta, pages, prices, costs)

    horizon=10
    delay=2
    margins=[]
    numberOfDailyVisit=10
    # user that visited our website at time t
    # <list(users)>
    possibleReturnersAtTimeT=[]
    # TODO: difference between clairvoyant solution and our solution
    for t in range(horizon):

        userVisitingToday=[]
        dailyMargins=[0]
        possibleReturningUser=[]
        randomNumberNewVisits=int(np.random.normal(numberOfDailyVisit,numberOfDailyVisit/4))
        # generate a random number of new users
        for i in range(randomNumberNewVisits):
            userVisitingToday.append(env.generateUser())

        # get the possible returning user from t-delay time
        if t-delay>=0:
            returnerUsers=possibleReturnersAtTimeT.pop(0)
            userVisitingToday=userVisitingToday+returnerUsers


        np.random.shuffle(userVisitingToday)

        for u in userVisitingToday:

            if u.returner:
                landingProduct = env.returningLandingProduct(u)
                margin = env.userVisits(u, landingProduct)

            # if first visit just compute the margin
            else:
                margin = env.userVisits(u,u.firstLandingItem)
                if margin is not None:
                    possibleReturningUser.append(u)
            # if the user actually navigated our website
            # add it to possible returners and give the appropriate discount
            if margin is not None:
                if margin >0:
                    u.discountedItem=matchingBestDiscountCode.matcher(weights,returnerWeights,M[u.firstLandingItem],M0[u.firstLandingItem],u)

                dailyMargins.append(margin)

        possibleReturnersAtTimeT.append(possibleReturningUser)
        margins.append(math.fsum(dailyMargins))
    print(margins)

