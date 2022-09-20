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
    prices = {Products.P1: 13,
              Products.P2: 10,
              Products.P3: 17,
              Products.P4: 3,
              Products.P5: 7}
    costs = {Products.P1: 5,
             Products.P2: 3,
             Products.P3: 7,
             Products.P4: 0.5,
             Products.P5: 2}

    # initialization of the alpha values
    alphas = {Products.P1: 0.2,
              Products.P2: 0.2,
              Products.P3: 0.2,
              Products.P4: 0.2,
              Products.P5: 0.1,
              Products.P0: 0.1}

    # initialization of the weights

    # initialization of the weights
    weights = {Products.P1: {Products.P1: 0.7, Products.P2: 0.67, Products.P3: 0.5, Products.P4: 0.35, Products.P5: 0.55},
         Products.P2: {Products.P1: 0.69, Products.P2: 0.54, Products.P3: 0.55, Products.P4: 0.59, Products.P5:0.55},
         Products.P3: {Products.P1: 0.75, Products.P2: 0.5, Products.P3: 0.58, Products.P4: 0.52, Products.P5: 0.64},
         Products.P4: {Products.P1: 0.7, Products.P2: 0.6, Products.P3: 0.62, Products.P4: 0.53, Products.P5: 0.63},
         Products.P5: {Products.P1: 0.73, Products.P2: 0.55, Products.P3: 0.65, Products.P4: 0.57, Products.P5: 0.59},
         }

    # initialization of the matrix M
    M = {Products.P1: {Products.P1: 0.3, Products.P2: 0.4, Products.P3: 0.38, Products.P4: 0.45, Products.P5: 0.2},
         Products.P2: {Products.P1: 0.2, Products.P2: 0.5, Products.P3: 0.34, Products.P4: 0.4, Products.P5: 0.1},
         Products.P3: {Products.P1: 0.25, Products.P2: 0.31, Products.P3: 0.15, Products.P4: 0.28, Products.P5: 0.44},
         Products.P4: {Products.P1: 0.19, Products.P2: 0.28, Products.P3: 0.42, Products.P4: 0.17, Products.P5: 0.35},
         Products.P5: {Products.P1: 0.35, Products.P2: 0.25, Products.P3: 0.29, Products.P4: 0.5, Products.P5: 0.47},
         }
    # initialization of the matrix M0
    M0 = {Products.P1: {Products.P1: 0.01, Products.P2: 0.05, Products.P3: 0.02, Products.P4: 0.03, Products.P5: 0.025},
          Products.P2: {Products.P1: 0.02, Products.P2: 0.025, Products.P3: 0.015, Products.P4: 0.05, Products.P5: 0.02},
          Products.P3: {Products.P1: 0.03, Products.P2: 0.01, Products.P3: 0.04, Products.P4: 0.05, Products.P5: 0.038},
          Products.P4: {Products.P1: 0.01, Products.P2: 0.017, Products.P3: 0.02, Products.P4: 0.015, Products.P5: 0.05},
          Products.P5: {Products.P1: 0.03, Products.P2: 0.02, Products.P3: 0.05, Products.P4: 0.035, Products.P5: 0.01},
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
        Products.P1: {Products.P1:0, Products.P2: 0.65, Products.P3: 0.5, Products.P4: 0.4, Products.P5: 0.25},
        Products.P2: {Products.P1: 0.6, Products.P2: 0, Products.P3: 0.35, Products.P4: 0.2, Products.P5: 0.5},
        Products.P3: {Products.P1: 0.48, Products.P2: 0.50, Products.P3: 0, Products.P4: 0.3, Products.P5: 0.4},
        Products.P4: {Products.P1: 0.49, Products.P2: 0.6, Products.P3: 0.45, Products.P4: 0, Products.P5: 0.55},
        Products.P5: {Products.P1: 0.35, Products.P2: 0.45, Products.P3: 0.55, Products.P4: 0.65, Products.P5: 0},
    }
    returnerWeights[Products.P2] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.45, Products.P3: 0.48, Products.P4: 0.35, Products.P5: 0.5},
        Products.P2: {Products.P1: 0.55, Products.P2: 0, Products.P3: 0.45, Products.P4: 0.63, Products.P5: 0.38},
        Products.P3: {Products.P1: 0.59, Products.P2: 0.47, Products.P3: 0, Products.P4: 0.58, Products.P5: 0.61},
        Products.P4: {Products.P1: 0.41, Products.P2: 0.45, Products.P3: 0.6, Products.P4: 0, Products.P5: 0.5},
        Products.P5: {Products.P1: 0.45, Products.P2: 0.39, Products.P3: 0.47, Products.P4: 0.55, Products.P5: 0},
    }
    returnerWeights[Products.P3] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.43, Products.P3: 0.58, Products.P4: 0.55, Products.P5:0.39},
        Products.P2: {Products.P1: 0.42, Products.P2: 0, Products.P3: 0.57, Products.P4: 0.61, Products.P5: 0.6},
        Products.P3: {Products.P1: 0.47, Products.P2: 0.53, Products.P3: 0, Products.P4: 0.58, Products.P5: 0.62},
        Products.P4: {Products.P1: 0.41, Products.P2: 0.49, Products.P3: 0.58, Products.P4: 0, Products.P5: 0.6},
        Products.P5: {Products.P1: 0.45, Products.P2: 0.5, Products.P3: 0.35, Products.P4: 0.6, Products.P5: 0},
    }
    returnerWeights[Products.P4] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.4, Products.P4: 0.37, Products.P5: 0.61},
        Products.P2: {Products.P1: 0.39, Products.P2: 0, Products.P3: 0.47, Products.P4: 0.43, Products.P5: 0.53},
        Products.P3: {Products.P1: 0.65, Products.P2: 0.55, Products.P3: 0, Products.P4: 0.47, Products.P5: 0.6},
        Products.P4: {Products.P1: 0.5, Products.P2: 0.37, Products.P3: 0.47, Products.P4: 0, Products.P5: 0.44},
        Products.P5: {Products.P1: 0.51, Products.P2: 0.57, Products.P3: 0.60, Products.P4: 0.63, Products.P5: 0},
    }
    returnerWeights[Products.P5] = {
        Products.P1: {Products.P1: 0, Products.P2: 0.45, Products.P3: 0.42, Products.P4: 0.37, Products.P5: 0.58},
        Products.P2: {Products.P1: 0.57, Products.P2: 0, Products.P3: 0.5, Products.P4: 0.6, Products.P5: 0.57},
        Products.P3: {Products.P1: 0.38, Products.P2: 0.45, Products.P3: 0, Products.P4: 0.47, Products.P5: 0.50},
        Products.P4: {Products.P1: 0.52, Products.P2: 0.39, Products.P3: 0.49, Products.P4: 0, Products.P5: 0.5},
        Products.P5: {Products.P1: 0.65, Products.P2: 0.54, Products.P3: 0.45, Products.P4: 0.59, Products.P5: 0},
    }
    theta=0.8
    # initialization of the environment
    env = Environment(alphas, weights, returnerWeights, M, M0, theta, prices, costs, pages, 3)
    matchingBestDiscountCode=matchingBestDiscountCode(theta, pages, prices, costs)

    horizon=10
    delay=2
    margins=[]
    numberOfDailyVisit=100
    # user that visited our website at time t
    # <list(users)>
    possibleReturnersAtTimeT=[]
    # TODO: difference between clairvoyant solution and our solution
    for t in range(horizon):

        dailyMargins=[0]
        possibleReturningUser=[]
        randomNumberNewVisits=int(np.random.normal(numberOfDailyVisit,numberOfDailyVisit/4))
        # generate a random number of new users

        userVisitingToday=env.generateRandomUser(randomNumberNewVisits)

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
                possibleReturningUser.append(u)
            # if the user actually navigated our website
            # add it to possible returners and give the appropriate discount
            if margin>=0:
                if margin >0:
                    u.discountedItem=matchingBestDiscountCode.matcher(weights,returnerWeights,M[u.firstLandingItem],M0[u.firstLandingItem],u)
                    dailyMargins.append(margin)

        possibleReturnersAtTimeT.append(possibleReturningUser)
        margins.append(math.fsum(dailyMargins))
    print(margins)

