# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
from Step2_maximization.monteCarlo import monteCarlo
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

    weights = {Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
               Products.P2: {Products.P1:0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
               Products.P3: {Products.P1: 0.8, Products.P2: 0.3, Products.P3: 0, Products.P4: 0.4, Products.P5: 0.15},
               Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
               Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
               }
    # initialization of the matrix M
    M = {Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
         Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
         Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
         Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
         Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
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
    # initialization of the environment
    env = Environment(alphas, weights, returnerWeights, M, M0, 0.8, prices, costs, pages, 3)
    monteCarlo=monteCarlo(0.8,pages,prices,costs)

    horizon=1
    delay=2
    margins=[]
    numberOfDailyVisit=10
    # user that visited our website at time t
    # <list(users)>
    possibleReturnersAtTimeT=[]
    # TODO: difference between clairvoyant solution and our solution
    for t in range(horizon):

        userVisitingToday=[]
        dailyMargins=[]
        possibleReturningUser=[]
        randomNumberNewVisits=int(np.random.normal(numberOfDailyVisit,numberOfDailyVisit/4))
        # randomNumberNewVisits=1
        # generate a random number of new users
        for i in range(randomNumberNewVisits):
            userVisitingToday.append(env.generateUser())

        # get the possible returning user from t-delay time
        if t-delay>=0:
            returnerUsers=possibleReturnersAtTimeT.pop(0)
            userVisitingToday=userVisitingToday+returnerUsers


        np.random.shuffle(userVisitingToday)

        for u in userVisitingToday:
            # if he is a possible returner compute the landing product
            # reward 1 if he returned 0 otherwise
            # update the distribution of M and M0
            # compute the margin and apply the discount
            if u.returner:
                landingProduct = env.returningLandingProduct(u)
                reward = env.round(landingProduct)
                # learner.update(u.firstLandingItem,u.discountedItem,landingProduct, reward)
                margin = env.userVisits(u, landingProduct)

            # if first visit just compute the margin
            else:
                margin = env.userVisits(u,u.firstLandingItem)

            # if the user actually navigated our website
            # add it to possible returners and give the appropriate discount
            if margin is not None:
                # print(env.returnEpisode())
                print('margin'+"   "+str(margin))
                print(u.episode)
                # TODO: what to do in case the user actually didn't use the discount
                if margin >0:
                    # u.discountedItem = learner.pull_arm()
                    print(monteCarlo.monteCarloRuns(100, Products.P1, env.weights, u.productsSeen))
                possibleReturningUser.append(u)
                dailyMargins.append(margin)

        possibleReturnersAtTimeT.append(possibleReturningUser)
        margins.append(math.fsum(dailyMargins))
    print(margins)

