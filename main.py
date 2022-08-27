# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math

from Environment import Environment
from Products import Products
from Products import buyableProducts
from User import User
# given a user and the page from which the user will start to navigate our website,return the reward
# check if product is diff from P0
# if he didn't buy anything don't call method finalizePurchase and don't assign the discount
# add the user to the possible returner list
# set user.returner=True
import numpy as np
def userVisits(user,product):
    if product is not Products.P0:
        env.shoppingItems(product,user,env.userWeights(user),[])
        if len(user.cart)>0:
            dailyRewards.append(env.finalizePurchase(user))
        #     TODO: to decide to give discount or not
        user.returner=True
        possibleReturningUser.append(user)
    return

# update the distribution in the bandit
# decide if the user will return
def returnerVisit(user):
    product=env.userWillReturn(user)
    if product is not None:
        userVisits(user,product)
    return
if __name__ == '__main__':
    # TODO: defines parameters
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
    weights = {Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
               Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
               Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
               Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
               Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
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
          Products.P2: {Products.P1: 0.05, Products.P2: 0.05, Products.P3: 0.05, Products.P4: 0.05, Products.P5: 0.05},
          Products.P3: {Products.P1: 0.05, Products.P2: 0.05, Products.P3: 0.05, Products.P4: 0.05, Products.P5: 0.05},
          Products.P4: {Products.P1: 0.05, Products.P2: 0.05, Products.P3: 0.05, Products.P4: 0.05, Products.P5: 0.05},
          Products.P5: {Products.P1: 0.05, Products.P2: 0.05, Products.P3:0.05, Products.P4:0.05, Products.P5: 0.05},
          }
    # initialization of the 5 fixed webpages
    pages = {Products.P1: [Products.P1, Products.P2, Products.P3],
             Products.P2: [Products.P2, Products.P2, Products.P3],
             Products.P3: [Products.P3, Products.P4, Products.P5],
             Products.P4: [Products.P4, Products.P3, Products.P5],
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

    #     generation of the visits
    #     cycle over a time horizon
    # save the user at time t/\
    # generate the user new
    # add the returner users
    # shuffle them
    # make them visit each time
    # sum the rewards and save them
    horizon=10
    delay=2
    rewards=[]
    numberOfDailyVisit=10
    # user that visited our website at time t
    # pair <list(users)>
    possibleReturnersAtTimeT=[]
    for t in range(horizon):
        userVisitingToday=[]
        dailyRewards=[]
        possibleReturningUser=[]
        # randomNumberNewVisits=int(np.random.normal(numberOfDailyVisit,numberOfDailyVisit/4))
        randomNumberNewVisits=2

        for i in range(randomNumberNewVisits):
            userVisitingToday.append(env.generateUser())


        if t-delay>=0:
            print('returners')
            returnerUsers=possibleReturnersAtTimeT[t - delay]
            userVisitingToday=userVisitingToday+returnerUsers

        print(userVisitingToday)
        np.random.shuffle(userVisitingToday)

        for u in userVisitingToday:
            if u.returner:
                returnerVisit(u)
            else :
                userVisits(u,u.firstLandingItem)
        print(len(possibleReturningUser))
        possibleReturnersAtTimeT.append(possibleReturningUser)
        rewards.append(math.fsum(dailyRewards))
    print(rewards)

