# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
from Step2_maximization.matchingBestDiscountCode import matchingBestDiscountCode
from Step1_Environment.Environment import Environment

# given a user and the page from which the user will start to navigate our website,return the reward
# check if product is diff from P0
# if he didn't buy anything don't call method finalizePurchase and don't assign the discount
# add the user to the possible returner list
# set user.returner=True
import numpy as np


if __name__ == '__main__':
    # TODO: defines parameters, two different graph weights
    # initialization of the prices and costs
    prices = np.array([2,3,4,5,7])
    costs = np.array([0,0,0,0,0])

    # initialization of the alpha values
    alphas = np.array([0.2,0.2,0.2,0.1,0.2,0.1])



    # initialization of the weights
    w = np.array([[0.6, 0.5, 1, 0, 0],
                  [0, 0, 0.6, 0, 0.4],
                  [0, 0, 0, 0.5, 0.3],
                  [0.4, 0, 0, 0, 0.2],
                  [0.4, 0, 1, 0, 0]])


    # initialization of the matrix M
    M = np.array([[0.4, 0.5, 1, 0, 0],
                  [0, 0, 0.6, 0, 0.4],
                  [0, 0, 0, 0.5, 0.3],
                  [0.4, 0, 0, 0, 0.2],
                  [0.4, 0, 1, 0, 0]])
    # initialization of the matrix M0
    M0 = np.array([[0.1, 0.05, 0.1, 0, 0],
                  [0, 0, 0.06, 0, 0.04],
                  [0, 0, 0, 0.05, 0.03],
                  [0.04, 0, 0, 0, 0.02],
                  [0.04, 0, 0.01, 0, 0]])
    # initialization of the 5 fixed webpages
    pages = np.array([[0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 1, 0, 0]])

    # initialization of the weight for each class defined by discounted product
    returnerWeights =np.array([[[0, 0.5, 1, 0, 0],
                                [0, 0, 0.6, 0, 0.4],
                                [0, 0, 0, 0.5, 0.3],
                                 [0.4, 0, 0, 0, 0.2],
                                 [0.4, 0, 1, 0, 0]],
                              [[0, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]],
                              [[0.7, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]],
                              [[0, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]],
                              [[0.6, 0.5, 1, 0, 0],
                               [0, 0, 0.6, 0, 0.4],
                               [0, 0, 0, 0.5, 0.3],
                               [0.4, 0, 0, 0, 0.2],
                               [0.4, 0, 1, 0, 0]]]
                              )
    theta=0.8
    # initialization of the environment

    env = Environment(alphas, w*pages, returnerWeights*pages, M, M0, theta, prices, costs, 3)
    matchingBestDiscountCode=matchingBestDiscountCode(theta, pages, prices, costs,100)

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

                print(u.episode)
            # if the user actually navigated our website
            # add it to possible returners and give the appropriate discount
            if margin>=0:
                if margin >0:
                    u.discountedItem=matchingBestDiscountCode.matcher(w,returnerWeights,M[u.firstLandingItem],M0[u.firstLandingItem],u)
                    print(u.discountedItem)
                    dailyMargins.append(margin)

        possibleReturnersAtTimeT.append(possibleReturningUser)
        margins.append(math.fsum(dailyMargins))
    print(margins)

