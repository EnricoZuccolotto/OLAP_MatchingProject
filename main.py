# https://www.jetbrains.com/pycharm/
# Log in with your polimi email to have free access otherwise you need to pay
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from Environment import Environment
from Products import Products
from Products import buyableProducts
from User import User

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prices = {Products.P1: 7, Products.P2: 4, Products.P3: 5, Products.P4: 3, Products.P5: 2}
    costs = {Products.P1: 0, Products.P2: 0, Products.P3: 0, Products.P4: 0, Products.P5: 0}
    alphas = {Products.P1: 0.2, Products.P2: 0.2, Products.P3: 0.2, Products.P4: 0.2, Products.P5: 0.1,
              Products.P0: 0.1}
    weights = {Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
               Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
               Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
               Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
               Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
               }
    M = {Products.P1: {Products.P1: 0, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 1},
         Products.P2: {Products.P1: 1, Products.P2: 0, Products.P3: 1, Products.P4: 1, Products.P5: 1},
         Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
         Products.P4: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 0, Products.P5: 1},
         Products.P5: {Products.P1: 1, Products.P2: 1, Products.P3: 1, Products.P4: 1, Products.P5: 0},
         }
    M0 = {Products.P1: {Products.P1: 0.1, Products.P2: 0.05, Products.P3: 0.2, Products.P4: 0.05, Products.P5: 0.05},
          Products.P2: {Products.P1: 0.05, Products.P2: 0.05, Products.P3: 0.05, Products.P4: 0.05, Products.P5: 0.05},
          Products.P3: {Products.P1: 0.05, Products.P2: 0.05, Products.P3: 0.05, Products.P4: 0.05, Products.P5: 0.05},
          Products.P4: {Products.P1: 0.05, Products.P2: 0.05, Products.P3: 0.05, Products.P4: 0.05, Products.P5: 0.05},
          Products.P5: {Products.P1: 0.05, Products.P2: 0.05, Products.P3:0.05, Products.P4:0.05, Products.P5: 0.05},
          }
    pages = {Products.P1: [Products.P1, Products.P2, Products.P3],
             Products.P2: [Products.P2, Products.P2, Products.P3],
             Products.P3: [Products.P3, Products.P4, Products.P5],
             Products.P4: [Products.P4, Products.P3, Products.P5],
             Products.P5: [Products.P5, Products.P1, Products.P3], }
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

    env = Environment(alphas, weights, returnerWeights, M, M0, 0.8, prices, costs, pages, 3)
    user = env.generateUser()
    user.firstLandingItem = Products.P1
    if user.firstLandingItem is not Products.P0:
        env.shoppingItems(user.firstLandingItem, user, env.userWeights(user), [])
    print(user.cart)


    print(env.userWillReturn(user))
