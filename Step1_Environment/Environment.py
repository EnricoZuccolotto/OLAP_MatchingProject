import math
from Step1_Environment.Products import Products
from Step1_Environment.Products import buyableProducts
import numpy as np
from Step1_Environment.User import User
import copy

class Environment():
    def __init__(self, alphas, weights, returnerWeights, M, M0, theta, prices, costs, pages, maxQuantity):
        self.alphas = alphas
        self.weights = weights
        self.returnerWeights = returnerWeights
        self.M = M
        self.M0 = M0
        self.theta = theta
        self.prices = prices
        self.costs = costs
        self.pages = pages
        self.maxQuantity = maxQuantity


        self.episode=[]

        # TODO:some other assertions
        assert len(prices) == len(buyableProducts())
        assert len(costs) == len(buyableProducts())
        assert math.fsum(list(alphas.values())) == 1
        assert len(pages) == len(buyableProducts())
        assert len(list(Products)) == len(list(alphas.values()))
        for p in buyableProducts():
            assert math.fsum(M0[p].values()) <= 1


    # Generate a User and randomly choose an reservation price for each product and the landing page
    def generateUser(self):
        reservationPrice = {}
        for p in buyableProducts():
            # np.random.normal (mean,std deviation)
            price = np.random.normal(self.prices[p], 2)
            reservationPrice[p] = float(f'{price:.2f}')
        #     to keep the values in order
        prob = [self.alphas[p] for p in list(Products)]
        firstLandingProduct = np.random.choice(list(Products), p=prob)
        # firstLandingProduct=Products.P1
        user = User(reservationPrice, firstLandingProduct)
        return user

    def shoppingItems(self, product, user, usableWeights):
        seen = set()
        # first visit
        seen.add(product)
        productsUserWantToVisitNext= [[] for _ in range(5)]
        result = self.shoppingItem(product, user, usableWeights)
        productsUserWantToVisitNext[0] = [product]
        productsUserWantToVisitNext[1] = result[0]
        usableWeights = result[1]
        # next visits
        episode=[[0 for _ in range(5)]]
        episode[0][product.value]=1

        i=0
        t=1
        while t<5 and len(productsUserWantToVisitNext[t])>0 :
            if i == 0:
                episode.append([0 for _ in range(5)])
            result = self.shoppingItem(productsUserWantToVisitNext[t][i], user, usableWeights)
            seen.add(productsUserWantToVisitNext[t][i])
            episode[t][productsUserWantToVisitNext[t][i].value] = 1
            i+=1
            if t<4:
                pi=productsUserWantToVisitNext[t+1]+result[0]
                productsUserWantToVisitNext[t+1]= [p for p in pi if p not in seen]
                usableWeights = result[1]

            if len(productsUserWantToVisitNext[t])==i:
                i=0
                t+=1

        self.episode=episode


    # Given a webpage,a user, the weights graph related to the user will fill up the cart of the user
    # Queue-->queue of webpages that the user wants to visit
    # Does not take Product.P0
    def shoppingItem(self, product, user, usableWeights):
        queue=[]
        page = self.pages[product]

        # Buy the quantity of products and set to 0 the possibility to return to this webpage
        if self.prices[page[0]] < user.reservationPrice[page[0]]:
            quantity = max(1, int(np.random.normal(self.maxQuantity, 1)))
            user.addCart(page[0], quantity)
        else:
            quantity = 0

        for p in buyableProducts():
            usableWeights[p][page[0]] = 0

        if quantity != 0:
            # Add to the queue the secondary product webpage with a certain probability given by the graph
            if np.random.rand() < usableWeights[page[0]][page[1]]:
                queue.append(page[1])
            # Add to the queue the tertiary product webpage with a certain probability given by the graph
            if np.random.rand() < (usableWeights[page[0]][page[2]] * self.theta):
                queue.append(page[2])

        queue=self.unique(queue)

        return [queue,usableWeights]

    # Method to eliminates duplicates and keep the order preserved
    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    # Method used to compute the value of the cart,if a discount is present it will use it (and delete it)
    def finalizePurchase(self, user):
        total = 0
        # item[0]-->product , item[1]-->quantity
        for i in user.cart:
            total += (self.prices[i[0]]-self.costs[i[0]]) * i[1]
        itemInCart = [i[0] for i in user.cart]
        user.emptyCart()
        # apply the discount and remove it
        if user.discountedItem in itemInCart and total != 0:
            total = total - self.prices[user.discountedItem]
            user.discountedItem = Products.P0
        #  round the sum
        return float(f'{total:.2f}')

    # if user will return,return the product in which it will lands otherwise return None
    def returningLandingProduct(self, user):
        # if user has a discounted item will land for sure on the discounted page
        if user.discountedItem is not Products.P0:
            if np.random.rand() < self.M[user.firstLandingItem][user.discountedItem]:
                return user.discountedItem
        else:
            # No discounted item-->will land to page of product p with probability M0[firstLandingProduct][p],
            # otherwise it will not return
            prob = [self.M0[user.firstLandingItem][p] for p in buyableProducts()]
            userWontReturn = 1 - math.fsum(prob)
            prob = prob + [userWontReturn]
            product = np.random.choice(list(Products), p=prob)
            # Product.p0 it means the user wont return
            return product
        return Products.P0

    #     return the appropriate weights to each user depending on the class of the user
    def userWeights(self, user):
        if user.returner and user.discountedItem is not Products.P0:
            return copy.deepcopy(self.returnerWeights[user.discountedItem])
        else:
            return copy.deepcopy(self.weights)

    def userVisits(self,user, product):
        margin=0
        # if the user doesn't land on the competitor website and he returned
        if product is not Products.P0:
            self.shoppingItems(product, user, self.userWeights(user))
            if len(user.cart) > 0:
                margin=self.finalizePurchase(user)
            user.returner = True
            return margin
        return None

    # update the distribution in the bandit
    # decide if the user will return

    def round(self, product):
        if product is Products.P0:
            return 0
        return 1

    def returnEpisode(self):
        return self.episode