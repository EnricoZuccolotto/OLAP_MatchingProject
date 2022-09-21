import math

import numpy as np
from Step1_Environment.User import User
import copy

class Environment():
    def __init__(self, alphas, weights, returnerWeights, M, M0, prices, costs, maxQuantity):
        self.alphas = alphas
        self.weights = weights
        self.returnerWeights = returnerWeights
        self.M = M
        self.M0 = M0
        self.prices = prices
        self.costs = costs

        self.maxQuantity = maxQuantity



        # TODO:some other assertions
        assert len(prices) == 5
        assert len(costs) == 5
        assert math.fsum(list(alphas)) == 1

        assert 6 == len(list(alphas))
        # for p in range(5):
        #     assert math.fsum(M0[p]) <= 1

    def generateRandomUser(self,n):
        l=[]
        for i in range(n):
            u=self.generateUser()
            if u is not None:
                l.append(u)
        return l

    # Generate a User and randomly choose an reservation price for each product and the landing page
    def generateUser(self):
         # np.random.normal (mean,std deviation)
        reservationPrice= [max(0,np.random.normal(self.prices[p], 2)) for p in range(5)]
        #     to keep the values in order
        firstLandingProduct = np.random.choice([5,0,1,2,3,4],p=self.alphas)

        if firstLandingProduct == 5:
            return
        return User(reservationPrice, firstLandingProduct)

    def simulateEpisode(self, product, user,usableWeights):
        prob_matrix = usableWeights.copy()

        active_nodes = np.zeros(5)
        active_nodes [product]=1
        history = np.array([active_nodes])
        if not self.shoppingItem(product,user):
            return history
        newly_active_nodes = active_nodes
        t = 0
        while t < 5 and np.sum(newly_active_nodes) > 0:
            p = (prob_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            prob_matrix = prob_matrix * ((p != 0) == activated_edges)
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)
            if np.sum(newly_active_nodes)>0:
                history = np.concatenate((history, [newly_active_nodes]), axis=0)

            newly_active_nodes=[self.shoppingItem(i,user) if newly_active_nodes[i]!=0 else 0 for i in range(len(newly_active_nodes))]


            t += 1

        return history





    # Given a webpage,a user, the weights graph related to the user will fill up the cart of the user
    # Queue-->queue of webpages that the user wants to visit

    def shoppingItem(self, product, user):
        # Buy the quantity of products and set to 0 the possibility to return to this webpage
        if self.prices[product] < user.reservationPrice[product]:
            quantity = max(1, int(np.random.normal(self.maxQuantity, 1)))
            user.probabilityFutureBehaviour[product] = 1
        else:
            quantity = 0
            user.probabilityFutureBehaviour[product] = 0
        user.cart[product]=quantity

        return quantity>0

    # Method used to compute the value of the cart,if a discount is present it will use it (and delete it)
    def finalizePurchase(self, user):
        total = np.sum((self.prices-self.costs) * user.cart)

        # apply the discount and remove it
        if user.discountedItem<5:
            if user.cart[user.discountedItem] and total != 0:
                total = total - self.prices[user.discountedItem]
                user.discountedItem = 5
        user.emptyCart()
        #  round the sum
        return total

    # if user will return,return the product in which it will lands otherwise return None
    def returningLandingProduct(self, user):
        # if user has a discounted item will land for sure on the discounted page
        if user.discountedItem <5:
            if np.random.rand() < self.M[user.firstLandingItem][user.discountedItem]:
                return user.discountedItem
        else:
            # No discounted item-->will land to page of product p with probability M0[firstLandingProduct][p],
            # otherwise it will not return
            product = np.random.choice([5,0,1,2,3,4], p=np.append([1 - math.fsum(self.M0[user.firstLandingItem])],self.M0[user.firstLandingItem]))
            # Product.p0 it means the user wont return
            return product
        return 5

    #     return the appropriate weights to each user depending on the class of the user
    def userWeights(self, user):
        if user.returner and user.discountedItem <5:
            return copy.deepcopy(self.returnerWeights[user.discountedItem])
        else:
            return copy.deepcopy(self.weights)

    def userVisits(self,user, product):
        # if the user doesn't land on the competitor website and he returned
        if product <5:
            user.episode=self.simulateEpisode(product, user, self.userWeights(user))
            margin=self.finalizePurchase(user)
            user.returner = True
            return margin
        return 0










