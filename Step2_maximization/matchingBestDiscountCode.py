import copy
import math

import numpy as np
from Step1_Environment.Products import buyableProducts
from Step1_Environment.Products import Products

class matchingBestDiscountCode():
    def __init__(self,theta,pages,prices,costs):
        self.theta=theta
        self.pages=pages
        self.costs=costs
        self.prices=prices
        self.z=np.zeros(5)
        return
    # product->fist product the user will visit
    # weight associated to user
    # n number of repetitions
    def monteCarloRuns(self, n, product, weights, productsSeen):
        self.z = np.zeros(5)
        if productsSeen[product.value]==0:
            return 0

        for i in range(len(productsSeen)):
            if productsSeen[i]==0:
                for p in buyableProducts():
                    weights[p][buyableProducts()[i]]=0

        for i in range(n):
            self.shoppingItems(product,copy.deepcopy(weights))
        rew=0
        print(self.z/n)
        for i in self.z/n:
            assert i<=1
        for p in buyableProducts():
            rew+= self.z[p.value] / n * (self.prices[p]-self.costs[p]) * productsSeen[p.value]
        return rew

    def shoppingItems(self, product, usableWeights):
        seen = set()
        # first visit
        seen.add(product)
        productsUserWantToVisitNext= [[] for _ in range(5)]
        result = self.shoppingItem(product, usableWeights)
        productsUserWantToVisitNext[1] = self.unique([p for p in result[0] if p not in seen])
        usableWeights = result[1]
        # next visits
        i=0
        t=1
        while t<5 and len(productsUserWantToVisitNext[t])>0 :
            result = self.shoppingItem(productsUserWantToVisitNext[t][i], usableWeights)
            seen.add(productsUserWantToVisitNext[t][i])
            i+=1

            if t<4:
                pi=self.unique(productsUserWantToVisitNext[t+1]+result[0])
                productsUserWantToVisitNext[t+1]= [p for p in pi if p not in seen]
                usableWeights = result[1]

            if len(productsUserWantToVisitNext[t])==i:
                i=0
                t+=1
        # print(episode)
        # for i in range(len(episode)):
        #     for j in range(5):
        #         assert episode[i][j]<=1




    # Given a webpage,a user, the weights graph related to the user will fill up the cart of the user
    # Queue-->queue of webpages that the user wants to visit
    # Does not take Product.P0
    def shoppingItem(self, product, usableWeights):
        queue=[]
        page = self.pages[product]

        # Update z value
        self.z[product.value]+=1

        for p in buyableProducts():
            usableWeights[p][page[0]] = 0

            # Add to the queue the secondary product webpage with a certain probability given by the graph
        if np.random.rand() <= usableWeights[page[0]][page[1]]:
            queue.append(page[1])
            # Add to the queue the tertiary product webpage with a certain probability given by the graph
        if np.random.rand() <= (usableWeights[page[0]][page[2]] * self.theta):
            queue.append(page[2])

        queue=self.unique(queue)

        return [queue,usableWeights]

    # Method to eliminates duplicates and keep the order preserved
    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]



    def matcher(self,weights,returnerWeights,M,M0,user):
        w=np.zeros(6)
        print(user.probabilityFutureBehaviour)
        for p in list(Products):
            if p is Products.P0:
                print('no discount case')
                w[p.value]=self.noDiscountCase(weights,M0,user)
            else:
                print('discount case '+str(p))
                w[p.value]=self.discountCase(returnerWeights[p],M,user,p)

        print(w)
        return list(Products)[np.argmax(w)]


    def noDiscountCase(self,weights,M0,user):
        reward=0
        for p in buyableProducts():
            reward+= self.monteCarloRuns(100, p, copy.deepcopy(weights), user.probabilityFutureBehaviour) * M0[user.firstLandingItem][p]
        return reward

    def discountCase(self,returnerWeights,M,user,p):
        reward= (self.monteCarloRuns(100, p, copy.deepcopy(returnerWeights), user.probabilityFutureBehaviour) - self.prices[p]) * M[user.firstLandingItem][p]
        return reward
