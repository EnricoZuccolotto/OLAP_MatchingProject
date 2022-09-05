from Step1_Environment.Products import buyableProducts
from Step1_Environment.Products import Products


class User():
    def __init__(self, reservationPrice, firstLandingItem):
        self.reservationPrice = reservationPrice
        self.discountedItem = Products.P0
        self.firstLandingItem = firstLandingItem
        self.returner = False
        # list of pairs <Products,quantity>
        self.cart = []
        # 1 means we have seen the product and bought it
        # 0 means we have seen the product and didn't bought it
        # 0.5 means the user didn't see the product
        self.probabilityFutureBehaviour=[0.5 for _ in range(5)]
        # contains how the user explored our webpages
        self.episode=None
        assert len(reservationPrice) == len(buyableProducts())

    def addCart(self, product, quantity):
        self.cart.append([product, quantity])

    def emptyCart(self):
        self.cart = []
