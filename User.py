from Products import buyableProducts


class User():
    def __init__(self, reservationPrice, firstLandingItem):
        self.reservationPrice = reservationPrice
        self.discountedItem = None
        self.firstLandingItem = firstLandingItem
        self.returner = None
        # list of pairs <Products,quantity>
        self.cart = []
        assert len(reservationPrice) == len(buyableProducts())

    def addCart(self, product, quantity):
        self.cart.append([product, quantity])

    def emptyCart(self):
        self.cart = []
