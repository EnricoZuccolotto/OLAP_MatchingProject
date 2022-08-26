from enum import Enum
# product P0 means competitor product
# a list of all the products and allow us to cycle over them

# returns a list of the products that the user can buy in our website
def buyableProducts():
    return [x for x in Products if x is not Products.P0]


class Products(Enum):
    P1 = 0
    P2 = 1
    P3 = 2
    P4 = 3
    P5 = 4
    P0 = 5
