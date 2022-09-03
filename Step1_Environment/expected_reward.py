
from Step1_Environment.Environment import Products
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
        self.prob=1

def build_tree(product,pages):
    root=Node(product)
    NodesToExpand=[]
    NodesToExpand.append(root)
    for i in range(4):
        newNodes=[]
        for node in NodesToExpand:
            node.left = Node(pages[node.val][1])
            node.right = Node(pages[node.val][2])
            newNodes.append(node.left)
            newNodes.append(node.right)
        NodesToExpand=newNodes

    return root

def buildTree(n,seen,productsThatUserCanBuy):
    if n.val in productsThatUserCanBuy:
        # First print the data of node
        print(n.val)
        print(n.prob)
        seen.add(n.val)

        # Then recur on left child
        if pages[n.val][1] not in seen and pages[n.val][1] in productsThatUserCanBuy:
            n.left = Node(pages[n.val][1])
            n.left.prob=n.prob*W[n.val][n.left.val]
            buildTree(n.left,seen,productsThatUserCanBuy)
            seen.remove(n.left.val)
        if pages[n.val][2] not in seen and pages[n.val][2] in productsThatUserCanBuy:
            n.right = Node(pages[n.val][2])
            n.right.prob = n.prob * W[n.val][n.right.val]*theta
            buildTree(n.right, seen, productsThatUserCanBuy)
            seen.remove(n.right.val)


# productsThatUserCanBuy list of products or set

W = {Products.P1: {Products.P1: 0, Products.P2: 0.5, Products.P3: 0.7, Products.P4: 0.2, Products.P5: 0.3},
           Products.P2: {Products.P1: 0.4, Products.P2: 0, Products.P3: 0.3, Products.P4: 0.4, Products.P5: 0.8},
           Products.P3: {Products.P1: 1, Products.P2: 1, Products.P3: 0, Products.P4: 1, Products.P5: 1},
           Products.P4: {Products.P1: 0.5, Products.P2: 0.6, Products.P3: 0.4, Products.P4: 0, Products.P5: 0.3},
           Products.P5: {Products.P1: 0.7, Products.P2: 0.23, Products.P3: 0.16, Products.P4: 0.36, Products.P5: 0},
           }
pages = {Products.P1: [Products.P1, Products.P2, Products.P3],
            Products.P2: [Products.P2, Products.P3, Products.P5],
             Products.P3: [Products.P3, Products.P4, Products.P5],
             Products.P4: [Products.P4, Products.P5, Products.P1],
             Products.P5: [Products.P5, Products.P1, Products.P3], }

root=Node(Products.P1)
theta=0.7
see=set()
buildTree(root,see,[Products.P1,Products.P2,Products.P3])