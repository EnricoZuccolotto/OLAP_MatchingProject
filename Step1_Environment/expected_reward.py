
from Step1_Environment.Environment import Products
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

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

# productsThatUserCanBuy list of products or set
def printPreorder(node,seen,productsThatUserCanBuy,tot):
    if node and node.val in productsThatUserCanBuy:
        # First print the data of node
        print(node.val)
        seen.add(node.val)

            # Then recur on left child
        if node.left:
            if node.left.val not in seen and node.left.val in productsThatUserCanBuy:
                tot=tot*W[node.val][node.left.val]
                tot=printPreorder(node.left,seen,productsThatUserCanBuy,tot)
                seen.remove(node.left.val)
        if node.right:
            if node.right.val not in seen and node.right.val in productsThatUserCanBuy:
                # Finally recur on right child
                tot = tot * W[node.val][node.left.val]
                tot = printPreorder(node.left, seen, productsThatUserCanBuy, tot)
                printPreorder(node.right,seen,productsThatUserCanBuy,tot)
                seen.remove(node.right.val)
        return tot


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
root=build_tree(Products.P5,pages)
seen=set()
printPreorder(root,seen,[Products.P1,Products.P3,Products.P2])