import numpy as np

class Node:

    def __init__(self, value=None, data=None, left=None, right=None):
        self.value  = value     # Float: Value of the node
        self.data   = data      # Float: Data value associated with that node
        self.left   = left      # Node: Left child
        self.right  = right     # Node: Right child

    def add(self, node):
        if  not self.value and self.value != 0:
            self.value = node.value
            self.data = node.data
        else:
            if node.value < self.value:
                if not self.left and self.left != 0 : self.left = Node()
                self.left.add(node)
            else:
                if not self.right and self.right != 0 : self.right = Node()
                self.right.add(node)



def binary_hvg(X, sort='mergesort'):
    '''
    use 'mergesort' for worst case O(n log n)
    or 'quicksort' for worst case O(n^2) but perhaps better average performance
    '''

    root = Node()

    sorted_indexes = np.argsort(X, kind=sort)[::-1]
    sorted_data = X[sorted_indexes]

    for (index, data) in zip(sorted_indexes, sorted_data):
        root.add(Node(value = index, data = data))

    return root
