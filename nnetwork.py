__author__ = 'syueh'
import sys
import numpy

# Neural Network Definition

class NeuralNetwork(object):
    def __init__(self, size_height, size_width, dimension):
        self._size_h = size_height
        self._size_w = size_width
        self._dim     = dimension
        self._matrix_weights = [[None] * self._size_w for i in range(self._size_h)]
        for i in range(self._size_h):
            for j in range(self._size_w):
                self._matrix_weights[i][j] = numpy.random.random_sample((dimension,))
                print (" Matriz({},{}) : {}").format(i,j,self._matrix_weights[i][j] )

    def update(self, i, j, entry, learning_rate, inf = 1.0):
        #wi2 (new) = wi2(old) + 0.6[xi-wi2(old)]
        w_temp = self._matrix_weights[i][j]
        print ("old: {}, ratel: {}").format( w_temp, learning_rate)
        for u in range(len(w_temp)):
            w_old = w_temp[u]
            w_temp[u] = w_old + inf*learning_rate * (entry[u] - w_old)
        self._matrix_weights[i][j] = w_temp
        print ("new ") , self._matrix_weights[i][j]

    def printNet(self):
        for i in range(self._size_h):
            for j in range(self._size_w):
                print (" Matriz({},{}) : {}").format(i,j,self._matrix_weights[i][j] )

    def get_weight(self, neuron_idi, neuron_idj):
        return self._matrix_weights[neuron_idi][neuron_idj]

    def __setitem__(self, neuron_idi, neuron_idj, values):
        self._matrix_weights[neuron_idi][neuron_idj] = values

    def __len__(self):
        return self._size_h * self._size_w

    def __str__(self):
        return str(self._matrix_weights)
