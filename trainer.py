__author__ = 'syueh'
import sys
import numpy
import math

def distance_euclidean(data, neuron):
    return math.sqrt(numpy.sum(numpy.power((data - neuron), 2)))

class Trainer(object):
    def __init__(self, network, dimension):
        self._network  = network
        self._dimension = dimension
        self._p_matrix   = [ [0] * self._network._size_w for i in range(self._network._size_h)]
        #self._p_matrix   = [ 0 for i in range(self._network._size_w) * self._network._size_h]
        print ("Neuron Activations: {}").format(self._p_matrix)

    def getPmatrix(self):
        return self._p_matrix

    def pick_nearest(self, entry):
        distances = [distance_euclidean(entry, self._network.get_weight(i,j))
        for i in range(self._network._size_h) for j in range(self._network._size_w) ]

        print ("Distances: {}").format(distances)
        winner = distances.index(min(distances))
        return (winner / self._network._size_w,winner % self._network._size_w)

    def update_winnerWeights(self, i, j, entry, learning_rate):
        #wi2 (new) = wi2(old) + 0.6[xi-wi2(old)]
        self._network.update(i, j, entry, learning_rate)

    def reset_Pmatrix(self):
        self._p_matrix   = [ [0] * self._network._size_w for i in range(self._network._size_h)]

    def train(self, entry, learning_rate):
        winner  = self.pick_nearest(entry)

        self._p_matrix[winner[0]][winner[1]] += 1
        print ("Neuron winner: {}").format (winner)
        self.update_winnerWeights(winner[0], winner[1], entry, learning_rate)
        self.update_neighborns(entry, learning_rate, winner)
        self._network.printNet()

    def update_neighborns(self, entry, learning_rate, winner):
        distances = [distance_euclidean(entry, self._network.get_weight(i,j))
        for i in range(self._network._size_h) for j in range(self._network._size_w) ]
        #print "Distances: {}".format(distances)
        for p in range(len(distances)):
            i = p / self._network._size_w
            j = p % self._network._size_w
            influence = math.exp(-(distances[p]) / (2*(self._dimension*self._dimension)))
            if i != winner[0] and j != winner[1]:
                self._network.update(i, j, entry, learning_rate, influence)



