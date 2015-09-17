__author__ = 'syueh'

# Kohonen implementation for IA Topic materia

import sys
#import pylab
import numpy as np
import random
from nnetwork import NeuralNetwork
from trainer import Trainer, distance_euclidean
import matplotlib.pyplot as plt


RATE_LEARNING = 0.6
REDUCTION_RATE_LEARNING = 0.5

def main(args):
    """
		the program entry should be: main.py  height_size_kohonen width_size_ Kohonen
			path_of_trainer_data_set epocas
	"""
    size_height = int(args[1])
    size_width  = int(args[2])
    path 		= args[3]
    epocas 		= int(args[4])
    entry = []
    with open(path,"r") as entries:
        for line in entries:
            print (map(float,line.split(" ")))
            entry.append(map(float,line.split(" ")))
    dimension = len(entry[0])
    #neural network for kohonen algorithm

    net = NeuralNetwork(size_height, size_width, dimension)
    trainer = Trainer(net, dimension)

    #training the net
    current_learning_rate = RATE_LEARNING
    for i in range(epocas):
        trainer.reset_Pmatrix()
        for current_entry in entry:
            trainer.train(current_entry, current_learning_rate)
        current_learning_rate *= REDUCTION_RATE_LEARNING

    print ("Final Neuron Activations: {}").format(trainer._p_matrix)

    conf_arr = trainer._p_matrix
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a == 0.0:
                tmp_arr.append(0.0)
                continue
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
	                interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
	                    horizontalalignment='center',
	                    verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('pmatrix.png', format='png')


if __name__ == "__main__":
    main(sys.argv)