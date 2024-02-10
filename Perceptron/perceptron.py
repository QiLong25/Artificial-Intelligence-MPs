# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4

    N = np.shape(train_set)[0]
    D = np.shape(train_set)[1]

    ### Step 1: Initialize W, b, lr, train_x
    W = np.zeros(D+1, dtype='float64')          # append b into w
    lr = 0.1
    one = np.ones(N).reshape([-1, 1])
    train_set = np.append(one, train_set, axis=1)       # append ones to x

    for epoch in range(max_iter):
        for x_idx in range(len(train_set)):
            x = train_set[x_idx]

            ### Step 2: Calculate prediction
            y_hat = float(x @ W)                           # pred in {0, 1} form
            if y_hat > 0:
                y_hat = 1
            else:
                y_hat = 0
            if y_hat == train_labels[x_idx]:
                continue

            if train_labels[x_idx] > 0:
                y_use = 1
            else:
                y_use = -1

            # y_use = np.ones(np.shape(train_labels))    # pred in {-1, 1} form
            # y_use[train_labels == 0] = -1

            ### Step 3: Update W
            W += lr * x * y_use

    ### Step 4: split W and b
    b = W[0]
    W = W[1:]

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4

    ### Step 1: train perceptron
    W, b = trainPerceptron(train_set, train_labels, max_iter)

    ### Step 2: make prediction
    pred = dev_set @ W + b          # pred in {0, 1} form
    pred[pred > 0] = 1
    pred[pred <= 0] = 0

    return list(pred)



