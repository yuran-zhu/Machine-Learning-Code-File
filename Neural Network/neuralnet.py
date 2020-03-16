# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:00:08 2020

@author: Yuran Zhu
"""
import numpy as np

def readFile(f):
    data = np.genfromtxt(f, delimiter=',')
    y = data[:,0]
    x = data[:, 1:]
    M = np.shape(x)[1]
    x = np.insert(x, M, values=1, axis=1)
    return x, y  

def sigmoid(x):
    '''
    calculate the sigmoid function
    '''
    return 1/(1+np.exp(-x))

def softmax(x):
    '''
    return probability of each class
    '''
    s = np.exp(x).sum()
    return np.exp(x)/s


def SGD(x, y, lamda, alpha, beta, M, D):
    '''
    return updated weights after operating on one example
    '''
    # forward
    x_i = x
    x_i.shape = (M,1)
    y_i = np.zeros((10,1))   # force to reshape
    y_i[int(y)] = 1   # create one-hot vector
    a = np.dot(alpha, x_i)   # First layer output (before sigmoid)
    z = sigmoid(a)  
    z = np.insert(z, D, values=1, axis=0)  # Hidden layer output (after sigmoid)
    b = np.dot(beta, z)  # Second layer output (before softmax)
    y_hat = softmax(b)  # highest probability
    # loss = - np.dot(y_i.T, np.log(y_hat))
    # backward
    dldb = y_hat - y_i
    dldbeta = np.dot(dldb, z.T)
    dldz = np.dot(dldb.T, beta[:,:-1]).T
    dlda = dldz * z[:-1,] * (1 - z[:-1,])
    dldalpha = np.dot(dlda, x_i.T)
    # update
    alpha -= lamda * dldalpha 
    beta -= lamda * dldbeta
    return alpha, beta


def meanLoss(x, y, alpha, beta, M, D):
    '''
    use updated weights to calculate mean cross entropy
    '''
    J = 0
    for i in range(0, len(x)):
        x_i = x[i]
        x_i.shape = (M,1)
        y_i = np.zeros((10,1))   # force to reshape
        y_i[int(y[i])] = 1   # create one-hot vector
        a = np.dot(alpha, x_i)   # First layer output (before sigmoid)
        z = sigmoid(a)  
        z = np.insert(z, D, values=1, axis=0)  # Hidden layer output (after sigmoid)
        b = np.dot(beta, z)  # Second layer output (before softmax)
        y_hat = softmax(b)  # highest probability
        loss = - np.dot(y_i.T, np.log(y_hat))
        J += loss
    return J/len(x)


def train(x, y, lamda, numEpoch, alpha, beta, x_test, y_test, M, D):
    '''
    given # of epoch, return mean cross entropy after each epoch
    '''
    cnt = 0
    J_train_mean = []
    J_test_mean = []
    while cnt < numEpoch:
        for i in range(0, len(x)):
            alpha, beta = SGD(x[i], y[i], lamda, alpha, beta, M,D)
        J_train_mean.append(meanLoss(x, y, alpha, beta, M, D))
        J_test_mean.append(meanLoss(x_test, y_test, alpha, beta, M, D))
        cnt += 1   
    return alpha, beta, J_train_mean, J_test_mean


def initialize(numFeature, numUnit, flag):
    '''
    initialize weights, alpha and beta
    '''
    M = numFeature
    D = numUnit
    if flag == 1:
        alpha = (np.random.rand(D,M)-0.5)/5  # uniform distribution over[-0.1,0.1] 
        alpha[:,-1] = 0  # bias = 0
        beta = (np.random.rand(10,D+1)-0.5)/5 
        beta[:,-1] = 0
    if flag == 2:
        alpha = np.zeros((D,M))
        beta = np.zeros((10,D+1))
    return alpha, beta
            
def predict(x, alpha, beta, M, D):
    '''
    predict label for given data set with updated weights
    '''
    label = []
    for i in range(0, len(x)):
        x_i = x[i]
        x_i.shape = (M,1)
        a = np.dot(alpha, x_i)   # First layer output (before sigmoid)
        z = sigmoid(a)  
        z = np.insert(z, D, values=1, axis=0)  # Hidden layer output (after sigmoid)
        b = np.dot(beta, z)  # Second layer output (before softmax)
        y_hat = softmax(b)  # highest probability
        y_hat_list = y_hat.tolist()
        label.append(y_hat_list.index(y_hat.max()))
    return label


def main(): 
    x_train, y_train = readFile(train_in)
    x_test, y_test = readFile(test_in)
    M = np.shape(x_train)[1]
    numEpoch = num_epoch
    D = hidden_units
    flag = init_flag
    lamda = learn_rate
    
    alpha, beta = initialize(M, D, flag)
    
    alpha, beta, J_train_mean, J_test_mean = train(
            x_train, y_train, lamda, numEpoch, alpha, beta, x_test, y_test, M, D)
    train_label = predict(x_train, alpha, beta, M, D)
    test_label = predict(x_test, alpha, beta, M, D)
    
    error_train = sum(np.array(train_label) != y_train)
    error_test = sum(np.array(test_label) != y_test)
    
    with open(train_out, 'w') as g:
        for i in range(0, len(train_label)):
            g.write('%d\n' % (train_label[i]))
        
    with open(test_out, 'w') as g:
        for i in range(0, len(test_label)):
            g.write('%d\n' % (test_label[i]))
            
    with open(metrics_out, 'w') as g:
        for i in range(1, numEpoch+1):
            g.write('epoch=%d crossentropy(train): %f\nepoch=%d crossentropy(test): %f\n' 
                    % (i, J_train_mean[i-1], i, J_test_mean[i-1]))
        g.write('error(train): %f\nerror(test): %f'
                    % (error_train/len(x_train), error_test/len(x_test)))
   

import sys 

if __name__=='__main__':
    
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learn_rate = float(sys.argv[9])

    main()
