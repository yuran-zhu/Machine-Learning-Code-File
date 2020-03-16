# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:18:07 2020

@author: Yuran Zhu
"""

import numpy as np

def readFile(f):
    with open(f, 'r') as f:
        data = []
        for line in f:
            record = dict()
            record['y'] = int(line[0])
            X = dict()
            X[-1] = 1   # add for counting for bias term
            for x in line[1:].strip().split('\t'):
                X[int(x.split(':')[0])] = 1   # dict for all x(i)
            record['x'] = X
            data.append(record)
        return data

def readDict(f):
    with open(f, 'r') as f:
        v = dict()
        for line in f:
            v[line.split(' ')[0]] = line.split(' ')[1].strip()
        return v

def sigmoid(a):
    '''
    calculate the sigmoid function
    '''
    return 1/(1+np.exp(-a))

def sparse_dot(X, W):
    product = 0.0
    for i, v in X.items():
        if i not in W:
            continue
        product += W[i] * v
    return product

def sgd(data, W, lamda):    # for each line
    '''
    return updated weight after SGD
    '''
    lab = data['y']
    X = data['x']
    product = sparse_dot(X, W)
    for j, v in X.items():
        W[j] += lamda *v* (lab - sigmoid(product))   # update W
    return W

def train(data, W, lamda, numEpoch):
    '''
    train data set and return the final updated weight
    '''
    cnt = 0
    while cnt < numEpoch:
        for line in data:
            W = sgd(line, W, lamda)
        cnt += 1    
    return W
        
def predict(d, wt):
    label = []
    for line in d:
        X = line['x']
        product = sparse_dot(X, wt)
        prob = sigmoid(product)
        if prob >= 0.5:
            lab = 1
        else:
            lab = 0
        label.append(lab)
    return label

    
def main(): 
    # read data
    trainD = readFile(trainIn)
    validD = readFile(validIn)
    testD = readFile(testIn)
    vocab = readDict(dictIn)
    num = numEpoch
    
    # initialize W and get final wt
    W = dict()
    for i in range(-1, len(vocab.keys())):
        W[i] = 0
    wt = train(trainD, W, 0.1, num)  
    
    # predict
    result1 = predict(trainD, wt)
    result2 = predict(testD, wt)
    
    # error
    trainError = 0
    testError = 0
    for i in range(0, len(result1)):
        if result1[i] != trainD[i]['y']:
           trainError +=1
    trainRate = trainError/len(result1)
    for i in range(0, len(result2)):
        if result2[i] != testD[i]['y']:
           testError +=1
    testRate = testError/len(result2)
    
    # write files
    with open(trainOut, 'w') as g:
        for r in result1:
            g.write('%s\n' % (r))
            
    with open(testOut, 'w') as g:
        for r in result2:
            g.write('%s\n' % (r))
            
    with open(metricsOut, 'w') as g:
        g.write('error(train): %f\nerror(test): %f' % (trainRate, testRate))
         

import sys 

if __name__=='__main__':
    
    trainIn = sys.argv[1]
    validIn = sys.argv[2]
    testIn = sys.argv[3]
    dictIn = sys.argv[4]
    trainOut = sys.argv[5]
    testOut = sys.argv[6]
    metricsOut = sys.argv[7]
    numEpoch = int(sys.argv[8])

    main()
