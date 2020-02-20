# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 02:34:38 2020

@author: Yuran Zhu
"""

import sys    
import numpy as np

def readFile(f):
    '''
    def function to read the file
    @return: dataset, attribute list
    '''
    with open(f, 'r'): 
        data = np.genfromtxt(f, delimiter="\t", dtype=None, encoding=None)
    dataset = data[1:,]   #  examples
    attributes = data[0,:-1]  # first line: attibutes
    return dataset, attributes


def giniImpurity(data):
    '''
    Calculate gini impurity of dataset
    '''
    n = len(data)   # total number of examples
    cnt1 = sum(data[:,-1]==data[0,-1])   # cnt for label 1
    cnt2 = sum(data[:,-1]!=data[0,-1])   # cnt for label 2
    giniImp = (cnt1/n)*(cnt2/n)*2
    return giniImp


def giniGain(data, index):
    '''
    Calculate gini gain for splitting on specific index
    '''
    attrVals = list(set(data[:, index]))   # 2 values of attribute[index]
    if len(attrVals) != 1:
        rootImp = giniImpurity(data)
        sub0 = data[data[:, index]== attrVals[0]]
        prob0 = sum(data[:, index]== attrVals[0])/len(data)   
        sub1 = data[data[:, index]== attrVals[1]]
        prob1 = sum(data[:, index]== attrVals[1])/len(data)
        gain = rootImp - prob0 * giniImpurity(sub0) - prob1 * giniImpurity(sub1)    
    else:
        gain = 0   # only 1 value for attribute, cannot split
    return gain


def bestToSplit(data):
    '''
    Choose the best attribute to split data (largest gini_gain)
    @return: the index of the best attribute
    '''
    numAttr = len(data[0,:])-1   # number of attributes
    if len(list(set(data[:, -1]))) == 1:   # one label, no need to split
        return None
    
    else: 
        bestGain = 0  # original gain to be compared
        bestAttr = None
        for i in range(0, numAttr):  # loop on each attribute, decide if it's the best
            if giniGain(data, i) > bestGain:
                bestGain = giniGain(data, i)
                bestAttr = i
        return bestAttr


def majorityVote(data):
    classes = list(set(data[:, -1]))  # two values for class
    if len(classes) != 1:
        if sum(data[:, -1] == classes[0]) > 0.5 * len(data):
            return classes[0]  # determine majority-vote value2
        elif sum(data[:, -1] == classes[0]) == 0.5 * len(data): 
            return max(classes)   
        else:
            return classes[1] 
    else:
        return classes[0] 


class Node:
    def __init__(self, key, data, depth):
        self.val = key   # attr index
        self.left = None
        self.right = None
        self.data = data
        self.depth = depth
        self.lbranch = None     # attr value[0]
        self.rbranch  = None    # value[1]
        self.lresult = None    # majorityvote[0]
        self.rresult = None    # majorityvote[1]


def trainTree(node):   
    '''
    Create dicision tree with the root
    '''
    data = node.data
    attrVals = list(set(data[:, node.val]))    # 2 values of attribute[index]
    if len(attrVals) != 1:
        node.lbranch = attrVals[0]
        node.rbranch = attrVals[1]
        sub0 = data[data[:, node.val]==  node.lbranch]
        sub1 = data[data[:, node.val]==  node.rbranch]

        if node.depth < maxDepth and node.depth < (len(data[0,:])-1): 
            # have enough attr to consider splitting
            if bestToSplit(sub0) != None:
                node.left = Node(bestToSplit(sub0), sub0, node.depth+1) 
                trainTree(node.left)  # continue growing
            else:
                node.lresult = majorityVote(sub0)
            
            if bestToSplit(sub1) != None:
                node.right = Node(bestToSplit(sub1), sub1, node.depth +1)
                trainTree(node.right)
            else:
                node.rresult = majorityVote(sub1)
        else: # stop splitting or cannot split anymore
            node.lresult = majorityVote(sub0)
            node.rresult = majorityVote(sub1)
 
    else:  # one value for attribute, cannot split, do majority-vote
        node.lbranch = attrVals[0]
        node.lresult = majorityVote(data)
    return node   


def printTree(data, node, attributes):
    classes = list(set(data[:, -1]))
    attr = attributes
    
    while node != None:
        if node.depth == 1:   # print sum for the whole dataset
            cnt0 = sum(data[:,-1] == classes[0])   # cnt for class 1
            cnt1 = sum(data[:,-1] == classes[1]) 
            print("[%d %s/%d %s]" % (cnt0, classes[0], cnt1, classes[1])) 
            
        if len(classes) != 1:
            sub0 = data[data[:, node.val]==  node.lbranch]
            leftCnt0 = sum(sub0[:,-1] == classes[0])  
            leftCnt1 = sum(sub0[:,-1] == classes[1]) 
            print("| "*node.depth, "%s = %s: [%d %s/%d %s]" % 
                  (attr[node.val], node.lbranch, leftCnt0, classes[0], leftCnt1, classes[1])) 
            printTree(sub0, node.left, attributes)
            
            sub1 = data[data[:, node.val]==  node.rbranch]
            rightCnt0 = sum(sub1[:,-1] == classes[0])   
            rightCnt1 = sum(sub1[:,-1] == classes[1]) 
            print("| "*node.depth, "%s = %s: [%d %s/%d %s]" % 
                  (attr[node.val], node.rbranch, rightCnt0, classes[0], rightCnt1, classes[1])) 
            printTree(sub1, node.right, attributes)    
        break     
      
 
def predict(r, node):
    '''
    Predict label for given examples
    @parameter: dataset, decision tree created
    '''
    if r[node.val] == node.lbranch and node.left == None: 
        label = node.lresult
    elif r[node.val] == node.lbranch and node.left != None:   # continue searching
        label = predict(r, node.left)    
    if r[node.val] == node.rbranch and node.right == None:
        label = node.rresult
    elif r[node.val] == node.rbranch and node.right != None:
        label = predict(r, node.right)    
    return label


def main():
    
    trainData, trainAttr = readFile(trainIn)
    testData, testAttr = readFile(testIn)
    root = Node(bestToSplit(trainData), trainData, 1)
    myTree = trainTree(root)
    printTree(trainData, myTree, trainAttr)
    trainErr = 0
    testErr = 0
    
    with open(trainOut, "w") as g:
        for r in trainData:
            g.write(predict(r, myTree) + "\n")
            if r[-1] != predict(r, myTree):
                trainErr += 1
    
    with open(testOut, "w") as g:
        for r in testData:
            g.write(predict(r, myTree)+ "\n")
            if r[-1] != predict(r, myTree):
                testErr += 1
    
    trainErrRate = trainErr/len(trainData)
    testErrRate = testErr/len(testData)
    
    with open(metricsOut, "w") as g:
        g.write("error(train): %f\nerror(test): %f" % (trainErrRate, testErrRate))


if __name__=='__main__':
    # trainIn = sys.argv[1]
    # testIn = sys.argv[2]
    # maxDepth = int(sys.argv[3])
    # trainOut = sys.argv[4]
    # testOut = sys.argv[5]
    # metricsOut = sys.argv[6]
    
    trainIn = "education_train.tsv"
    testIn = "education_test.tsv"
    maxDepth = 3
    trainOut = "edu_3_train.labels"
    testOut = "edu_3_test.labels"
    metricsOut = "edu_3_metrics.txt"
    
    main()
    