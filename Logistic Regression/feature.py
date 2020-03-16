# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 01:52:25 2020

@author: Yuran Zhu
"""

import numpy as np

def readFile(f):
	'''
	read and save data set in dictionary
	'''
    with open(f, 'r') as f:
        data = []
        for line in f:
            record = dict()
            record['y'] = line.split('\t')[0]
            record['x'] = line.split('\t')[1].split(' ')
            data.append(record)
        return data

def readDict(f):
	'''
	read and save vocab
	'''
    with open(f, 'r') as f:
        v = dict()
        for line in f:
            v[line.split(' ')[0]] = line.split(' ')[1].strip()
        return v
    
def featureCatchOne(data, v):
    final = []
    for record in data:
        result = []
        result.append(record['y'])
        for w in sorted(set(record['x']), key = record['x'].index): # exist diplicated words
            if w in v.keys():
                result.append('%s:1' % (v[w]))
        final.append(result)
    return final
  
def featureCatchTwo(data, v):
    t = 4   # set threshold t, count of the word is LESS THAN it
    final = []
    for record in data:
        result = []
        result.append(record['y'])
        for w in sorted(set(record['x']), key = record['x'].index):
            if w in v.keys():
                cnt = record['x'].count(w)
                if cnt < t:
                    result.append('%s:1' % (v[w]))
        final.append(result)
    return final

def modelOut(final, out):
    '''
    output model result
    '''
    with open(out, 'w') as g:
        for record in final:
            for r in record:
                g.write('%s\t' % (r))
            g.write('\n')

def main():
    trainD = readFile(trainIn)
    validD = readFile(validIn)
    testD = readFile(testIn)
    vocab = readDict(dictIn)
    
    if flag == 1:
        trainFnl = featureCatchOne(trainD, vocab)
        validFnl = featureCatchOne(validD, vocab)
        testFnl = featureCatchOne(testD, vocab)
    if flag == 2:
        trainFnl = featureCatchTwo(trainD, vocab)
        validFnl = featureCatchTwo(validD, vocab)
        testFnl = featureCatchTwo(testD, vocab)
        
    modelOut(trainFnl, trainOut)
    modelOut(validFnl, validOut)
    modelOut(testFnl, testOut)
    

import sys 

if __name__=='__main__':
    
    trainIn = sys.argv[1]
    validIn = sys.argv[2]
    testIn = sys.argv[3]
    dictIn = sys.argv[4]
    trainOut = sys.argv[5]
    validOut = sys.argv[6]
    testOut = sys.argv[7]
    flag = int(sys.argv[8])

    main()
