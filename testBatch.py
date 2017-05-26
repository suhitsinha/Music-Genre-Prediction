##########################
# Code: Test Batch Data  #
##########################
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pickle

def testBatch():
    # nGen: no of genres
    nGen=10
    
    #creating a neural network with 13 input nodes, 20 hidden nodes in 1 hidden layer, and 10 output nodes to each class
    netX = buildNetwork(13, 20, 10, bias=True)
    #setting parameters to previously generated values so update continues.
    new_params = np.loadtxt('test.out',delimiter=',')
    netX._setParameters(new_params)
    
    # Load testing data from csv file
    X=np.loadtxt(open("testData.csv","rb"),delimiter=",",skiprows=0)
    row,col=X.shape
    
    # testIn contains the test input feature vectors of dimesion 13
    testIn=X[:,0:col-nGen]
    
    # totalcnt: total correct count so far
    totalcnt=0      
    
    # Iterate over all genres 
    for genN in range(0,nGen):
        cnt=0
        
        # For all songs in current genre
        for curSong in range(10*genN, 10*genN+10):
            
            # Feedforward step
            Y=netX.activate(testIn[curSong])
            
            # Increase count for every correct dataset
            if(max(Y)==Y[genN]):
                cnt=cnt+1
        
        # Print current genre count
        print str(genN)+": "+str(cnt)
        
        # Calculate total correct count so far
        totalcnt = totalcnt + cnt
            
    # Print total recognized songs
    print "====================="
    print "Total: " + str(totalcnt)
    print "====================="
