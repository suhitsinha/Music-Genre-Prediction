##########################################################
# Code: Train the Generated Data using Back Propagation  #
##########################################################
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pickle

def trainData(nVar,nVarHid):
    #######################################################
    # Local Variables using for building Neural Network:  #  
    # nIn  : No of input layer neurons                    #
    # nHid : No of hidden layer neurons                   #
    # nOut : No of output layer neurons                   #
    # epochNow : no of epoches made so far                #
    # nEpoche  : no of epoches to be made                 #
    #######################################################
    nIn=13
    nHid=nVarHid
    nOut=10
    epochNow=0
    nEpoch=nVar
    
    # load training feature vectors from csv file
    X=np.loadtxt(open("trainData.csv","rb"),delimiter=",",skiprows=0)
    row,col=X.shape
    
    # trainIn contains the input feature vectors of dimesion 13
    trainIn=X[:,0:col-nOut]
    
    # outIn contain the expected output per input feature vector of dimension 10
    outIn=X[:,col-nOut:col]
    
    # Set parameters from pybrain Neural-Network
    ds=SupervisedDataSet(nIn, nOut)
    ds.setField('input', trainIn)
    ds.setField('target', outIn)
    
    # Build the Neural-Network with assiciated parameters set
    net = buildNetwork(nIn, nHid, nOut, bias=True)


    new_params = np.loadtxt('test.out',delimiter=',')
    net._setParameters(new_params)
    
    # Initiate Backpropagation trainer of pybrain library
    trainer = BackpropTrainer(net, ds)
    
    print 'Train Started'
    
    # Run Training for nEpoches
    while(epochNow<=nEpoch): 
        epochNow=epochNow+1
        trainer.train()
        
        # Print Current Status after every 100 epoches 
        if(epochNow%100==0):
            totalcnt=0      # total correct count so far
            
            # Print how many correct recognized songs per genre so far
            for k in range(0,nOut):
                cnt=0
    
                # For all songs in current genre
                for j in range(90*k,90*k+90):
                    
                    #Feedforward Step
                    Y=net.activate(trainIn[j])
                    
                    if(max(Y)==Y[k]):
                        cnt=cnt+1
                
                # Update Status and print curcount
                totalcnt = totalcnt +cnt
                print str(k)+": "+str(cnt)
            
            # Print total recognized songs so far
            print "=================================="
            print "Iteration: " +str(epochNow)+ " Total: " + str(totalcnt)
            print "=================================="


    np.savetxt('test.out',net.params,delimiter=',')

    print 'Train Completed'
