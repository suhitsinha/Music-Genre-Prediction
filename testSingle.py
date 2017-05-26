###########################
# Code: Test Single Data  #
###########################
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import pickle

def testSingle(nVar):

    #creating a neural network with 13 input nodes, 20 hidden nodes in 1 hidden layer, and 10 output nodes to each class
    netX = buildNetwork(13, 20, 10, bias=True)
    #setting parameters to previously generated values so update continues.
    new_params = np.loadtxt('test.out',delimiter=',')
    netX._setParameters(new_params)
    
    songIn=[]   
            
    # pathTemp: generate path of the music file
    pathTemp=nVar
            
    # Read musicfile to be used for generating train-data        
    sample_rate, X = scipy.io.wavfile.read(pathTemp)
            
    # Calculate mfcc coefficients of the music file
    ceps, mspec, spec = mfcc(X)
    row,col=ceps.shape
        
    # Take mean of all mfcc coef
    # The test vector contains 13 features for each song
    ceps=np.mean(ceps, axis=0)
    songIn=ceps.tolist()
            # Append the ouput label for the particular Genre to be used for Verification
        
    # Feedforward step
    Y=netX.activate(songIn)
    
    # Find Genre
    for i in range(0,10):
        if(max(Y)==Y[i]):
            print 'Genre Detected: '+str(i)
            break
        
