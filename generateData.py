####################################
# Code: Generate Data for Training #
####################################

import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

def generateData():
    ############################################################################################
    # Local Variables:                                                                         #
    # genrePath  : Denotes the music genres                                                    #
    # genreLabel : Denotes the genre labels where '1' specifes the genre-number is belongs to  #
    # trainIn    : Training Feature Vector Matrix 900X23 [13 input features, 10 output genres] #
    # nGen       : No of music genres                                                          #
    # nSong      : No of songs to be used per Genre in training set                            #                            
    ############################################################################################
    genrePath=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    genreLabel=[[1,-1,-1,-1,-1,-1,-1,-1,-1,-1],[-1,1,-1,-1,-1,-1,-1,-1,-1,-1],[-1,-1,1,-1,-1,-1,-1,-1,-1,-1],[-1,-1,-1,1,-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1,1,-1,-1],[-1,-1,-1,-1,-1,-1,-1,-1,1,-1],[-1,-1,-1,-1,-1,-1,-1,-1,-1,1]]
    trainIn=[]
    testIn=[]
    nGen=10
    nSong=90
    
    ###################################################################################
    # Generate Training Data:                                                         #
    # The following code iterates over songs in all 10 genres and calculates the      #
    # mfcc coefficients having 13 features.                                           #
    ################################################################################### 
    for genN in range(0,nGen):
        print 'Current Genre Processing: ' + str(genN)   # print current Genre 
        
        # For Training Data:
        # Iterate over all songs in current Genre
        for numSong in range(0,nSong):
            # songIn: Current song train data
            songIn=[]   
            
            # pathTemp: generate path of the music file
            pathTemp="genre/" + genrePath[genN] + "/" + genrePath[genN] + ".%05d" % numSong + ".wav"
            
            # Read musicfile to be used for generating train-data        
            sample_rate, X = scipy.io.wavfile.read(pathTemp)
            
            # Calculate mfcc coefficients of the music file
            ceps, mspec, spec = mfcc(X)
            row,col=ceps.shape
        
            # Take mean of all mfcc coef
            # The test vector contains 13 features for each song
	    ootlist = []
	    for xrow in ceps:
			flag = 0
			for xcol in xrow :
				if not(np.isfinite(xcol)) :
					flag = 1
					break
			if(flag == 0) :
				ootlist.append(xrow.tolist())
	    ootlist=np.array([np.array(xi) for xi in ootlist])
	    ceps =np.mean(ootlist, axis=0)
            #ceps=np.nanmean(ceps, axis=0)
            songIn=ceps.tolist()
            
            
            # Append the ouput label for the particular Genre
            songIn=songIn+genreLabel[genN]
            
            # Put the current song features in the Feature-Vector-Matrix
            trainIn.append(songIn)
        
        # For Testing Data:
        # Iterate over all songs in current Genre
        for numSong in range(nSong,nSong+10):
            # songIn: Current song train data
            songIn=[]   
            
            # pathTemp: generate path of the music file
            pathTemp="genre/"+genrePath[genN] + "/" + genrePath[genN] + ".%05d" % numSong + ".wav"
            
            # Read musicfile to be used for generating train-data        
            sample_rate, X = scipy.io.wavfile.read(pathTemp)
            
            # Calculate mfcc coefficients of the music file
            ceps, mspec, spec = mfcc(X)
            row,col=ceps.shape
        
            # Take mean of all mfcc coef
            # The test vector contains 13 features for each song
            ceps=np.nanmean(ceps, axis=0)
            songIn=ceps.tolist()
            
            # Append the ouput label for the particular Genre to be used for Verification
            songIn=songIn+genreLabel[genN]
            
            # Put the current Testing features in the Test-Matrix
            testIn.append(songIn)
            
    # Save the current training feature vector matrix to csv file     
    np.savetxt("trainData.csv", trainIn, delimiter=",")
    np.savetxt("testData.csv", testIn, delimiter=",")
