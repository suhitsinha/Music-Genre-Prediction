#######################################################
# Main Python Script : main.py                        #
# Usage: Follow the userfriendly interface provided   #
#######################################################

from generateData import generateData
from testBatch import testBatch
from testSingle import testSingle
from trainData import trainData

ch = 1
while ch != 4:
    print "=========================================================="
    print "Train Data: Press 1"
    print "Test on Single Music file[.wav type music file]: Press 2"
    print "Test on Batch Data: Press 3"
    print "Exit: Press 4"
    print "=========================================================="

    ch = input('Provide the choice: ')
    
    if ch == 1:     # Train Data

	#################################################################
	# The following three lines are used to generate the MFCC	#
	# coefficients from input audio data. Since our data is already #
	# converted to files "trainData.csv" & "testData.csv" , we have #
	# commented this out. To generate coefficients from new data 	#
	# run these lines. 						#
	#################################################################

        chGenData = raw_input('Genrerate new data from the available training data set? [Y/N]: ')
        if chGenData == 'Y' or chGenData == 'y':
            generateData()

        chIteration = input('Enter the number of iterations: ')
        #chHid = input('Enter the number of hidden node: ')
        chHid=20		#number of nodes in the hidden layer
        trainData(chIteration,chHid)
        
    elif ch == 2:    # Test Single Data
        chPath = raw_input('Enter full Path name [User inputs path for .wav file]: ')
        testSingle(chPath)

    elif ch == 3:   # Test on Batch Data
        testBatch()

    elif ch == 4:   # Exit
        print 'Exit'
        
    else:
        print 'Please enter the correct input choice'
