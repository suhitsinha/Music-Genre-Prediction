# Music-Genre-Prediction

#################################################
#                                               #
#         MUSIC GENRE CLASSIFICATION            #
#                                               #
#################################################

directory structure : 
=====================
  .
  |--main.py
  |--generateData.py
  |--trainData.py
  |--testSingle.py
  |--testBatch.py



================================
Programming language : python2.7
================================

Extra package requirements : 
============================
  1. scikit-learn 0.17.1.
  2. PyBrain v0.3 for neural networks. Official installation guide : 'https://github.com/pybrain/pybrain/wiki/installation'
  3. mfcc from scikit for MFCC features extraction from audio files [scikits.talkbox.features].
  4. numpy

==============================
Main executable file : main.py

Note : The training must run before any testing occurs ( option 1 from the supplied user interface when main.py is run).

Command: python main.py
==============================

Sample Run example :
-------------------
	============================================================
	Train Data: Press 1
	Test on Single Music file[.wav type music file]: Press 2
	Test on Batch Data: Press 3
	Exit: Press 4
	============================================================
	Provide the choice: 1
	Genrerate new data from the available training data set? [Y/N]: Y

	Enter the number of iterations: 2000
	Enter number of hidden nodes: 30

	Training Started
	......
	......
	Train Completed

	============================================================
	Train Data: Press 1
	Test on Single Music file[.wav type music file]: Press 2
	Test on Batch Data: Press 3
	Exit: Press 4
	============================================================
	Provide the choice: 2
	Enter full Path name [User inputs path for .wma file] : /home/suhit/music1.wav
	Predicted Genre: Classical

	============================================================
	Train Data: Press 1
	Test on Single Music file[.wav type music file]: Press 2
	Test on Batch Data: Press 3
	Exit: Press 4
	============================================================

	Provide the choice: 3
	....
	.... [Batch Information]
	....
	============================================================

	Train Data: Press 1
	Test on Single Music file[.wav type music file]: Press 2
	Test on Batch Data: Press 3
	Exit: Press 4
	============================================================
	Provide the choice: 4
	Exit



