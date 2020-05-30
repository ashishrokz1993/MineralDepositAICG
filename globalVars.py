'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
'''
## This module contains all the global variables used in the program and should never import anything!!!!!!!!!!!!!!!!!!
## General Stuff
NoDataValue = -1  # Make sure this cannot be a value if there is a data available
xsize = 1 # Size of block in x direction
ysize = 1 # Size of block in y direction
zsize = 1 # Size of block in z direction

## NeuralNetworkStuff
trainPolicy=False # Option suggeesting if to train the policy again or not
working2D = True # If true then work on 2D otherwise will work for 3D. Might have to fix some bugs for 3D
if trainPolicy and working2D:
    sectionalZ =4 # Define the section of 2d to train on
else:
    sectionalZ =16 # define the section to test on
if trainPolicy:
    keepprob = 0.9 # Tried using this but didn't work
else:
    keepprob=1
## Neural Network Parameter
TrainingRandomPaths = 100
TestingRandomPaths = 1
rewardNormalization = 1
valuenormalization = 1
sensorrewardnormalization = 10
ActionMaxRange = 1
normalizationSimError =1

## Deposit simulation stuff
DataFile = 'StanfordReservoir'
HardDataPercentage = 0.04
SoftDataPercentage = 0.2
SensorDataPercentage = 0.4
VarianceSoft = 0.1
VarianceHard = 0.05
VarianceSensor = 0.15
IsHardDataAvailable = True
IsHardDataRegular = True
IsSimulationsAvailable = True

## Deposit updating stuff
UseSoftDataUpdating = True
UseSensorDataUpdating = True
IsSofDataAvailable = False
IsSoftDataRegular = True
IsSamplingSoftDataRegularGridFashion = False   ## Only turn this false if you don't want evenly spaced samples
IsSamplingHardDataRegularGridFashion = True
IsSensorDataAvailable = False # If False will generate sensor data from the true image.
similarityCheckUpdatingReplicates = True # If true will check if the replicate is similar to the data event with the use of variance
compareCenteralsoUpdatingReplicates = False  ## Something to try, but not so sure if this is good
HardDataAccuracy = 0.95 # Accuracy of hard data
SoftDataAccuracy = 0.55 # Accuracy of soft data
SensorDataAccuracy = 0.4 # Accuracy of sensor data
LegendrePolynomialOrderUpdating = 10 # Order of polynomial to approximate the cpdf

## Search stuff Simulation
LegendrePolynomialOrder = 15
SearchRadius =10
maxConditioningData = 10
numMinReplicates = 20
compareCenteralso = False
toleranceLag = 0.8
varianceTolerance = 1


## Search stuff Updating
searchRadiusUpdatingSim = 5
searchRadiusUpdatingHardSamples = 15
searchRadiusUpdatingSoftSamples = 10
maxSamplesUpdatingDataEvent = 8
maxSamplesUpdatingDataEventSoft = int(round(maxSamplesUpdatingDataEvent*SoftDataPercentage))
maxSamplesInSoftData = maxSamplesUpdatingDataEvent*50
maxSamplesInHardData = maxSamplesInSoftData*50
toleranceLagUpdatingSoft = 0.1
toleranceLagUpdatingHard = 0.5
correctPDF = True
debug = False

# Training Parameters
numSimTrain = 20
numSensors = 2
senseColumnName = ['Sensor'+str(i) for i in range(numSensors) ]
numBlocksY = 10
numBlocksX = 10
numBlockZ = 0
centerNodeLocation = (((numBlocksX*2)+1)*numBlocksY)+numBlocksX # this will be 220 if we start from 0 until 440 then 220 is the center node. Also for the slicing we need 220 elements each side
batchSize = 500
sensorLocationsToPlot = 2600
DecayIter = 100000

#Testing Parameters
updateNumber = 0
numberUpdates = 5
numSimTest = 20

## Visualization RL stuff
whichIndexToDisplay = 500
TotalImages=TrainingRandomPaths
ImagesGifindex = 5

