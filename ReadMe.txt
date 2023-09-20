Copyright (c) 2020, Ashish Kumar. All rights reserved. 
This code is for updating geostatiscially simulated models with incoming noisy spatial and temporal sensor production data.
To generate spatial and temporal sensor data: Run the program with options isSensordataavailable and issoftdataavailable in globalVars as False. This will generate the data
To train and test the agent switch this option to True and run the program again. 

The program contains the following python scripts:
Adam - Has the classes and functions for Adam optimizer
DDPG - Has the classes and functions for DDPG algorithm
ddpg_learner - Has the functions training the agent by interacting with the environment
GenerateData - Has the  functions to generate soft spatial and temporal sensor data
globalVars - Contains all the constants used in the program
Grid - Has the classes and functions managing and updaitng the data
log_File -Has the classes and functions generating a log file
main - Main script is the one that is executed in the begining
Memory - Has the classes and functions for storing and sampling the training data during DDPG algorithm
Models - Has the classes and functions for generating agents in DDPG algorithm
Networks - Has the functions to generate parts of the agents in DDPG algorithm
Noise - Has the classes and functions for the noise process used during training of the agents
ReadData - Has the functions for reading the data
ResultPlotting - Has the functions for plotting results
Utilities - Has the functions for computing reward
Validation - Has the functions for validating the histogram of simulated models
Visualize - Has the  functions to generate GIF for real-time display
Simulations - Has the method to generate initial simualations


Required Libraries:
Tensorflow GPU version 1.14
Pandas
Numpy
Scipy
Copy
Matplotlib
Operator
Sortedcontainers
Progressbar
OS
Time
Random
Functools
Sys
Shutil
Json
Datetime
Tempfile
Collections
Contextlib
MPL
PIL
Imageio
