'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module is used for saving the initial and updated simulation files.
It also contains the funciton for generating the cumulant maps required for validating the simulations and the updating process
It also has a funciton to visualize the sensor vs model prediciton during training and testing
'''
import numpy as np
import globalVars as gv
import pandas as pd
import matplotlib.pyplot as plt
import os
import log_File
from mpl_toolkits.mplot3d import Axes3D
import os


def outputSchedule(blocks,updatenumber,simnumber,transformer):
    print('Saving the simulation files')
    filename = 'Output/Update'+str(updatenumber)+'/Section'+str(gv.sectionalZ)+'/Sim'+str(simnumber)+'.csv'
    X = []
    Y = []
    Z = []
    Value = []
    for blocknumber in blocks:
        X.append(blocknumber.x)
        Y.append(blocknumber.y)
        Z.append(blocknumber.z)
        Value.append(blocknumber.simVal)
    Valuesarray = np.array(Value)
    ValInverse = transformer.inverse_transform(Valuesarray.reshape(-1,1))
    allvals = np.transpose([X,Y,Z,ValInverse.reshape(1,-1)[0]])
    OuputResult = pd.DataFrame(allvals)
    OuputResult.columns=['X','Y','Z','Val']
    OuputResult.to_csv(filename,index=False)
def outputhardData(DictionaryGrid):
    filename = 'Data/HardScaledSection'+str(gv.sectionalZ)+'.csv'
    masked = DictionaryGrid.hardData!=gv.NoDataValue
    masked2=masked[:-1]
    xyz = np.array(DictionaryGrid.Tree.data[masked2])
    allvals = np.concatenate((xyz.astype(int),np.array(DictionaryGrid.hardData[masked][:,np.newaxis])),axis=1)
    OuputResult = pd.DataFrame(allvals)
    OuputResult.columns=['X','Y','Z','ScaledValHard']
    OuputResult.to_csv(filename,index=False)
def outputsoftData(DictionaryGrid):
    filename = 'Data/SoftScaledSection'+str(gv.sectionalZ)+'.csv'
    masked = DictionaryGrid.softData!=gv.NoDataValue
    masked2=masked[:-1]
    xyz = np.array(DictionaryGrid.Tree.data[masked2])
    allvals = np.concatenate((xyz.astype(int),np.array(DictionaryGrid.softData[masked][:,np.newaxis])),axis=1)
    OuputResult = pd.DataFrame(allvals)
    OuputResult.columns=['X','Y','Z','ScaledValSoft']
    OuputResult.to_csv(filename,index=False)
def outputUdpatedSims(Block,updatenumber,simnumber):
    print('Saving the simulation files to folder Update', gv.updateNumber+1)
    filename = 'Output/Update'+str(updatenumber+1)+'/Section'+str(gv.sectionalZ)+'/Sim'+str(simnumber)+'.csv'
    X = []
    Y = []
    Z = []
    Value = []
    for blocknumber in blocks:
        X.append(blocknumber.x)
        Y.append(blocknumber.y)
        Z.append(blocknumber.z)
        Value.append(blocknumber.simVal)
    Valuesarray = np.array(Value)
    allvals = np.transpose([X,Y,Z,Value])
    OuputResult = pd.DataFrame(allvals)
    OuputResult.columns=['X','Y','Z','Val']
    OuputResult.to_csv(filename,index=False)
def UdpatedSims(GridDictionary,updatenumber,simnumber):
    print('Saving the simulation files to folder Update', gv.updateNumber+1)
    filename = 'Output/Update'+str(updatenumber+1)+'/Section'+str(gv.sectionalZ)+'/Sim'+str(simnumber)+'.csv'
    xyz = np.array(GridDictionary.Tree.data)
    #Valuesarray = np.array([GridDictionary.simulationUpdating[simnumber]])
    allvals = np.concatenate((xyz,np.array([GridDictionary.simulationUpdating[simnumber][:-1]]).T),axis=1)
    OuputResult = pd.DataFrame(allvals)
    OuputResult.columns=['X','Y','Z','Val']
    OuputResult.to_csv(filename,index=False)
# Plot sensor vs model prediciton error graphs for initial simulations
def outputSensorVsModelPredictionGraphIndividualSimInitial(DictionaryGrid,simnumber,location):
    sensorPredicActual = DictionaryGrid.generateSensorPrediction(DictionaryGrid.locationSensorData)
    modelPrediction = DictionaryGrid.generateModelPrediction(DictionaryGrid.locationSensorData,simnumber,'Initial')
    ErrorSensorPrediction = DictionaryGrid.generateSensorError(DictionaryGrid.locationSensorData)
    Days = np.arange(1,len(sensorPredicActual[sensorPredicActual!=gv.NoDataValue])+1)
    withoutNegativeSensorPrediction = sensorPredicActual[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeModelPrediction = modelPrediction[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeSensorError = ErrorSensorPrediction[sensorPredicActual!=gv.NoDataValue]
    mean = withoutNegativeSensorPrediction
    upper = withoutNegativeSensorPrediction+withoutNegativeSensorError*withoutNegativeSensorPrediction
    lower = withoutNegativeSensorPrediction-withoutNegativeSensorError*withoutNegativeSensorPrediction
    data_to_plot = np.concatenate(([upper],[mean],[lower]),axis=0)
    #location = np.random.randint(0,Days.shape[0],gv.sensorLocationsToPlot)
    plt.plot(Days[:gv.sensorLocationsToPlot],upper[location], label='SensorUpper',linestyle=':',color='green',zorder=3)
    plt.plot(Days[:gv.sensorLocationsToPlot],mean[location],label='SensorActual',color='green',zorder=1)
    plt.plot(Days[:gv.sensorLocationsToPlot],lower[location],label='SensorLower',linestyle=':',color='green',zorder=2)
    plt.plot(Days[:gv.sensorLocationsToPlot],withoutNegativeModelPrediction[location], label='Model', color='blue',zorder=4)
    #plt.xticks(np.arange(1, np.max(Days)+1, 120))
    #plt.boxplot(data_to_plot[:,location],showmeans=False, notch=False)
    plt.ylabel('Prediction', fontsize=15)
    plt.xlabel('Minutes', fontsize=15)
    plt.title('PredictionError: InitialSimulation', fontsize=20)
    plt.legend()
    plt.savefig('Output/TrainResult/InitialSection'+str(gv.sectionalZ)+'Sim'+str(simnumber)+'Sensor.png')
    plt.clf()
    plt.close()
def outputSensorPredictionGraph(DictionaryGrid):
    sensorPredicActual = DictionaryGrid.generateSensorPrediction(DictionaryGrid.locationSensorData)
    ErrorSensorPrediction = DictionaryGrid.generateSensorError(DictionaryGrid.locationSensorData)
    Days = np.arange(1,len(sensorPredicActual[sensorPredicActual!=gv.NoDataValue])+1)
    withoutNegativeSensorPrediction = sensorPredicActual[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeSensorError = ErrorSensorPrediction[sensorPredicActual!=gv.NoDataValue]
    mean = withoutNegativeSensorPrediction
    upper = withoutNegativeSensorPrediction+withoutNegativeSensorError*withoutNegativeSensorPrediction
    lower = withoutNegativeSensorPrediction-withoutNegativeSensorError*withoutNegativeSensorPrediction
    data_to_plot = np.concatenate(([upper],[mean],[lower]),axis=0)
    location = Days-1
    location = location[:gv.sensorLocationsToPlot]
    #location = location[np.argsort(mean[location])]
    #location = np.random.randint(0,Days.shape[0],gv.sensorLocationsToPlot)
    plt.plot(Days[:gv.sensorLocationsToPlot],upper[location], label='SensorUpper',linestyle=':',color='green',zorder=3)
    plt.plot(Days[:gv.sensorLocationsToPlot],mean[location],label='SensorActual',color='green',zorder=1)
    plt.plot(Days[:gv.sensorLocationsToPlot],lower[location],label='SensorLower',linestyle=':',color='green',zorder=2)
    #plt.plot(Days[:gv.sensorLocationsToPlot],withoutNegativeModelPrediction[location], label='Model', color='blue',zorder=4)
    #plt.xticks(np.arange(1, np.max(Days)+1, 120))
    #plt.boxplot(data_to_plot[:,location],showmeans=False)
    plt.ylabel('Prediction', fontsize=15)
    plt.xlabel('Minutes', fontsize=15)
    plt.title('Sensor Observations Unsorted', fontsize=20)
    plt.legend()
    plt.savefig('Output/TrainResult/UnsortedSection'+str(gv.sectionalZ)+'Sensor.png')
    plt.clf()
    plt.close()
    location = location[np.argsort(mean[location])]
    plt.plot(Days[:gv.sensorLocationsToPlot],upper[location], label='SensorUpper',linestyle=':',color='green',zorder=3)
    plt.plot(Days[:gv.sensorLocationsToPlot],mean[location],label='SensorActual',color='green',zorder=1)
    plt.plot(Days[:gv.sensorLocationsToPlot],lower[location],label='SensorLower',linestyle=':',color='green',zorder=2)
    plt.ylabel('Prediction', fontsize=15)
    plt.xlabel('Minutes', fontsize=15)
    plt.title('Sensor Observations Sorted', fontsize=20)
    plt.legend()
    plt.savefig('Output/TrainResult/SortedSection'+str(gv.sectionalZ)+'Sensor.png')
    plt.clf()
    plt.close()
    return location
# Plot sensor vs model prediciton error graphs for updated simualtions
def outputSensorVsModelPredictionGraphIndividualSimUpdatedTrain(DictionaryGrid,simnumber,epochnumber,location):
    sensorPredicActual = DictionaryGrid.generateSensorPrediction(DictionaryGrid.locationSensorData)
    modelPrediction = DictionaryGrid.generateModelPrediction(DictionaryGrid.locationSensorData,simnumber,'Updated')
    ErrorSensorPrediction = DictionaryGrid.generateSensorError(DictionaryGrid.locationSensorData)
    Days = np.arange(1,len(sensorPredicActual[sensorPredicActual!=gv.NoDataValue])+1)
    withoutNegativeSensorPrediction = sensorPredicActual[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeModelPrediction = modelPrediction[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeSensorError = ErrorSensorPrediction[sensorPredicActual!=gv.NoDataValue]
    mean = withoutNegativeSensorPrediction
    upper = withoutNegativeSensorPrediction+withoutNegativeSensorError*withoutNegativeSensorPrediction
    lower = withoutNegativeSensorPrediction-withoutNegativeSensorError*withoutNegativeSensorPrediction
    data_to_plot = np.concatenate(([upper],[mean],[lower]),axis=0)
    plt.plot(Days[:gv.sensorLocationsToPlot],upper[location], label='SensorUpper',linestyle=':',color='green',zorder=3)
    plt.plot(Days[:gv.sensorLocationsToPlot],mean[location],label='SensorActual',color='green',zorder=1)
    plt.plot(Days[:gv.sensorLocationsToPlot],lower[location],label='SensorLower',linestyle=':',color='green',zorder=2)
    plt.plot(Days[:gv.sensorLocationsToPlot],withoutNegativeModelPrediction[location], label='Model', color='blue',zorder=4)
    #plt.xticks(np.arange(1, np.max(Days)+1, 120))
    #plt.boxplot(data_to_plot[:,location],showmeans=False)
    plt.ylabel('Prediction',fontsize=15)
    plt.xlabel('Minutes', fontsize=15)
    plt.title('PredictionError: UpdatedSimulationTrain', fontsize=20)
    plt.legend()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(epochnumber)+'Sim'+str(simnumber)+'Sensor.png')
    plt.clf()
    plt.close()
def outputSensorVsModelPredictionGraphIndividualSimUpdated(DictionaryGrid,simnumber,location):
    sensorPredicActual = DictionaryGrid.generateSensorPrediction(DictionaryGrid.locationSensorData)
    modelPrediction = DictionaryGrid.generateModelPrediction(DictionaryGrid.locationSensorData,simnumber,'Updated')
    ErrorSensorPrediction = DictionaryGrid.generateSensorError(DictionaryGrid.locationSensorData)
    Days = np.arange(1,len(sensorPredicActual[sensorPredicActual!=gv.NoDataValue])+1)
    withoutNegativeSensorPrediction = sensorPredicActual[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeModelPrediction = modelPrediction[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeSensorError = ErrorSensorPrediction[sensorPredicActual!=gv.NoDataValue]
    mean = withoutNegativeSensorPrediction
    upper = withoutNegativeSensorPrediction+withoutNegativeSensorError*withoutNegativeSensorPrediction
    lower = withoutNegativeSensorPrediction-withoutNegativeSensorError*withoutNegativeSensorPrediction
    data_to_plot = np.concatenate(([upper],[mean],[lower]),axis=0)
    plt.plot(Days[:gv.sensorLocationsToPlot],upper[location], label='SensorUpper',linestyle=':',color='green',zorder=3)
    plt.plot(Days[:gv.sensorLocationsToPlot],mean[location],label='SensorActual',color='green',zorder=1)
    plt.plot(Days[:gv.sensorLocationsToPlot],lower[location],label='SensorLower',linestyle=':',color='green',zorder=2)
    plt.plot(Days[:gv.sensorLocationsToPlot],withoutNegativeModelPrediction[location], label='Model', color='blue',zorder=4)
    #plt.xticks(np.arange(1, np.max(Days)+1, 120))
    #plt.boxplot(data_to_plot[:,location],showmeans=False)
    plt.ylabel('Prediction',fontsize=15)
    plt.xlabel('Minutes', fontsize=15)
    plt.title('PredictionError: UpdatedSimulation', fontsize=20)
    plt.legend()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Sim'+str(simnumber)+'Sensor.png')
    plt.clf()
    plt.close()
# Plot sensor vs model prediciton error graphs for updated simualtions evaluation
def outputSensorVsModelPredictionGraphIndividualSimUpdatedEval(DictionaryGrid,simnumber,epochnumber,location):
    sensorPredicActual = DictionaryGrid.generateSensorPrediction(DictionaryGrid.locationSensorData)
    modelPrediction = DictionaryGrid.generateModelPrediction(DictionaryGrid.locationSensorData,simnumber,'Updated')
    ErrorSensorPrediction = DictionaryGrid.generateSensorError(DictionaryGrid.locationSensorData)
    Days = np.arange(1,len(sensorPredicActual[sensorPredicActual!=gv.NoDataValue])+1)
    withoutNegativeSensorPrediction = sensorPredicActual[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeModelPrediction = modelPrediction[sensorPredicActual!=gv.NoDataValue]
    withoutNegativeSensorError = ErrorSensorPrediction[sensorPredicActual!=gv.NoDataValue]
    mean = withoutNegativeSensorPrediction
    upper = withoutNegativeSensorPrediction+withoutNegativeSensorError*withoutNegativeSensorPrediction
    lower = withoutNegativeSensorPrediction-withoutNegativeSensorError*withoutNegativeSensorPrediction
    data_to_plot = np.concatenate(([upper],[mean],[lower]),axis=0)
    plt.plot(Days[:gv.sensorLocationsToPlot],upper[location], label='SensorUpper',linestyle=':',color='green',zorder=3)
    plt.plot(Days[:gv.sensorLocationsToPlot],mean[location],label='SensorActual',color='green',zorder=1)
    plt.plot(Days[:gv.sensorLocationsToPlot],lower[location],label='SensorLower',linestyle=':',color='green',zorder=2)
    plt.plot(Days[:gv.sensorLocationsToPlot],withoutNegativeModelPrediction[location], label='Model', color='blue',zorder=4)
    #plt.boxplot(data_to_plot[:,location],showmeans=False)
    plt.ylabel('Prediction',fontsize=15)
    plt.xlabel('Minutes', fontsize=15)
    plt.title('PredictionError: UpdatedSimulationEval', fontsize=20)
    plt.legend()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(epochnumber)+'UpdatedSim'+str(simnumber)+'Sensor.png')
    plt.clf()
    plt.close()
# Display the udpated simulation figure
def outputVisualizationSimUpdate(DictionaryGrid,epochNumber,simNumber):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.simulationUpdating[simNumber][:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Train Simulation',fontsize=30)       
        p = ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4, vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(epochNumber)+'Sim'+str(simNumber)+'.png')
    plt.clf()
    plt.close(fig)
def outputVisualizationSimTemp(DictionaryGrid,epochNumber,simNumber):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.simulationTempData[simNumber][:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Updated Simulation',fontsize=30)       
        p = ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4, vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(epochNumber)+'Sim'+str(simNumber)+'.png')
    plt.clf()
    plt.close(fig)
#Displays the initial simulation figure
def outputVisualizationSim(DictionaryGrid,simNumber):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.simulationData[simNumber][:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Initial Simulation',fontsize=30)     
        p = ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4,edgecolors='face', vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/InitialSection'+str(gv.sectionalZ)+'Sim'+str(simNumber)+'.png')
    plt.clf()
    plt.close(fig)
# Display the soft data figure
def outputVisualizationSoft(DictionaryGrid):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.softData[:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Soft Data',fontsize=30)      
        ax.scatter(xyz[:,0][vals==gv.NoDataValue],xyz[:,1][vals==gv.NoDataValue],c = 'white', marker='s',s=1, edgecolors='face')
        p=ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4,edgecolors='face', vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
        ax.set_title('Soft Data',fontsize=30) 
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Soft.png')
    plt.clf()
    plt.close(fig)
#Output updated simulations
def outputVisualizationSimAfterUpdate(DictionaryGrid,simNumber,number):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.simulationUpdating[simNumber][:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Evaluation Simulation',fontsize=30)       
        p = ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4, vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(number)+'UpdatedSim'+str(simNumber)+'.png')
    plt.clf()
    plt.close(fig)
#Output true image
def outputVisualizationTrueImage(DictionaryGrid,):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.trueImageData[:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Updated Simulation',fontsize=30)       
        p = ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4, vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'TrueImage'+'.png')
    plt.clf()
    plt.close(fig)
# Display the hard data figure
def outputVisualizationHard(DictionaryGrid):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.hardData[:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Hard Data',fontsize=30)      
        ax.scatter(xyz[:,0][vals==gv.NoDataValue],xyz[:,1][vals==gv.NoDataValue],c = 'white', marker='s',s=1, edgecolors='face')
        p=ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4,edgecolors='face', vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
        ax.set_title('Hard Data',fontsize=30) 
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'Hard.png')
    plt.clf()
    plt.close(fig)

def outputSimAlloneFile(GriDictionary,updatenumber, name=None):
    print('Saving the simulation files to folder Update', gv.updateNumber)
    filename = 'Output/Update'+str(updatenumber)+'/Section'+str(gv.sectionalZ)+'/SimAll'+name+'.csv'
    xyz = np.array(GriDictionary.Tree.data)
    if name=='Initial':
        allvals = np.concatenate((xyz,np.array(GriDictionary.simulationData[:,:-1]).T),axis=1)
    else:
        allvals = np.concatenate((xyz,np.array(GriDictionary.simulationUpdating[:,:-1]).T),axis=1)
    OuputResult = pd.DataFrame(allvals)
    OuputResult.to_csv(filename,index=False)
#Output updated simulations while testing
def outputVisualizationSimTest(DictionaryGrid,simNumber):
    fig = plt.figure()
    xyz = np.array(DictionaryGrid.Tree.data)
    vals = DictionaryGrid.simulationUpdating[simNumber][:-1]
    if gv.working2D:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_title('Updated Simulation',fontsize=30)       
        p = ax.scatter(xyz[:,0][vals!=gv.NoDataValue],xyz[:,1][vals!=gv.NoDataValue],c = vals[vals!=gv.NoDataValue], marker='s',s=4, vmin=0, vmax=gv.ActionMaxRange)
    else:
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c = vals, marker='s') 
        ax.set_xlabel('X', fontsize=20, rotation=150)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20, rotation=60)
    fig.colorbar(p)
    #plt.show()
    plt.savefig('Output/TrainResult/Section'+str(gv.sectionalZ)+'UpdatedSim'+str(simNumber)+'.png')
    plt.clf()
    plt.close(fig)