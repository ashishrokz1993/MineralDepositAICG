'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module generates the spatial and temporal sensor data
'''
import numpy as np
import globalVars as gv
import Grid
import pandas as pd
import random
import matplotlib.pyplot as plt
import log_File
def generateSoftData(Index,XYZ,DictionaryGrid,percentage,name):
    size = len(Index)
    s1, s2 = np.split(np.random.permutation(Index), [int(np.round(percentage*size))])
    output = []
    Cords = []
    ID = []
    SampleValue = []
    for i in s1:
        xselected = XYZ[i].x
        yselected =XYZ[i].y
        zselected = XYZ[i].z
        sampledValue = XYZ[i].exhausitiveImage
        id = XYZ[i].id
        Cords.append([xselected,yselected,zselected])
        ID.append(id)
        SampleValue.append(sampledValue)
        accuracy = np.random.uniform(gv.SoftDataAccuracy-gv.VarianceSoft,gv.SoftDataAccuracy+gv.VarianceSoft,1)[0]
        output.append([xselected,yselected,zselected,id,sampledValue,accuracy])
        key = (xselected,yselected,zselected)
        indexID = DictionaryGrid.dictionary.get(key,-99)
        if indexID==-99:
            log_File.info('Soft data out of bounds')
        else:
            DictionaryGrid.softData[indexID]=sampledValue
            DictionaryGrid.softDataAccuracy[indexID]=accuracy
    saveSampleData(output,name,percentage)
def generateSoftDataRG(Index,XYZ,DictionaryGrid,percentage,name):
    size = len(Index)
    output = []
    Cords = []
    ID = []
    SampleValue = []
    keys = np.array(DictionaryGrid.dictionary.keys())
    xmin = np.min(keys[:,0])
    xmax = np.max(keys[:,0])
    ymin = np.min(keys[:,1])
    ymax = np.max(keys[:,1])
    zmin = np.min(keys[:,2])
    zmax = np.max(keys[:,2])
    sliceX = int(np.sqrt(xmax/(xmax*gv.SoftDataPercentage)))
    sliceY = int(np.sqrt(ymax/(ymax*gv.SoftDataPercentage)))
    sliceZ = int(np.sqrt(zmax/(zmax*gv.SoftDataPercentage)))
    if gv.trainPolicy:
        xIndex = np.arange(xmin,xmax+1,sliceX)
        yIndex = np.arange(ymin,ymax+1,sliceY)
        zIndex = np.arange(zmin,zmax+1,sliceZ)
    else:
        xIndex = np.arange((xmin+(xmax/gv.numberUpdates)*(gv.updateNumber)), ((xmax/gv.numberUpdates)*(gv.updateNumber+1)), sliceX)
        yIndex = np.arange((ymin+(ymax/gv.numberUpdates)*(gv.updateNumber)), ((ymax/gv.numberUpdates)*(gv.updateNumber+1)), sliceY)
        zIndex = np.arange((zmin+(zmax/gv.numberUpdates)*(gv.updateNumber)), ((zmax/gv.numberUpdates)*(gv.updateNumber+1)), sliceZ)   
    if gv.working2D:
        for i in xIndex:
            for j in yIndex:
                ind = DictionaryGrid.dictionary.get((i,j,gv.sectionalZ),-99)
                if ind==-99:
                    log_File.info('Soft data out of bound')
                else:
                    sampledValue = XYZ[ind].exhausitiveImage
                    id = ind
                    SampleValue.append(sampledValue)
                    accuracy = np.random.uniform(gv.SoftDataAccuracy-gv.VarianceSoft,gv.SoftDataAccuracy+gv.VarianceSoft,1)[0]
                    Cords.append([i,j,gv.sectionalZ])
                    output.append([i,j,gv.sectionalZ,id,sampledValue, accuracy])
                    DictionaryGrid.softData[ind]=sampledValue
                    DictionaryGrid.softDataAccuracy[ind]=accuracy
    else:
        for i in xIndex:
            for j in yIndex:
                for k in zIndex:
                    ind = DictionaryGrid.dictionary.get((i,j,k),-99)
                    if ind==-99:
                        log_File.info('Soft data out of bound')
                    else:
                        sampledValue = XYZ[ind].exhausitiveImage
                        id = ind
                        SampleValue.append(sampledValue)
                        Cords.append([i,j,k])
                        accuracy = np.random.uniform(gv.SoftDataAccuracy-gv.VarianceSoft,gv.SoftDataAccuracy+gv.VarianceSoft,1)[0]
                        output.append([i,j,k,id,sampledValue, accuracy])
                        DictionaryGrid.softDataAccuracy[ind]=accuracy
                        DictionaryGrid.softData[ind]=sampledValue
    saveSampleData(output,name,percentage)
def generateSensorData(Index,XYZ,DictionaryGrid,percentage,name):
    size = len(Index)
    s1, s2 = np.split(np.random.permutation(Index), [int(np.round(percentage*size))])
    output = []
    Cords = []
    ID = []
    SampleValue = []
    NewShape = s1.shape[0]-s1.shape[0]%gv.numSensors
    s1New = s1[:NewShape]
    for i in s1New:
        xselected = XYZ[i].x
        yselected =XYZ[i].y
        zselected = XYZ[i].z
        sampledValue = XYZ[i].exhausitiveImage
        id = XYZ[i].id
        Cords.append([xselected,yselected,zselected])
        ID.append(id)
        SampleValue.append(sampledValue)
        key = (xselected,yselected,zselected)
        indexID = DictionaryGrid.dictionary.get(key,-99)
        if indexID==-99:
            log_File.info('Sensor data out of bounds')
        else:
            DictionaryGrid.DitionarySensorLocation[indexID]=-99
            DictionaryGrid.sensorAllData[indexID]=sampledValue
        output.append([xselected,yselected,zselected,id,sampledValue])
    saveSampleData(output,name+'SensorAll',percentage)
    IndexSensor = np.split(np.random.permutation(s1New),gv.numSensors)
    SensorZeroLocation = []
    AllLocationSensorData = [[] for i in range(gv.numSensors)]
    for sensorIndex in range(gv.numSensors):
        Cords = []
        output = []
        ID = []
        SampleValue = []
        counter = 0
        for i in IndexSensor[sensorIndex]:
            xselected = XYZ[i].x
            yselected =XYZ[i].y
            zselected = XYZ[i].z
            sampledValue = XYZ[i].exhausitiveImage
            id = XYZ[i].id
            Cords.append([xselected,yselected,zselected])
            ID.append(id)
            SampleValue.append(sampledValue)
            key = (xselected,yselected,zselected)
            indexID = DictionaryGrid.dictionary.get(key,-99)
            if indexID==-99:
                log_File.info('Sensor data out of bounds')
            else:
                AllLocationSensorData[sensorIndex].append(indexID)
                if sensorIndex==0:
                    DictionaryGrid.locationSensorData[sensorIndex][indexID]=indexID
                    SensorZeroLocation.append(indexID)
                else:
                    DictionaryGrid.locationSensorData[sensorIndex][SensorZeroLocation[counter]]=indexID
                    counter+=1
                acc = np.random.uniform(gv.SensorDataAccuracy-gv.VarianceSensor,gv.SensorDataAccuracy+gv.VarianceSensor,1)[0]
                DictionaryGrid.sensorIndividualDataAccuracy[sensorIndex][indexID]=acc
                DictionaryGrid.sensorIndividualData[sensorIndex][indexID]=sampledValue
                DictionaryGrid.sensorAllDataAccuracy[indexID]=acc
            output.append([xselected,yselected,zselected,id,sampledValue,acc])
        saveSampleData(output,name+'Sensor'+str(sensorIndex+1),percentage)
    AllLocationSensorData = np.array(AllLocationSensorData)
    for j in range(AllLocationSensorData.shape[0]):
        for i in range(AllLocationSensorData.shape[1]):
            foundIndex = DictionaryGrid.DitionarySensorLocation.get(AllLocationSensorData[j][i],-999)
            if foundIndex==-999:
                print('Error in creating new dictionary of sensor location')
                exit()
            else:
                valslocations = AllLocationSensorData[:,i]
                valslocations = np.delete(valslocations,j)
                DictionaryGrid.DitionarySensorLocation[AllLocationSensorData[j][i]] = valslocations
def sampleHardData(Index,XYZ,percentage,name,transformer):
    size = len(Index)
    s1, s2 = np.split(np.random.permutation(Index), [int(np.round(percentage*size))])
    output = []
    Cords = []
    ID = []
    SampleValue = []
    for i in s1:
        xselected = XYZ[i].x
        yselected =XYZ[i].y
        zselected = XYZ[i].z
        sampledValue = XYZ[i].exhausitiveImage
        id = XYZ[i].id
        Cords.append([xselected,yselected,zselected])
        ID.append(id)
        SampleValue.append(sampledValue)
        output.append([xselected,yselected,zselected,id,sampledValue])
    if gv.UseTrainingImageSimulation:
        transformedVals = transformer.transform(np.array(SampleValue).reshape(-1,1))       
    else:
        transformedVals = transformer.fit_transform(np.array(SampleValue).reshape(-1,1))
    transformedVals = transformedVals.reshape(1,-1)[0]
    counter=0
    for i in s1:
        if name=='HardSamples':
            XYZ[i].harddataValue=transformedVals[counter]
            XYZ[i].harddata=True
            XYZ[i].simVal = transformedVals[counter]
        counter+=1
    saveSampleData(output,name,percentage)
def sampleHardDataRG(Index,XYZ,DictionaryGrid,percentage,name,transformer):
    size = len(Index)
    output = []
    Cords = []
    ID = []
    SampleValue = []
    keys = np.array(DictionaryGrid.dictionary.keys())
    xmin = np.min(keys[:,0])
    xmax = np.max(keys[:,0])
    ymin = np.min(keys[:,1])
    ymax = np.max(keys[:,1])
    zmin = np.min(keys[:,2])
    zmax = np.max(keys[:,2])
    sliceX = int(np.sqrt(xmax/(xmax*gv.HardDataPercentage)))
    sliceY = int(np.sqrt(ymax/(ymax*gv.HardDataPercentage)))
    sliceZ = int(np.sqrt(zmax/(zmax*gv.HardDataPercentage)))
    if gv.trainPolicy:
        xIndex = np.arange(xmin,xmax+1,sliceX)
        yIndex = np.arange(ymin,ymax+1,sliceY)
        zIndex = np.arange(zmin,zmax+1,sliceZ)
    else:
        xIndex = np.arange((xmin+(xmax/gv.numberUpdates)*(gv.updateNumber)), ((xmax/gv.numberUpdates)*(gv.updateNumber+1)), sliceX)
        yIndex = np.arange((ymin+(ymax/gv.numberUpdates)*(gv.updateNumber)), ((ymax/gv.numberUpdates)*(gv.updateNumber+1)), sliceY)
        zIndex = np.arange((zmin+(zmax/gv.numberUpdates)*(gv.updateNumber)), ((zmax/gv.numberUpdates)*(gv.updateNumber+1)), sliceZ)   
    if gv.working2D:
        for i in xIndex:
            for j in yIndex:
                ind = DictionaryGrid.dictionary.get((i,j,gv.sectionalZ),-99)
                if ind==-99:
                    log_File.info('hard data out of bound')
                else:
                    sampledValue = XYZ[ind].exhausitiveImage
                    id = ind
                    ID.append(id)
                    SampleValue.append(sampledValue)
                    accuracy = np.random.uniform(gv.HardDataAccuracy-gv.VarianceHard,gv.HardDataAccuracy+gv.VarianceHard,1)[0]
                    Cords.append([i,j,gv.sectionalZ])
                    output.append([i,j,gv.sectionalZ,id,sampledValue, accuracy])
                    if gv.IsSimulationsAvailable:
                        DictionaryGrid.hardData[ind]=sampledValue
                        DictionaryGrid.hardDataAccuracy[ind]=accuracy
    else:
        for i in xIndex:
            for j in yIndex:
                for k in zIndex:
                    ind = DictionaryGrid.dictionary.get((i,j,k),-99)
                    if ind==-99:
                        log_File.info('hard data out of bound')
                    else:
                        sampledValue = XYZ[ind].exhausitiveImage
                        id = ind
                        SampleValue.append(sampledValue)
                        Cords.append([i,j,k])
                        accuracy = np.random.uniform(gv.HardDataAccuracy-gv.VarianceHard,gv.HardDataAccuracy+gv.VarianceHard,1)[0]
                        output.append([i,j,k,id,sampledValue, accuracy])
                        if gv.IsSimulationsAvailable:
                            DictionaryGrid.hardDataAccuracy[ind]=accuracy
                            DictionaryGrid.hardData[ind]=sampledValue
    if gv.UseTrainingImageSimulation:
        transformedVals = transformer.transform(np.array(SampleValue).reshape(-1,1))
    else:
        transformedVals = transformer.fit_transform(np.array(SampleValue).reshape(-1,1))
    transformedVals = transformedVals.reshape(1,-1)[0]
    for i in range(len(ID)):
        if gv.UseHardDataUpdating:
            XYZ[ID[i]].harddataValue=SampleValue[i]
            XYZ[ID[i]].harddata=True
            XYZ[ID[i]].simVal = SampleValue[i]
        else:
            XYZ[ID[i]].harddataValue=transformedVals[i]
            XYZ[ID[i]].harddata=True
            XYZ[ID[i]].simVal = transformedVals[i]
    saveSampleData(output,name,percentage)
def saveSampleData(sampledata,name,percentage):
    if name !='HardSamples' and name!='HardSamplesRG':
        savepath = 'Data/'+name+gv.DataFile+str(int(percentage*100))+'PercentUpdate'+str(gv.updateNumber)+'Section'+str(gv.sectionalZ)
    else:
        savepath = 'Data/'+name+gv.DataFile+str(int(percentage*100))+'Percent'+'Section'+str(gv.sectionalZ)
    Outputfile = pd.DataFrame(sampledata)
    if name!='SamplesSensorAll':
        Outputfile.columns=['X','Y','Z','Index','Val', 'Accuracy']
    else:
        Outputfile.columns=['X','Y','Z','Index','Val']
    Outputfile.to_csv(savepath+'.csv',index=False)

