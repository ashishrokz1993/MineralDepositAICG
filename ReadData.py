'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This file is used for importing the data and calling relevant funcitons to generate sensor and soft data
If you want to generate simulations you might want to change some options in global variables to True. Please look globalVars.py file
'''
import numpy as np
import pandas as pd
import globalVars as gv
import Grid
import GenerateData
import log_File

def inputFiles():
    ###############################################################################################################################
    print('Reading grid file')
    Cord = pd.read_csv('Data/StandfordReservoir.csv')
    X = Cord['X'].as_matrix()[0:]
    Y = Cord['Y'].as_matrix()[0:]
    Z = Cord['Z'].as_matrix()[0:]
    Zfilter = Cord['Z'].as_matrix()[0:]
    SimCord = Cord[['X', 'Y', 'Z']].as_matrix()[0:]
    Porosity = Cord['Porosity'].as_matrix()[0:]
    if gv.working2D:
        print('Working on 2D')
        Xgrid = X[Zfilter==gv.sectionalZ]
        Ygrid = Y[Zfilter==gv.sectionalZ]
        Zgrid = Z[Zfilter==gv.sectionalZ]
        SimCordgrid = SimCord[Zfilter==gv.sectionalZ]  
        Porositygrid = Porosity[Zfilter==gv.sectionalZ]
    else:
        print('Working on 3D')
        Xgrid = Cord['X'].as_matrix()[0:]
        Ygrid = Cord['Y'].as_matrix()[0:]
        Zgrid = Cord['Z'].as_matrix()[0:]
        SimCordgrid = Cord[['X', 'Y', 'Z']].as_matrix()[0:, :]
        Porositygrid = Cord['Porosity'].as_matrix()[0:]
        
    GridNode = [Grid.block() for i in range(Xgrid.size)]
    if gv.IsSimulationsAvailable:
        DictionaryGrid = Grid.GridDictionary(SimCordgrid,None,'Grid')
    else:
        print('Generate simulations first')
    Index = []
    SensorIndex = []
    index = 0
    for node in GridNode:
        Index.append(index)
        node.x=Xgrid[index]
        node.y = Ygrid[index]
        node.z = Zgrid[index]
        node.exhausitiveImage = Porositygrid[index]
        node.id = index
        DictionaryGrid.dictionary[(node.x,node.y,node.z)]=index
        
        if gv.working2D:
            if (node.x<=(np.max(Xgrid)/gv.numberUpdates)*(gv.updateNumber+1) and node.x>=(np.max(Xgrid)/gv.numberUpdates)*(gv.updateNumber) and 
                node.y<=(np.max(Ygrid)/gv.numberUpdates)*(gv.updateNumber+1) and node.y>=(np.max(Ygrid)/gv.numberUpdates)*(gv.updateNumber)):
                SensorIndex.append(node.id)
            if (node.x-1)%(gv.xsize*gv.numBlocksX)==0 and (node.y-1)%(gv.ysize*gv.numBlocksY)==0 and node.x!=1 and node.y!=1:
                DictionaryGrid.evaluationrolloutIndex.append(node.id)
            if node.x>gv.xsize*gv.numBlocksX and node.y>gv.ysize*gv.numBlocksY and node.x<=(np.max(Xgrid)-gv.numBlocksX*gv.xsize) and node.y<=(np.max(Ygrid)-gv.numBlocksY*gv.ysize):
                DictionaryGrid.rolloutIndex.append(node.id)
        else:
            if (node.x<=(np.max(Xgrid)/gv.numberUpdates)*(gv.updateNumber+1) and node.x>=(np.max(Xgrid)/gv.numberUpdates)*(gv.updateNumber) and 
                node.y<=(np.max(Ygrid)/gv.numberUpdates)*(gv.updateNumber+1) and node.y>=(np.max(Ygrid)/gv.numberUpdates)*(gv.updateNumber) and 
                node.z<=(np.max(Zgrid)/gv.numberUpdates)*(gv.updateNumber+1) and node.z>=(np.max(Zgrid)/gv.numberUpdates)*(gv.updateNumber)):
                SensorIndex.append(node.id)
            if node.x>gv.xsize*gv.numBlocksX and node.y>gv.ysize*gv.numBlocksY and node.z>gv.numBlockZ*gv.zsize and node.x<=(np.max(Xgrid)-gv.numBlocksX*gv.xsize) and node.y<=(np.max(Ygrid)-gv.numBlocksY*gv.ysize) and node.z<=(np.max(Zgrid)-gv.numBlockZ*gv.zsize):
                DictionaryGrid.rolloutIndex.append(node.id)
                if (node.x-1)%(gv.xsize*gv.numBlocksX)==0 and (node.y-1)%(gv.ysize*gv.numBlocksY)==0 and (node.z-1)%(gv.zsize*gv.numBlockZ)==0:
                    DictionaryGrid.evaluationrolloutIndex.append(node.id)
        index+=1
    ###############################################################################################################################
    if gv.UseSoftDataUpdating:
        print('Soft data will also be used for updating the simulations')
        if gv.IsSofDataAvailable:
            print('Reading soft data')
            if gv.IsSamplingSoftDataRegularGridFashion:
                softdatafile = pd.read_csv('Data/SoftSamplesRG'+gv.DataFile+str(int(gv.SoftDataPercentage*100))+'PercentUpdate'+str(gv.updateNumber)+'Section'+str(gv.sectionalZ)+'.csv')
            else:
                softdatafile = pd.read_csv('Data/SoftSamples'+gv.DataFile+str(int(gv.SoftDataPercentage*100))+'PercentUpdate'+str(gv.updateNumber)+'Section'+str(gv.sectionalZ)+'.csv')
            Xsoft = softdatafile['X'].as_matrix()[0:]
            Ysoft = softdatafile['Y'].as_matrix()[0:]
            Zsoft = softdatafile['Z'].as_matrix()[0:]
            Valsoft = softdatafile['Val'].as_matrix()[0:]
            Zsoftfilter = softdatafile['Z'].as_matrix()[0:]
            XYZsoft = softdatafile[['X','Y','Z']].as_matrix()[0:]
            Accuracysoft = softdatafile['Accuracy'].as_matrix()[0:]
            if gv.working2D:
                Xsoft = Xsoft[Zsoftfilter==gv.sectionalZ]
                Ysoft = Ysoft[Zsoftfilter==gv.sectionalZ]
                Zsoft = Zsoft[Zsoftfilter==gv.sectionalZ]
                XYZsoft = XYZsoft[Zsoftfilter==gv.sectionalZ]
                Valsoft = Valsoft[Zsoftfilter==gv.sectionalZ]
                Accuracysoft = Accuracysoft[Zsoftfilter==gv.sectionalZ]
            if gv.IsSoftDataRegular:
                for i in range(len(Xsoft)):
                    indexID = DictionaryGrid.dictionary.get((Xsoft[i],Ysoft[i],Zsoft[i]),-99)
                    if indexID==-99:
                        print('Soft data is not on regular grid, please switch on the option for irregular grid')
                        exit()
                    else:
                        DictionaryGrid.softData[indexID]=Valsoft[i]
                        DictionaryGrid.softDataAccuracy[indexID]=Accuracysoft[i]
            else:
                print('Rearranging soft data to regular grid')
                for i in range(len(Xsoft)):
                    key = (Xsoft[i], Ysoft[i],Zsoft[i])
                    closesKey = DictionaryGrid.closestTree(key)
                    indexID = DictionaryGrid.dictionary.get(closesKey,-99)
                    if indexID==-99:
                        print('Soft data not inside bounds')
                        exit()
                    else:
                        DictionaryGrid.softData[indexID]=Valsoft[i]
                        DictionaryGrid.softDataAccuracy[indexID]=Accuracysoft[i]
        else:
            print('Sampling soft data')
            if gv.trainPolicy:
                if gv.IsSamplingSoftDataRegularGridFashion:
                    softDataNode=GenerateData.generateSoftDataRG(Index,GridNode,DictionaryGrid,gv.SoftDataPercentage,'SoftSamplesRG')
                else:
                    softDataNode=GenerateData.generateSoftData(Index,GridNode,DictionaryGrid,gv.SoftDataPercentage,'SoftSamples')
            else:
                if gv.IsSamplingSoftDataRegularGridFashion:
                    GenerateData.generateSoftDataRG(Index,GridNode,DictionaryGrid,gv.SoftDataPercentage,'SoftSamplesRG')
                else:
                    GenerateData.generateSoftData(Index,GridNode,DictionaryGrid,gv.SoftDataPercentage,'SoftSamples')
    ###############################################################################################################################
    SimulationNode = [[Grid.block() for i in range(Xgrid.size)] for i in range(gv.numSimTrain)]
    if gv.IsSimulationsAvailable:
        print('Reading the simulation files from folder Update', gv.updateNumber)
        for simnumber in range(gv.numSimTrain):
            simulationfile = pd.read_csv('Output/Update'+str(gv.updateNumber)+'/Section'+str(gv.sectionalZ)+'/Sim'+str(simnumber)+'.csv')
            XSim = simulationfile['X'].as_matrix()[0:]
            YSim = simulationfile['Y'].as_matrix()[0:]
            ZSim = simulationfile['Z'].as_matrix()[0:]
            ValSim = simulationfile['Val'].as_matrix()[0:]
            XYZSim = simulationfile[['X', 'Y', 'Z']].as_matrix()[0:]
            ZfilterSim = simulationfile['Z'].as_matrix()[0:]
            if gv.working2D:
                XSim = XSim[ZfilterSim==gv.sectionalZ]
                YSim = YSim[ZfilterSim==gv.sectionalZ]
                ZSim = ZSim[ZfilterSim==gv.sectionalZ]
                XYZSim = XYZSim[ZfilterSim==gv.sectionalZ]
                ValSim = ValSim[ZfilterSim==gv.sectionalZ]
            index = 0
            for node in SimulationNode[simnumber]:
                node.x = XSim[index]
                node.y = YSim[index]
                node.z = ZSim[index]
                node.simVal=ValSim[index]
                node.id=index
                index+=1
                indexID = DictionaryGrid.dictionary.get((node.x,node.y,node.z),-99)
                if indexID==-99:
                    print('Error with Simulations')
                    exit()
                else:
                    DictionaryGrid.simulationData[simnumber][indexID]=node.simVal
                    DictionaryGrid.trueImageData[indexID]=GridNode[indexID].exhausitiveImage
    ###############################################################################################################################
    if gv.UseSensorDataUpdating:
        print('Sensor data will also be used for udpating the simualations')
        if gv.IsSensorDataAvailable:
            print('Reading sensor data')
            SensorArray = []
            SensorZeroLocation = []
            sensordatafile = pd.read_csv('Data/SamplesSensorAll'+gv.DataFile+str(int(gv.SensorDataPercentage*100))+'PercentUpdate'+str(gv.updateNumber)+'Section'+str(gv.sectionalZ)+'.csv')
            Xsensor = sensordatafile['X'].as_matrix()[0:]
            Ysensor = sensordatafile['Y'].as_matrix()[0:]
            Zsensor = sensordatafile['Z'].as_matrix()[0:]
            XYZsensor = sensordatafile[['X','Y','Z']].as_matrix()[0:]
            Zsensorfilter = sensordatafile['Z'].as_matrix()[0:]
            Valsensor = sensordatafile['Val'].as_matrix()[0:]
            if gv.working2D:
                Xsensor = Xsensor[Zsensorfilter==gv.sectionalZ]
                Ysensor = Ysensor[Zsensorfilter==gv.sectionalZ]
                Zsensor = Zsensor[Zsensorfilter==gv.sectionalZ]
                XYZsensor = XYZsensor[Zsensorfilter==gv.sectionalZ]
                Valsensor = Valsensor[Zsensorfilter==gv.sectionalZ]
            for xyz in range(len(XYZsensor)):
                indexID = DictionaryGrid.dictionary.get(tuple(XYZsensor[xyz]),-99)
                if indexID==-99:
                    print('Error in all sensor data')
                    exit()
                else:
                    DictionaryGrid.sensorAllData[indexID]=Valsensor[xyz]
                    DictionaryGrid.DitionarySensorLocation[indexID]=-99
            AllLocationSensorData = [[] for i in range(gv.numSensors)]
            for sensorIndex in range(gv.numSensors):
                sensorIndividualFile = pd.read_csv('Data/SamplesSensor'+str(sensorIndex+1)+gv.DataFile+str(int(gv.SensorDataPercentage*100))+'PercentUpdate'+str(gv.updateNumber)+'Section'+str(gv.sectionalZ)+'.csv')
                XsensorIndividual = sensorIndividualFile['X'].as_matrix()[0:]
                YsensorIndividual = sensorIndividualFile['Y'].as_matrix()[0:]
                ZsensorIndividual = sensorIndividualFile['Z'].as_matrix()[0:]
                XYZsensorIndividual = sensorIndividualFile[['X','Y','Z']].as_matrix()[0:]
                ZsensorfilterIndividual = sensorIndividualFile['Z'].as_matrix()[0:]
                ValsensorIndividual = sensorIndividualFile['Val'].as_matrix()[0:]
                AccuracysensorIndividual = sensorIndividualFile['Accuracy'].as_matrix()[0:]
                if gv.working2D:
                    XsensorIndividual =XsensorIndividual[ZsensorfilterIndividual==gv.sectionalZ]
                    YsensorIndividual =YsensorIndividual[ZsensorfilterIndividual==gv.sectionalZ]
                    ZsensorIndividual =ZsensorIndividual[ZsensorfilterIndividual==gv.sectionalZ]
                    XYZsensorIndividual =XYZsensorIndividual[ZsensorfilterIndividual==gv.sectionalZ]
                    ValsensorIndividual =ValsensorIndividual[ZsensorfilterIndividual==gv.sectionalZ]
                    AccuracysensorIndividual = AccuracysensorIndividual[ZsensorfilterIndividual==gv.sectionalZ]
                SensorNode = Grid.GridDictionaryUpdating(XYZsensorIndividual,ValsensorIndividual,name='SensorData'+str(sensorIndex+1))
                for xyz in range(len(XYZsensorIndividual)):
                    SensorNode.dictionary[tuple(XYZsensorIndividual[xyz])]=ValsensorIndividual[xyz]
                    indexID = DictionaryGrid.dictionary.get(tuple(XYZsensorIndividual[xyz]),-999)               
                    if indexID==-999:
                        print('Error with Sensor data', sensorIndex)
                        exit()
                    else:
                        DictionaryGrid.sensorIndividualData[sensorIndex][indexID]=ValsensorIndividual[xyz]
                        DictionaryGrid.sensorIndividualDataAccuracy[sensorIndex][indexID]=AccuracysensorIndividual[xyz]
                        DictionaryGrid.sensorAllDataAccuracy[indexID]=AccuracysensorIndividual[xyz]
                        AllLocationSensorData[sensorIndex].append(indexID)
                        if sensorIndex==0:
                            DictionaryGrid.locationSensorData[sensorIndex][indexID]=indexID
                            SensorZeroLocation.append(indexID)
                        else:
                            DictionaryGrid.locationSensorData[sensorIndex][SensorZeroLocation[xyz]]=indexID
                SensorArray.append(SensorNode)
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

        else:
            print('Generating sensor data')
            if gv.trainPolicy:
                GenerateData.generateSensorData(Index,GridNode,DictionaryGrid,gv.SensorDataPercentage,'Samples')
            else:
                GenerateData.generateSensorData(Index,GridNode,DictionaryGrid,gv.SensorDataPercentage,'Samples')
    ###############################################################################################################################
    if gv.IsSamplingHardDataRegularGridFashion:
         harddatafile = pd.read_csv('Data/HardSamplesRG'+gv.DataFile+str(int(gv.HardDataPercentage*100))+'Percent'+'Section'+str(gv.sectionalZ)+'.csv')
    else:
        harddatafile = pd.read_csv('Data/HardSamples'+gv.DataFile+str(int(gv.HardDataPercentage*100))+'Percent'+'Section'+str(gv.sectionalZ)+'.csv')
    Xhard = harddatafile['X'].as_matrix()[0:]
    Yhard = harddatafile['Y'].as_matrix()[0:]
    Zhard = harddatafile['Z'].as_matrix()[0:]
    Valhard = harddatafile['Val'].as_matrix()[0:]
    XYZhard = harddatafile[['X','Y','Z']].as_matrix()[0:]
    Zhardfilter = harddatafile['Z'].as_matrix()[0:]
    AccuracyHard = harddatafile['Accuracy'].as_matrix()[0:]
    if gv.working2D:
        Xhard = Xhard[Zhardfilter==gv.sectionalZ]
        Yhard = Yhard[Zhardfilter==gv.sectionalZ]
        Zhard = Zhard[Zhardfilter==gv.sectionalZ]
        Valhard = Valhard[Zhardfilter==gv.sectionalZ]
        XYZhard = XYZhard[Zhardfilter==gv.sectionalZ]
        AccuracyHard = AccuracyHard[Zhardfilter==gv.sectionalZ]
    if gv.IsHardDataRegular:
        for i in range(len(Xhard)):
            indexID = DictionaryGrid.dictionary.get((Xhard[i],Yhard[i],Zhard[i]),-99)
            if indexID==-99:
                print('Hard data is not regular, please switch on the option for irregular grid')
                exit()
            else:
                DictionaryGrid.hardData[indexID]=Valhard[i]
                DictionaryGrid.hardDataAccuracy[indexID]=AccuracyHard[i]
    else:
        print('Rearranging hard data to regular grid')
        for i in range(len(Xhard)):
            key = (Xhard[i],Yhard[i],Zhard[i])
            closesKey = DictionaryGrid.closestTree(key)
            indexID = DictionaryGrid.dictionary.get(closesKey,-99)
            if indexID==-99:
                print('Hard data not inside bounds')
                exit()
            else:
                DictionaryGrid.hardData[indexID]=Valhard[i]
                DictionaryGrid.hardDataAccuracy[indexID]=AccuracyHard[i]
    if gv.IsSofDataAvailable and gv.IsSensorDataAvailable:
        return DictionaryGrid,GridNode,Index, SensorIndex
    else:
        print('Soft and sensor data is now generated')
        print('Please switch on the option in the global vars that says the data is available and re run the program')
        exit()
