'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module contains the classes used in the project
'''
import globalVars as gv
import numpy as np
from scipy.spatial import cKDTree
from sortedcontainers import SortedDict
from numpy.polynomial.legendre import Legendre
import scipy.special as sp
from scipy.interpolate import interp1d
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import operator
class block:
    '''
    This will store the block coordinates and its simulated values
    '''
    def __init__(self):
        self.x=None
        self.y=None
        self.z=None
        self.TIVal = None
        self.simVal = gv.NoDataValue
        self.exhausitiveImage = None
        self.visited=None
        self.harddata = None
        self.id = None
        self.harddataValue = None
        self.softdata = None          
class GridDictionary:
    '''
    This is the main class for updating simulations which perfom all the necessary operations required for state and reward vector
    '''
    def __init__(self,data=None,vals=None,name=None):
        self.dictionary = SortedDict()
        self.Tree = cKDTree(data)
        if gv.working2D:
            self.data=np.append(data,[[-1,-1,gv.sectionalZ]],axis=0)
        else:
            self.data=np.append(data,[[-1,-1,-1]],axis=0)
        self.values = vals
        self.name = name   # HardData, SoftData, TI, Grid
        self.sensorAllData = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.sensorAllDataAccuracy = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.sensorIndividualData = np.array([np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32") for i in range(gv.numSensors)], dtype="float32")
        self.sensorIndividualDataAccuracy = np.array([np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32") for i in range(gv.numSensors)], dtype="float32")
        self.hardData = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.hardDataAccuracy = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.softData = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.softDataAccuracy = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.trainingImageData = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.trueImageData = np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32")
        self.simulationData = np.array([np.array([gv.NoDataValue]*(self.data.shape[0]), dtype="float32") for i in range(gv.numSimTrain)], dtype="float32")
        self.locationSensorData = np.array([np.array([gv.NoDataValue]*(self.data.shape[0])) for i in range(gv.numSensors)])
        self.rolloutIndex = []
        self.evaluationrolloutIndex = []
        self.DitionarySensorLocation = {}
    #####################################################################################################################
    # Returns the closes node to the current node
    def closestTree(self,key):
        index =  self.Tree.query(key)[1]
        return tuple(self.Tree.data[index])
    #####################################################################################################################
    # Returns the state of the simulation
    def getSimulationState(self,shapeArray,simnumber,name):
        simState = self.Tree.query(shapeArray,distance_upper_bound=0.01)[1]
        if name=='Initial':
            #np.var(self.simulationData[:,[self.simulationData[:,simState]!=-1]],axis=0)
            #self.computeLamdas(np.var(self.simulationData[:,simState][:,self.simulationData[:,simState].min(axis=0)>=0],axis=0))
            #self.simulationData[:,simState][:,self.simulationData[:,simState].min(axis=0)>=0]
            lamda = self.computeLamdas(np.var(self.simulationData[:,simState][:,self.simulationData[:,simState].min(axis=0)>gv.NoDataValue],axis=0))

            return self.simulationData[simnumber][simState], simState, lamda*gv.normalizationSimError, self.data[simState]
        elif name=='Updated':
            lamda = self.computeLamdas(np.var(self.simulationData[:,simState],axis=0))
            return self.simulationUpdating[simnumber][simState], simState, lamda*gv.normalizationSimError, self.data[simState]
        else:
            lamda = self.computeLamdas(np.var(self.simulationData[:,simState],axis=0))
            return self.simulationTempData[simnumber][simState], simState, lamda*gv.normalizationSimError, self.data[simState]
    #####################################################################################################################
    # Returns the state of the sensor prediction and model predictions
    def getSensorStateNewDictionary(self,State,simnumber,name):    
        Sensors = []   
        Zero = State
        SensorLocation = []
        SensorLocation.append(Zero)
        otherlocs = self.DitionarySensorLocation.get(State,-999)
        if otherlocs==-999:
            #print('Error with sensor location dictionary')
            SensorLocation.append(self.data.shape[0]-1)
        else:
            if otherlocs==-99:
                SensorLocation.append(self.data.shape[0]-1)
            else:
                SensorLocation.extend(otherlocs)
        Sensors = np.array(SensorLocation)    
        if name!='RollOut':
            sensorPrediction = self.generateSensorPredictionNewDictionary(Sensors)
            sensorError = self.generateSensorErrorNewDictionary(Sensors)
            modelvals = self.generateModelPredictionNewDictionary(Sensors,simnumber,name)
            if sensorError<=0:
                return sensorPrediction,modelvals, gv.NoDataValue, Sensors
            else:   
                return sensorPrediction,modelvals, sensorError, Sensors
        else:
            sensorPrediction = self.generateSensorPredictionNewDictionary(Sensors)
            modelvals = self.generateModelPredictionNewDictionary(Sensors,simnumber,name)
            sensorError = self.generateSensorErrorNewDictionary(Sensors)
            if sensorError<=0:
                return sensorPrediction,modelvals, gv.NoDataValue
            else:
                return sensorPrediction,modelvals, sensorError
    #####################################################################################################################
    # Returns the sensor based predictions
    def generateSensorPredictionNewDictionary(self,SensorArray):
        Vals = []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for i in SensorArray:
                Vals.append(self.sensorAllData[i])
            predictionSensor = np.average(Vals,axis=0)
            if predictionSensor<=0:
                predictionSensor=gv.NoDataValue
            return predictionSensor
    #####################################################################################################################
    # Returns the Error of the sensor
    def generateSensorErrorNewDictionary(self,SensorArray):
        Vals = []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for i in SensorArray:
                Vals.append(self.sensorAllDataAccuracy[i])
            predictionSensor = np.average(Vals,axis=0)
            if predictionSensor<=0:
                return gv.NoDataValue
            else:
                return 1-predictionSensor
    #####################################################################################################################
    # Returns the Error of the sensor
    def generateSensorError(self,SensorArray):
        Vals = []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for senseIndex in range(gv.numSensors):
                Vals.append(self.sensorIndividualDataAccuracy[senseIndex][SensorArray[senseIndex]])
            predictionSensor = np.average(Vals,axis=0)
            predictionSensor[predictionSensor<0]=gv.NoDataValue
            predictionSensor[predictionSensor!=-1] = 1-predictionSensor[predictionSensor!=-1]
            return predictionSensor
    #####################################################################################################################
    # Returns the sensor based predictions
    def generateSensorPrediction(self,SensorArray):
        Vals = []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for senseIndex in range(gv.numSensors):
                Vals.append(self.sensorIndividualData[senseIndex][SensorArray[senseIndex]])
            predictionSensor = np.average(Vals,axis=0)
            predictionSensor[predictionSensor<0]=gv.NoDataValue
            return predictionSensor
    #####################################################################################################################
    # Returns the model based predictions
    # Returns all the sensor predictin in the CNN patch
    def generateModelPrediction(self, SensorArray,simnumber,name):
        Vals= []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for senseIndex in range(gv.numSensors):
                if name=='Initial':
                    Vals.append(self.simulationData[simnumber][SensorArray[senseIndex]])
                elif name=='Updated':
                    Vals.append(self.simulationUpdating[simnumber][SensorArray[senseIndex]])
                else:
                    Vals.append(self.simulationTempData[simnumber][SensorArray[senseIndex]])
            modelPred = np.average(Vals,axis=0)
            modelPred[modelPred<0]=gv.NoDataValue
            return modelPred
    #####################################################################################################################
    # Returns the error os the model
    def generateModelPredicitonError(self,SensorArray, name):
        Vals = []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for senseIndex in range(gv.numSensors):
                if name=='Initial':
                    Vals.append(np.var(self.simulationData,axis=0)[SensorArray[senseIndex]])
                else:
                    Vals.append(np.var(self.simulationData,axis=0)[SensorArray[senseIndex]])
            predictionModel = np.average(Vals,axis=0)
            predictionModel[predictionModel<0]=gv.NoDataValue
            return predictionModel
    #####################################################################################################################
    # Returns the error os the model
    def generateModelPredicitonErrorNew(self,SensorArray, name):
        Vals = []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for senseIndex in range(gv.numSensors):
                if name=='Initial':
                    Vals.append(np.var(self.simulationData,axis=0)[SensorArray[senseIndex]])
                else:
                    Vals.append(np.var(self.simulationData,axis=0)[SensorArray[senseIndex]])
            
            predictionModel = np.average(Vals,axis=0)
            if predictionModel<=0:
                return gv.NoDataValue
            else:
                return predictionModel
    #####################################################################################################################
    # Returns the model based predictions
    # Returns only the prediction of current node in consideration
    def generateModelPredictionNewDictionary(self, SensorArray,simnumber,name):
        Vals= []
        if SensorArray.size==0:
            return gv.NoDataValue
        else:
            for i in SensorArray:
                if name=='Initial':
                    Vals.append(self.simulationData[simnumber][i])
                elif name=='Updated':
                    Vals.append(self.simulationUpdating[simnumber][i])
                else:
                    Vals.append(self.simulationTempData[simnumber][i])
            return Vals
    #####################################################################################################################
    # Returns the state of the simulation data
    def reset(self):
        self.simulationUpdating = copy(self.simulationData)
    #####################################################################################################################
    # Returns the state of the soft data
    def resetTemp(self):
        self.simulationTempData = copy(self.simulationUpdating)
    #####################################################################################################################
    # Returns the state of the soft data
    def getSoftDataState(self,State):
        lamda = self.computeLamdas(1-self.softDataAccuracy[State][self.softDataAccuracy[State]!=gv.NoDataValue])
        return self.softData[State], lamda
    #####################################################################################################################
    # Returns the state of the simulation data state
    def getSimulationDataState(self,State,simnumber,name):        
        if name=='Initial':  
            lamda = self.computeLamdas(np.var(self.simulationData[:,State],axis=0))        
            return self.simulationData[simnumber][State], lamda*100
        elif name=='Updated':
            lamda = self.computeLamdas(np.var(self.simulationData[:,State],axis=0))
            return self.simulationUpdating[simnumber][State],lamda*100
        else:
            lamda = self.computeLamdas(np.var(self.simulationData[:,State],axis=0))
            return self.simulationTempData[simnumber][State], lamda*100
    #####################################################################################################################
    # Compute the lamda factors
    def computeLamdas(self,vals):
        if vals.size==0:
            return gv.NoDataValue
        else:
            #if vals[vals>=0].size==0:
                #print('Empty')
            length = np.mean(vals[vals>=0])
            if length==0:
                length=1
            return length
    #####################################################################################################################
    # Compute the data event with cords and values
    def cordandValNearSamplesRadius(self,key, simnumber,name, name2):
        if name=='Simulation' or name=='TI':
            index =  self.Tree.query(key,k=gv.maxSamplesUpdatingDataEvent+1,distance_upper_bound=gv.searchRadiusUpdatingSim)[1]
            index = index[index!=len(self.Tree.data)]
        elif name=='SoftSamples':
            index =  self.Tree.query(key,k=gv.maxSamplesInSoftData,distance_upper_bound=gv.searchRadiusUpdatingSoftSamples)[1]
            index = index[index!=len(self.Tree.data)]
            index = index[self.softData[index]>=0]
        elif name=='HardSamples':
            index =  self.Tree.query(key,k=gv.maxSamplesInHardData,distance_upper_bound=gv.searchRadiusUpdatingHardSamples)[1]
            index = index[index!=len(self.Tree.data)]
            index = index[self.hardData[index]>=0]
        XYZ = self.Tree.data[index[0]]
        if np.absolute(np.subtract(XYZ,key)).sum()==0:
            if name == 'SoftSamples':
                lookingIndex = index[1:gv.maxSamplesUpdatingDataEventSoft+1]
            else:
                lookingIndex = index[1:gv.maxSamplesUpdatingDataEvent+1]
        else:
            if name == 'SoftSamples':
                lookingIndex = index[:gv.maxSamplesUpdatingDataEventSoft]
            else:
                lookingIndex = index[:gv.maxSamplesUpdatingDataEvent]
        Cord = self.Tree.data[lookingIndex]
        valuesSoft = self.softData[lookingIndex]
        if name2=='Initial':
            valuesSim = self.simulationData[simnumber][lookingIndex]
        elif name2=='Updated':
            valuesSim = self.simulationUpdating[simnumber][lookingIndex]
        else:
            valuesSim = self.simulationTempData[simnumber][lookingIndex]
        return Cord, valuesSim, valuesSoft, lookingIndex
    #####################################################################################################################
    # Compute the template configuration of the data event
    # Fix here so that it also returns the max data event template size
    def templateconfig(self,key,Cord):
        Template = np.subtract(Cord,key)
        patchSize = ((gv.numBlocksX**2+gv.numBlocksY**2+gv.numBlockZ**2)**0.5)
        Template2 = (Template+patchSize)/(patchSize*2)
        ToleranceSoft= (np.sqrt(np.sum(np.square(Template), axis=1)))*gv.toleranceLagUpdatingSoft
        Tolerancehard = (np.sqrt(np.sum(np.square(Template), axis=1)))*gv.toleranceLagUpdatingHard
        return Template,Template2, ToleranceSoft, Tolerancehard
    #####################################################################################################################
    # Calcualtes the pdf with high order moments with the replicates See Ling et al. 2019
    def computepdf(self,replicates,dataevent,order):
        centerNode = replicates[-1,:]
        otherVals = replicates[:-1,:]
        Xw = []
        XProd = np.ones(otherVals.shape[1])
        X = np.zeros(otherVals.shape)
        for ord in range(order+1):
            Xw.append((ord+0.5)*sp.legendre(ord)(centerNode))
            X += (ord+0.5)*(sp.legendre(ord)(otherVals))*(sp.legendre(ord)(dataevent)).reshape(-1,1)
        XProd = np.prod(X,axis=0)
        MultipliedX = Xw*XProd
        coefficients = np.sum(MultipliedX,axis=1)
        coefficients = coefficients/(2*coefficients[0])
        if gv.debug:
            self.plotpolynomialSeries(coefficients, 'InitialCPDF')
        return coefficients
    #####################################################################################################################
    # Calculates the cdf coefficients See Ling et al. 2019
    def computecdf(self,coefficients,order):
        dw = np.zeros(order+2)
        for ord in range(1,order+1):
            dw[ord+1]+=(coefficients[ord]/(2*ord+1))
            dw[ord-1]-=(coefficients[ord]/(2*ord+1))
        dw[0] = dw[0]+0.5
        dw[1] = dw[1]+0.5
        if gv.debug:
            self.plotpolynomialSeries(dw,'InitialCCDF')
        return dw
    #####################################################################################################################
    # Find the similarity between the replicates and the data event
    def similarityreplicate(self,replicates,dataevent):
        if gv.compareCenteralsoUpdatingReplicates:
            centerNodeDataEvent = np.average(dataevent)
            NewDataevent = np.append(dataevent,centerNodeDataEvent)
            VarDataEvent = np.var(NewDataevent)
            SubtractedReplicate = replicates-np.reshape(NewDataevent,(-1,1))
            SubtractedReplicate = np.sum(SubtractedReplicate**2,axis=0)/len(SubtractedReplicate[:,])
            Distance = SubtractedReplicate-VarDataEvent*gv.varianceTolerance
        else:
            ReplicatewithourCenter = replicates[:-1,:]
            VarDataEvent = np.var(dataevent)
            SubtractedReplicate = ReplicatewithourCenter-np.reshape(dataevent,(-1,1))
            SubtractedReplicate = np.sum(SubtractedReplicate**2,axis=0)/len(SubtractedReplicate[:,])
            Distance = SubtractedReplicate-VarDataEvent*gv.varianceTolerance
        return replicates[:,Distance<=0]      
    #####################################################################################################################
    # Returns the replicates of the data event within the patch   
    def findReplicates(self,IndexPatch,CordPatch, tolerance, template, dataevent, name,simnumber,name2):
        KeysLooking = []
        Indexes = []
        if name=='HardSamples':
            allIndexes = IndexPatch[self.hardData[IndexPatch]>=0]
            allkeys = CordPatch[self.hardData[IndexPatch]>=0]
        elif name=='SoftSamples':
            allIndexes = IndexPatch[self.softData[IndexPatch]>=0]
            allkeys = CordPatch[self.softData[IndexPatch]>=0]
        elif name=='Simulation':
            allIndexes = IndexPatch[self.simulationData[simnumber][IndexPatch]>=0]
            allkeys = CordPatch[self.simulationData[simnumber][IndexPatch]>=0]
        elif name=='TI':
            allIndexes = IndexPatch[self.trainingImageData[IndexPatch]>=0]
            allkeys = CordPatch[self.trainingImageData[IndexPatch]>=0]
        for i in range(len(template)):
            if name=='HardSamples':
                KeysLooking.append(np.array(allkeys)+template[i])
                Indexes.append(self.Tree.query(KeysLooking[i], k=1, distance_upper_bound=tolerance[i])[1])
            elif name=='SoftSamples':
                KeysLooking.append(np.array(allkeys)+template[i])
                Indexes.append(self.Tree.query(KeysLooking[i], k=1, distance_upper_bound=tolerance[i])[1])
            else:
                KeysLooking.append(np.array(allkeys)+template[i])
        if name=='Simulation':
            Indexes = self.Tree.query(KeysLooking,k=1,distance_upper_bound=0.1)[1]
        minimums = np.min(CordPatch,axis=0)
        maximums = np.max(CordPatch,axis=0)
        indicatorInside = np.ndarray.all(np.ndarray.all(np.logical_and(KeysLooking<=maximums, KeysLooking>=minimums),axis=2),axis=0)
        Indexes = np.append(Indexes,[allIndexes],axis=0)
        insideReplicates = np.array(Indexes)[:,indicatorInside]
        if name=='HardSamples':
            return self.hardData[insideReplicates][:,np.ndarray.all(self.hardData[insideReplicates]!=gv.NoDataValue,axis=0)]
        elif name=='SoftSamples':
            return self.softData[insideReplicates][:,np.ndarray.all(self.softData[insideReplicates]!=gv.NoDataValue,axis=0)]
        elif name=='Simulation':
            if name2=='Initial':
                return self.simulationData[simnumber][insideReplicates][:,np.ndarray.all(self.simulationData[simnumber][insideReplicates]!=gv.NoDataValue,axis=0)]
            elif name2=='Updated':
                return self.simulationUpdating[simnumber][insideReplicates][:,np.ndarray.all(self.simulationUpdating[simnumber][insideReplicates]!=gv.NoDataValue,axis=0)]
            else:
                return self.simulationTempData[simnumber][insideReplicates][:,np.ndarray.all(self.simulationTempData[simnumber][insideReplicates]!=gv.NoDataValue,axis=0)]
        elif name=='TI':
            return self.trainingImageData[insideReplicates][:,np.ndarray.all(self.trainingImageData[insideReplicates]!=gv.NoDataValue,axis=0)]
        else:
            print('Something is weird')
    #####################################################################################################################
    # Returns the reward vector of HO stats cdf 
    def HOstatsCompute(self,IndexPatch,CordPatch, tolerance, template, dataevent, name,order,simnumber,name2,centerNodeVal):
        replicates = self.findReplicates(IndexPatch,CordPatch, tolerance, template, dataevent, name,simnumber,name2)
        if replicates.size==0:
            return np.array([gv.NoDataValue])
        else:
            if gv.similarityCheckUpdatingReplicates:
                similarReplicates = self.similarityreplicate(replicates,dataevent)
                if similarityreplicate.size==0:
                    return np.array([gv.NoDataValue])
                else:
                    pdf = self.computepdf(similarReplicates,dataevent,order)
                    cdf = self.computecdf(pdf,order)
            else:
                pdf = self.computepdf(replicates,dataevent,order)
                cdf = self.computecdf(pdf,order)
            newcoefficient = self.correctcdf(cdf,order)
            if gv.correctPDF:
                prob = self.pdfcorrect(newcoefficient,order,centerNodeVal)
                return prob
            else:
                return newcoefficient
    #####################################################################################################################
    # Corrects the negative parts of the cdf
    def correctcdf(self,cdf,order):
        poly = Legendre(coef=cdf)
        x = poly.linspace(n=100)
        newy= x[1]/x[1].max()
        changedy = np.maximum.accumulate(newy)
        newpolynomial = Legendre.fit(x[0],changedy,deg=order+1)
        if gv.debug:
            self.plotpolynomialSeries(newpolynomial.coef,'CorrectedCCDF')
        if gv.correctPDF:
            return newpolynomial
        else:
            return changedy
    #####################################################################################################################
    # Compute the looking index (CNN patch) for all blocks
    def computeNeighboursall(self,Grid,Index,cnnShape):
        print('Creating Neighours once')
        dictionary = {}
        for ind in Index:
            xyz = [Grid[ind].x,Grid[ind].y,Grid[ind].z]
            NeighLooking = cnnShape+xyz
            neighIndexes = self.Tree.query(NeighLooking,distance_upper_bound=0.01)[1]
            dictionary[ind]=neighIndexes
        self.dictionaryAllIndex = dictionary
     #####################################################################################################################
    # compute the pdf of the corrected cdf
    def pdfcorrect(self,cdfpolynomial,order,centerNodeVal):
      correctedpdf = cdfpolynomial.deriv(1)
      if gv.debug:
          self.plotpolynomialSeries(correctedpdf.coef)
      prob = correctedpdf(centerNodeVal)
      return prob
    #####################################################################################################################
    # Returns the reward vector of HO stats cdf 
    def HOstatsComputeNew(self,IndexPatch,CordPatch, tolerance, template, dataevent, name,order,simnumber,name2,centerNodeVal,centerNodeValUpdated):
        replicates = self.findReplicates(IndexPatch,CordPatch, tolerance, template, dataevent, name,simnumber,name2)
        if replicates.size==0:
            return gv.NoDataValue, gv.NoDataValue
        else:
            if gv.similarityCheckUpdatingReplicates:
                similarReplicates = self.similarityreplicate(replicates,dataevent)
                if similarReplicates.size==0:
                    return gv.NoDataValue, gv.NoDataValue
                else:
                    pdf = self.computepdf(similarReplicates,dataevent,order)
                    cdf = self.computecdf(pdf,order)
            else:
                pdf = self.computepdf(replicates,dataevent,order)
                cdf = self.computecdf(pdf,order)
            newcoefficient = self.correctcdf(cdf,order)
            if gv.correctPDF:
                prob,prob2 = self.pdfcorrectnew(newcoefficient,order,centerNodeVal,centerNodeValUpdated)
                return prob, prob2
            else:
                return newcoefficient
    #####################################################################################################################
    def plotpolynomialSeries(self,polynomial,name):
        poly = Legendre(coef=polynomial)
        x = poly.linspace(n=100)
        plt.title(name)
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.plot(x[0],x[1])
        plt.savefig('Output/PDFCDFFigs/'+name+'.png')
        plt.clf()
        plt.close()
        #plt.show()
    #####################################################################################################################
    # compute the pdf of the corrected cdf
    def pdfcorrectnew(self,cdfpolynomial,order,centerNodeVal,centerNodeUpdated):
      correctedpdf = cdfpolynomial.deriv(1)
      if gv.debug:
          self.plotpolynomialSeries(correctedpdf.coef, 'CorrectedCPDF')
      prob = correctedpdf(centerNodeVal)
      prob2 = correctedpdf(centerNodeUpdated)
      return prob, prob2
    #####################################################################################################################
    # Standardize the data
    def standardizeData(self):
        self.maximumValSim = np.max(self.simulationData)
        if gv.UseSoftDataUpdating:
            self.maximumValSoft = np.max(self.softData)
        self.maximumSensorVal = np.max(self.sensorAllData)
        self.minimumValSim = np.min(self.simulationData[self.simulationData!=gv.NoDataValue])
        if gv.UseSoftDataUpdating:
            self.minimumValSoft = np.min(self.softData[self.softData!=gv.NoDataValue])
        self.minimumSensorVal = np.min(self.sensorAllData[self.sensorAllData!=gv.NoDataValue])
        if gv.UseSoftDataUpdating:
            my_list_max = [self.maximumValSim,self.maximumValSoft,self.maximumSensorVal]
            my_list_min = [self.minimumValSim,self.minimumValSoft,self.minimumSensorVal]
        else:
            my_list_max = [self.maximumValSim,self.maximumSensorVal]
            my_list_min = [self.minimumValSim,self.minimumSensorVal]
        indexmax, valuemax = max(enumerate(my_list_max), key=operator.itemgetter(1))
        indexmin, valuemin = min(enumerate(my_list_min), key=operator.itemgetter(1))
        self.stretchValue = valuemax-valuemin
        self.mininmumScalingValue = valuemin
        self.maximumScalingValue = valuemax
        print('Scaling all the data with minimum value of ', self.mininmumScalingValue)
        print('Scaling all the data with maximum value of ', self.maximumScalingValue)
        print('Stretch length is', self.stretchValue)
        self.simulationData[:,:-1] = (self.simulationData[:,:-1]-self.mininmumScalingValue)/self.stretchValue
        if np.min(self.simulationData[self.simulationData!=gv.NoDataValue])<0:
            print('Error in scaling min in simulation data')
            exit()
        self.softData[self.softData!=gv.NoDataValue]=(self.softData[self.softData!=gv.NoDataValue]-self.mininmumScalingValue)/self.stretchValue
        if gv.UseSoftDataUpdating:
            if np.min(self.softData[self.softData!=gv.NoDataValue])<0:
                print('Errror scaling min in soft data')
                exit()
        self.trueImageData[self.trueImageData!=gv.NoDataValue]=(self.trueImageData[self.trueImageData!=gv.NoDataValue]-self.mininmumScalingValue)/self.stretchValue
        if np.min(self.trueImageData[self.trueImageData!=gv.NoDataValue])<0:
            print('Error scaling min in true image')
            exit()
        self.hardData[self.hardData!=gv.NoDataValue]=(self.hardData[self.hardData!=gv.NoDataValue]-self.mininmumScalingValue)/self.stretchValue
        if np.min(self.hardData[self.hardData!=gv.NoDataValue])<0:
            print('Error scaling min in hard data')
            exit()
        for i in range(gv.numSensors):
            self.sensorIndividualData[i][self.sensorIndividualData[i]!=gv.NoDataValue] = (self.sensorIndividualData[i][self.sensorIndividualData[i]!=gv.NoDataValue]-self.mininmumScalingValue)/self.stretchValue
            if np.min(self.sensorIndividualData[i][self.sensorIndividualData[i]!=gv.NoDataValue])<0:
                print('Error scaling min in sensor ', i, ' data')
                exit()
        self.sensorAllData[self.sensorAllData!=gv.NoDataValue]=(self.sensorAllData[self.sensorAllData!=gv.NoDataValue]-self.mininmumScalingValue)/self.stretchValue
        if np.min(self.sensorAllData[self.sensorAllData!=gv.NoDataValue])<0:
            print('Error scaling min in all sensor data')
            exit()
    #####################################################################################################################
    # Inverse transform the standardize data
    def inverseStandardizeData(self):
        self.simulationData[:,:-1] = self.simulationData[:,:-1]*self.maximumVal
        self.simulationUpdating[:,:-1] = self.simulationUpdating[:,:-1]*self.maximumVal
        self.softData[self.softData!=-1]=self.softData[self.softData!=-1]*self.maximumVal
        self.trueImageData[self.trueImageData!=-1]=self.trueImageData[self.trueImageData!=-1]*self.maximumVal
        for i in range(gv.numSensors):
            self.sensorIndividualData[i][self.sensorIndividualData[i]!=-1] = self.sensorIndividualData[i][self.sensorIndividualData[i]!=-1]*self.maximumVal
        self.sensorAllData[self.sensorAllData!=-1]=self.sensorAllData[self.sensorAllData!=-1]*self.maximumVal

