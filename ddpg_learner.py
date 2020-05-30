'''
Copyright (c) 2020, Ashish Kumar All rights reserved.
This is the main module the train the agent and then use the trained agent to update the simulated models 
'''
from Models import Actor, Critic
import numpy as np
import log_File
import time
import globalVars as gv
import os
from Noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from Memory import Memory
from DDPG import DDPG
import ResultPlotting
from copy import deepcopy,copy
from ReadData import inputFiles
import Utilities
import tensorflow as tf
from progressbar import *
import ResultPlotting
import time
import Validation
import matplotlib.pyplot as plt
widgets = ['Completed: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
           ' ', ETA()] #see docs for other options
# This funciton allows for leanring the agent for updaitng resource models with incoming data
def learn(nbEpochs = gv.TrainingRandomPaths,centerMoment=False,
         rewardScale = 1.0,noiseType = 'ou_0.1', normalizeObservation=False, criticL2reg = 1e-3,
         actorLR = 1e-4, criticLR = 1e-3, gamma = 0.99, clipAction = None,tau=0.001, nbTrainSteps = 200,
         batchSize = gv.batchSize, paramNoiseAdaptInterval = 199, actionRange = (0,gv.ActionMaxRange), observationRange = (-1,1),
         actionShape=(1,), cnnShapeObservation =(((gv.numBlocksX*2+1)*(gv.numBlocksY*2+1)),),sess=None,clip_norm =1, saveIter = 10000,explorationNoise = 0.05, explorationMean=0,trainingiter =5, DecayStart = 0.5, Rate = 0.98):
    
    if gv.UseSoftDataUpdating:
        additionalObservationShape = (gv.numSensors-1+2+1+1+1*gv.maxSamplesUpdatingDataEvent+1*gv.maxSamplesUpdatingDataEventSoft+3*gv.maxSamplesUpdatingDataEvent+3*gv.maxSamplesUpdatingDataEventSoft,) 
        # Here 1 sensor + 1 sensor predict + 3 errors + num of samples dataevent in sim + num of samples data event soft+ DE config sim + and soft 3D
    else:
        additionalObservationShape = (gv.numSensors-1+2+1+1*gv.maxSamplesUpdatingDataEvent+3*gv.maxSamplesUpdatingDataEvent,) 
        # Here 1 sensor+ 1 sensor predict+ 2 error + num of samples in dataevent in sim +DE config 3D
    actionNoise = None
    paramNoise = None
    if noiseType is not None:
        for currentNoiseType in noiseType.split(','):
            if currentNoiseType=='none':
                pass
            elif 'adaptive-param' in currentNoiseType:
                _, stddev= currentNoiseType.split('_')
                paramNoise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in currentNoiseType:
                _, stddev = currentNoiseType.split('_')
                actionNoise = NormalActionNoise(mu=np.zeros(actionShape[0]), sigma=float(stddev)*np.ones(actionShape[0]))
            elif 'ou' in currentNoiseType:
                _, stddev = currentNoiseType.split('_')
                actionNoise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actionShape[0]), sigma=float(stddev)*np.ones(actionShape[0]))
            else:
                raise RuntimeError('Unknown noise type')
    
    print('scaling actions with min {} and max {} before executing'.format(actionRange[0],actionRange[1]))
    print('scaling observations with min {} and max {}'.format(observationRange[0], observationRange[1]))
    memory = Memory(limit=int(1e6), action_shape=actionShape, observation_shape=cnnShapeObservation,additionalObservation_shape=additionalObservationShape)
    if gv.UseSoftDataUpdating:
        name = 'CNN2Input'
    else:
        name = 'CNN1Input'
    critic = Critic(network=name)
    actor = Actor(network=name)
    agent = DDPG(actor,critic,memory,cnnShapeObservation,additionalObservationShape,actionShape,
                 paramNoise,actionNoise,gamma=gamma,tau=tau,observation_range=observationRange, additional_observation_range=observationRange,
                 action_range=actionRange,critic_l2_reg=criticL2reg,actor_lr=actorLR, critic_lr=criticLR)
    print('Using agent with the following configuration:')
    print(str(agent.__dict__.items()))
    agent.initialize(sess)
    saver = tf.train.Saver()
    agent.initilizeSaver(saver)
    if os.path.isfile('./Output/TrainedModel/'+noiseType+'/'+agent.actor.networkName+'/MyModel.meta'):
        agent.load(saver,noiseType)
    sess.graph.finalize()
    Neigh = Utilities.cnnshape()
    DictionaryGrid,GridNode,Index, SensorIndex = inputFiles()
    DictionaryGrid.standardizeData()
    ResultPlotting.outputhardData(DictionaryGrid)
    if gv.UseSoftDataUpdating:
        ResultPlotting.outputsoftData(DictionaryGrid)
    ResultPlotting.outputVisualizationSoft(DictionaryGrid)
    ResultPlotting.outputVisualizationHard(DictionaryGrid)
    ResultPlotting.outputVisualizationTrueImage(DictionaryGrid)
    DictionaryGrid.computeNeighboursall(GridNode,Index,Neigh)
    locationToPlot = ResultPlotting.outputSensorPredictionGraph(DictionaryGrid) 
    globalExpCounter = 0
    pbar = ProgressBar(widgets=widgets, maxval=nbEpochs*gv.numSimTrain*len(Index))
    pbar.start()
    agent.reset()
    for simnumber in range(gv.numSimTrain):
        print('Working on Sim {}'.format(simnumber))
        ResultPlotting.outputSensorVsModelPredictionGraphIndividualSimInitial(DictionaryGrid,simnumber,locationToPlot)
        ResultPlotting.outputVisualizationSim(DictionaryGrid,simnumber)
        epochNumber = 0
        aL = []
        cL = []           
        counter=0
        for i in range(nbEpochs):
            np.random.shuffle(Index)                                        
            DictionaryGrid.reset()
            noDataCounter = 0
            for ind in Index:
                NeighLooking = DictionaryGrid.dictionaryAllIndex.get(ind,-99)
                if NeighLooking.shape[0]!=Neigh.shape[0]:
                    print('Error1')
                    exit()
                else:
                    ZeroS_xyz = DictionaryGrid.data[ind]
                    NeighLooking_OS =  Neigh+ZeroS_xyz
                    ZeroS_sim,ZeroS_state, ZeroS_lamda_sim,ZeroS_cordpatch = DictionaryGrid.getSimulationState(NeighLooking_OS,simnumber,'Initial')
                    ZeroS_cord, ZeroS_valSim, ZeroS_valSoft, ZeroS_lookingIndex = DictionaryGrid.cordandValNearSamplesRadius(ZeroS_xyz,simnumber,'Simulation','Updated')
                    if gv.UseSoftDataUpdating:
                        Softbased_cord, Softbased_simvals, Softbased_softvals, Softbased_lookingIndex = DictionaryGrid.cordandValNearSamplesRadius(ZeroS_xyz,simnumber,'SoftSamples','Updated')
                        Softbased_template, Softbased_templatenorm, Softbased_tolsoft, Softbased_tolhard = DictionaryGrid.templateconfig(ZeroS_xyz,Softbased_cord)
                        soft_data, soft_lamda = DictionaryGrid.getSoftDataState(ZeroS_state)
                    ZeroS_template,ZeroS_templateNorm, ZeroS_tolsoft, ZeroS_tolHard = DictionaryGrid.templateconfig(ZeroS_xyz,ZeroS_cord)

                    ZeroS_sensor_pred,ZeroS_model_pred, ZeroS_lamda_sense, ZeroS_sensors_array = DictionaryGrid.getSensorStateNewDictionary(ind,simnumber,'Updated')
                    if gv.UseSoftDataUpdating:
                        ZeroS_additional = np.concatenate(([ZeroS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim, [soft_lamda],Softbased_softvals,np.ravel(ZeroS_templateNorm),np.ravel(Softbased_templatenorm)))
                    else:
                        ZeroS_additional = np.concatenate(([ZeroS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim,np.ravel(ZeroS_templateNorm)))                                                                                       
                    if globalExpCounter%gv.DecayIter==0 and globalExpCounter!=0:
                        DecayStart=DecayStart*Rate
                    if np.random.random()<DecayStart:
                        # This is done to ensure fast convergence. Even without this the framework converges just takes longer
                       action = np.array([DictionaryGrid.simulationData[simnumber][ind]+np.random.normal(explorationMean,explorationNoise)])
                       action=np.clip(action, actionRange[0], actionRange[1])
                    else:
                        if gv.UseSoftDataUpdating:
                            action,q=agent.step((ZeroS_sim)/gv.valuenormalization,(soft_data)/gv.valuenormalization,(ZeroS_additional)/gv.valuenormalization,compute_Q=False)
                        else:
                            action,q=agent.step((ZeroS_sim)/gv.valuenormalization,(ZeroS_additional)/gv.valuenormalization,compute_Q=False)
                    prob_original_simulation, prob_updated_simulation = DictionaryGrid.HOstatsComputeNew(ZeroS_state,ZeroS_cordpatch,None,ZeroS_template,ZeroS_valSim,'Simulation',gv.LegendrePolynomialOrderUpdating,simnumber,'Initial',DictionaryGrid.simulationData[simnumber][ind], action[0])
                    if gv.UseSoftDataUpdating:
                        prob_original_soft, prob_updated_soft = DictionaryGrid.HOstatsComputeNew(ZeroS_state,ZeroS_cordpatch,Softbased_tolsoft,Softbased_template,Softbased_softvals,'SoftSamples',gv.LegendrePolynomialOrderUpdating,simnumber,'Initial',DictionaryGrid.simulationData[simnumber][ind], action[0])
                    DictionaryGrid.simulationData[simnumber][ind]=action
                    NS_sim = DictionaryGrid.simulationData[simnumber][ZeroS_state]
                    DictionaryGrid.simulationData[simnumber][ind]=DictionaryGrid.simulationUpdating[simnumber][ind]
                    DictionaryGrid.simulationUpdating[simnumber][ind]=action[0]
                    NS_model_pred = DictionaryGrid.generateModelPredictionNewDictionary(ZeroS_sensors_array,simnumber,'Updated')
                    if gv.UseSoftDataUpdating:
                        if ZeroS_lamda_sense<0 and soft_lamda<0:
                            noDataCounter+=1
                        NS_additional = np.concatenate(([NS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim, [soft_lamda], Softbased_softvals, np.ravel(ZeroS_templateNorm),np.ravel(Softbased_templatenorm)))
                        reward=Utilities.computeRewardHOStatsNew1(prob_original_simulation, prob_updated_simulation,ZeroS_sensor_pred,NS_model_pred,ZeroS_model_pred,ZeroS_lamda_sim,ZeroS_lamda_sense, prob_original_soft,prob_updated_soft, soft_lamda,action[0],DictionaryGrid.simulationData[simnumber][ind])
                    else:
                        if ZeroS_lamda_sense<0:
                            noDataCounter+=1
                        NS_additional = np.concatenate(([NS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim,np.ravel(ZeroS_templateNorm)))  
                        reward=Utilities.computeRewardHOStatsNew1(prob_original_simulation, prob_updated_simulation,ZeroS_sensor_pred,NS_model_pred,ZeroS_model_pred,ZeroS_lamda_sim,ZeroS_lamda_sense)
                    reward = reward/gv.rewardNormalization
                    epochNumber+=1
                    globalExpCounter+=1

                    if gv.UseSoftDataUpdating:
                        agent.storeTransition(np.array([ZeroS_sim])/gv.valuenormalization,np.array(action),np.array([soft_data])/gv.valuenormalization,np.array([reward]),np.array([ZeroS_additional])/gv.valuenormalization,np.array([NS_sim])/gv.valuenormalization,np.array([NS_additional])/gv.valuenormalization)
                    else:
                        agent.storeTransition(np.array([ZeroS_sim])/gv.valuenormalization,np.array(action),np.array([reward]),np.array([ZeroS_additional])/gv.valuenormalization,np.array([NS_sim])/gv.valuenormalization,np.array([NS_additional])/gv.valuenormalization)
            ResultPlotting.outputVisualizationSimUpdate(DictionaryGrid,int(counter),simnumber)
            ResultPlotting.outputSensorVsModelPredictionGraphIndividualSimUpdatedTrain(DictionaryGrid,simnumber,int(counter),locationToPlot)
            print(noDataCounter)
            counter+=1
            if counter%(int(trainingiter))==0 and counter!=0:
                for t_train in range(nbTrainSteps):                                
                    c1,a1 =agent.train()                            
                    aL.append(a1)
                    cL.append(c1)
                    if t_train%paramNoiseAdaptInterval==0:
                        distance=agent.adaptParamNoise()
                agent.updateTargetNet()
                agent.save(noiseType)                        
                DictionaryGrid.reset()
                startime = time.time()
                for ind in Index:
                    NeighLooking = DictionaryGrid.dictionaryAllIndex.get(ind,-99)
                    if NeighLooking.shape[0]!=Neigh.shape[0]:
                        print('Error1')
                        exit()
                    else:
                        ZeroS_xyz = DictionaryGrid.data[ind]
                        NeighLooking_OS =  Neigh+ZeroS_xyz
                        ZeroS_sim,ZeroS_state, ZeroS_lamda_sim,ZeroS_cordpatch = DictionaryGrid.getSimulationState(NeighLooking_OS,simnumber,'Initial')
                        ZeroS_cord, ZeroS_valSim, ZeroS_valSoft, ZeroS_lookingIndex = DictionaryGrid.cordandValNearSamplesRadius(ZeroS_xyz,simnumber,'Simulation','Updated')
                        if gv.UseSoftDataUpdating:
                            Softbased_cord, Softbased_simvals, Softbased_softvals, Softbased_lookingIndex = DictionaryGrid.cordandValNearSamplesRadius(ZeroS_xyz,simnumber,'SoftSamples','Updated')
                            Softbased_template, Softbased_templatenorm, Softbased_tolsoft, Softbased_tolhard = DictionaryGrid.templateconfig(ZeroS_xyz,Softbased_cord)
                            soft_data, soft_lamda = DictionaryGrid.getSoftDataState(ZeroS_state)
                        ZeroS_template,ZeroS_templateNorm, ZeroS_tolsoft, ZeroS_tolHard = DictionaryGrid.templateconfig(ZeroS_xyz,ZeroS_cord)

                        ZeroS_sensor_pred,ZeroS_model_pred, ZeroS_lamda_sense, ZeroS_sensors_array = DictionaryGrid.getSensorStateNewDictionary(ind,simnumber,'Updated')
                        if gv.UseSoftDataUpdating:
                            ZeroS_additional = np.concatenate(([ZeroS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim, [soft_lamda],Softbased_softvals,  np.ravel(ZeroS_templateNorm),np.ravel(Softbased_templatenorm)))
                        else:
                            ZeroS_additional = np.concatenate(([ZeroS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim, np.ravel(ZeroS_templateNorm)))                                                                                       
                        if globalExpCounter%gv.DecayIter==0 and globalExpCounter!=0:
                            DecayStart=DecayStart*Rate
                        if gv.UseSoftDataUpdating:
                            action,q=agent.step((ZeroS_sim)/gv.valuenormalization,(soft_data)/gv.valuenormalization,(ZeroS_additional)/gv.valuenormalization,compute_Q=False, apply_noise=False)
                        else:
                            action,q=agent.step((ZeroS_sim)/gv.valuenormalization,(ZeroS_additional)/gv.valuenormalization,compute_Q=False, apply_noise=False)
                        DictionaryGrid.simulationUpdating[simnumber][ind]=action
                ResultPlotting.outputVisualizationSimAfterUpdate(DictionaryGrid,simnumber,int(counter/trainingiter))
                ResultPlotting.outputSensorVsModelPredictionGraphIndividualSimUpdatedEval(DictionaryGrid,simnumber,int(counter/trainingiter),locationToPlot)
                print('Time taken to update', time.time()-startime)
            pbar.update(globalExpCounter)                                 
    pbar.finish()                                            
                                                                              
    return agent
# This funciton allows for the adapting the resources models with incoming data
def act(nbEpochs = gv.TrainingRandomPaths,centerMoment=False,
         rewardScale = 1.0,noiseType = 'ou_0.1', normalizeObservation=False, criticL2reg = 1e-3,
         actorLR = 1e-4, criticLR = 1e-3, gamma = 0.99, clipAction = None,tau=0.001, nbTrainSteps = 200,
         batchSize = gv.batchSize, paramNoiseAdaptInterval = 199, actionRange = (0,gv.ActionMaxRange), observationRange = (-1,1),
         actionShape=(1,), cnnShapeObservation =(((gv.numBlocksX*2+1)*(gv.numBlocksY*2+1)),),sess=None,clip_norm =1):

    if gv.UseSoftDataUpdating:
        additionalObservationShape = (gv.numSensors-1+2+1+1+1*gv.maxSamplesUpdatingDataEvent+1*gv.maxSamplesUpdatingDataEventSoft+3*gv.maxSamplesUpdatingDataEvent+3*gv.maxSamplesUpdatingDataEventSoft,) # Here 1 sensor + 1 sensor predict + 3 errors + num of samples dataevent in sim + num of samples data event soft
    else:
        additionalObservationShape = (gv.numSensors-1+2+1+1*gv.maxSamplesUpdatingDataEvent+3*gv.maxSamplesUpdatingDataEvent,) # Here 1 sensor+ 1 sensor predict+ 2 error + num of samples in dataevent in sim + DE config 3D
    actionNoise = None
    paramNoise = None
    if noiseType is not None:
        for currentNoiseType in noiseType.split(','):
            if currentNoiseType=='none':
                pass
            elif 'adaptive-param' in currentNoiseType:
                _, stddev= currentNoiseType.split('_')
                paramNoise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in currentNoiseType:
                _, stddev = currentNoiseType.split('_')
                actionNoise = NormalActionNoise(mu=np.zeros(actionShape[0]), sigma=float(stddev)*np.ones(actionShape[0]))
            elif 'ou' in currentNoiseType:
                _, stddev = currentNoiseType.split('_')
                actionNoise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actionShape[0]), sigma=float(stddev)*np.ones(actionShape[0]))
            else:
                raise RuntimeError('Unknown noise type')
    
    print('scaling actions with min {} and max {} before executing'.format(actionRange[0],actionRange[1]))
    print('scaling observations with min {} and max {}'.format(observationRange[0], observationRange[1]))
    memory = Memory(limit=int(1e6), action_shape=actionShape, observation_shape=cnnShapeObservation,additionalObservation_shape=additionalObservationShape)
    if gv.UseSoftDataUpdating:
        name = 'CNN2Input'
    else:
        name = 'CNN1Input'
    critic = Critic(network=name)
    actor = Actor(network=name)
    agent = DDPG(actor,critic,memory,cnnShapeObservation,additionalObservationShape,actionShape,
                 paramNoise,actionNoise,gamma=gamma,tau=tau,observation_range=observationRange, additional_observation_range=observationRange,
                 action_range=actionRange,critic_l2_reg=criticL2reg,actor_lr=actorLR, critic_lr=criticLR)
    print('Using agent with the following configuration:')
    print(str(agent.__dict__.items()))
    agent.initialize(sess)
    saver = tf.train.Saver()
    agent.initilizeSaver(saver)
    if os.path.isfile('./Output/TrainedModel/'+noiseType+'/'+agent.actor.networkName+'/MyModel.meta'):
        agent.load(saver,noiseType)
    sess.graph.finalize()
    Neigh = Utilities.cnnshape()
    DictionaryGrid,GridNode,Index, SensorIndex = inputFiles()
    DictionaryGrid.standardizeData()   
    ResultPlotting.outputVisualizationSoft(DictionaryGrid)
    ResultPlotting.outputVisualizationHard(DictionaryGrid)
    ResultPlotting.outputVisualizationTrueImage(DictionaryGrid)
    DictionaryGrid.computeNeighboursall(GridNode,Index,Neigh)
    ResultPlotting.outputSimAlloneFile(DictionaryGrid,gv.updateNumber,'Initial')
    Validation.validateHistogram(DictionaryGrid,'Initial')
    globalExpCounter = 0
    pbar = ProgressBar(widgets=widgets, maxval=gv.numSimTest*len(Index))
    pbar.start()
    locationToPlot = ResultPlotting.outputSensorPredictionGraph(DictionaryGrid)
    DictionaryGrid.reset()
    for simnumber in range(gv.numSimTest):
        print('Working on Sim {}'.format(simnumber))                                
        np.random.shuffle(Index)
        startime = time.time()
        for ind in Index:
                    NeighLooking = DictionaryGrid.dictionaryAllIndex.get(ind,-99)
                    if NeighLooking.shape[0]!=Neigh.shape[0]:
                        print('Error1')
                        exit()
                    else:
                        ZeroS_xyz = DictionaryGrid.data[ind]
                        NeighLooking_OS =  Neigh+ZeroS_xyz
                        ZeroS_sim,ZeroS_state, ZeroS_lamda_sim,ZeroS_cordpatch = DictionaryGrid.getSimulationState(NeighLooking_OS,simnumber,'Initial')
                        ZeroS_cord, ZeroS_valSim, ZeroS_valSoft, ZeroS_lookingIndex = DictionaryGrid.cordandValNearSamplesRadius(ZeroS_xyz,simnumber,'Simulation','Updated')
                        if gv.UseSoftDataUpdating:
                            Softbased_cord, Softbased_simvals, Softbased_softvals, Softbased_lookingIndex = DictionaryGrid.cordandValNearSamplesRadius(ZeroS_xyz,simnumber,'SoftSamples','Updated')
                            Softbased_template, Softbased_templatenorm, Softbased_tolsoft, Softbased_tolhard = DictionaryGrid.templateconfig(ZeroS_xyz,Softbased_cord)
                            soft_data, soft_lamda = DictionaryGrid.getSoftDataState(ZeroS_state)
                        ZeroS_template,ZeroS_templateNorm, ZeroS_tolsoft, ZeroS_tolHard = DictionaryGrid.templateconfig(ZeroS_xyz,ZeroS_cord)
                        ZeroS_sensor_pred,ZeroS_model_pred, ZeroS_lamda_sense, ZeroS_sensors_array = DictionaryGrid.getSensorStateNewDictionary(ind,simnumber,'Updated')
                        if gv.UseSoftDataUpdating:
                            ZeroS_additional = np.concatenate(([ZeroS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim, [soft_lamda],Softbased_softvals, np.ravel(ZeroS_templateNorm),np.ravel(Softbased_templatenorm)))
                        else:
                            ZeroS_additional = np.concatenate(([ZeroS_model_pred[-1]],[ZeroS_sensor_pred],[ZeroS_lamda_sim],[ZeroS_lamda_sense], ZeroS_valSim, np.ravel(ZeroS_templateNorm)))                                                                                       
                        if gv.UseSoftDataUpdating:
                            action,q=agent.step((ZeroS_sim)/gv.valuenormalization,(soft_data)/gv.valuenormalization,(ZeroS_additional)/gv.valuenormalization,compute_Q=False, apply_noise=False)
                        else:
                            action,q=agent.step((ZeroS_sim)/gv.valuenormalization,(ZeroS_additional)/gv.valuenormalization,compute_Q=False, apply_noise=False)
                        DictionaryGrid.simulationUpdating[simnumber][ind]=action
                        globalExpCounter+=1
                        pbar.update(globalExpCounter)
        print('Time taken to update', time.time()-startime)
    for simnumber in range(gv.numSimTest):
        ResultPlotting.outputVisualizationSim(DictionaryGrid,simnumber)    
        ResultPlotting.outputSensorVsModelPredictionGraphIndividualSimInitial(DictionaryGrid,simnumber,locationToPlot)           
        ResultPlotting.outputVisualizationSimTest(DictionaryGrid,simnumber)
        ResultPlotting.outputSensorVsModelPredictionGraphIndividualSimUpdated(DictionaryGrid,simnumber,locationToPlot)                                                                 
    pbar.finish()                                            
    ResultPlotting.outputSimAlloneFile(DictionaryGrid,gv.updateNumber,'Updated')
    Validation.validateHistogram(DictionaryGrid,'Updated')                                                                             