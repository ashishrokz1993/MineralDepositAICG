'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
'''
import numpy as np
import globalVars as gv
'''
This module store various utilities funcitons required within the project. 
This include create a mask for CNN shape 
Also the reward evaluation with High-order and central moments
'''
# Use this function cnnshape.
def cnnshape():
    shapeArray = []
    if gv.working2D:
        xArray = np.arange(-gv.numBlocksX,gv.numBlocksX+1,1)
        yArray = np.arange(-gv.numBlocksY,gv.numBlocksY+1,1)
        for i in yArray:
            for j in xArray:
                shapeArray.append([j*gv.xsize,i*gv.ysize,0])
    else:
        xArray = np.arange(-gv.numBlocksX,gv.numBlocksX+1,1)
        yArray = np.arange(-gv.numBlocksY,gv.numBlocksY+1,1)
        zArray = np.arange(-gv.numBlockZ,gv.numBlockZ+1,1)
        for i in zArray:
            for j in yArray:
                for k in xArray:
                    shapeArray.append([k*gv.xsize,j*gv.ysize,i*gv.zsize])
    return np.array(shapeArray)
# This funciton computes the reward using high-order pdf and is generic with type of data. 
if gv.UseSoftDataUpdating:
    def computeRewardHOStatsNew1(prob_original_simulation, prob_updated_simulation,OS_sensor_pred,NS_model_pred,OS_model_pred,OS_lamda_sim,OS_lamda_sense, prob_original_soft, prob_updated_soft, OS_lamda_soft,oldval,newval):
        reward = 0
        modelPredNS = np.average(NS_model_pred)
        modelpredOS = np.average(OS_model_pred)
        if OS_lamda_sense<0:
            lamdaSense = 0
        else:
            lamdaSense = 1-OS_lamda_sense
        if OS_lamda_sim<0:
            lamdaSim = 0
        else:
            lamdaSim = OS_lamda_sim  # this was done to ensure that the accuracy is low
        if OS_lamda_soft<0:
            lamdaSoft = 0
        else:
            lamdaSoft = 1-OS_lamda_soft
        if prob_original_simulation>0:
            if lamdaSense<0 and lamdaSoft<0:
                reward+=(prob_updated_simulation-prob_original_simulation)*(lamdaSim/(lamdaSim+lamdaSense+lamdaSoft))
            else:
                reward+=(prob_updated_simulation-prob_original_simulation)*(lamdaSim/(lamdaSim+lamdaSense+lamdaSoft))
        if not OS_sensor_pred<=0:
            reward-=(np.abs(OS_sensor_pred-modelPredNS)-np.abs(OS_sensor_pred-modelpredOS))*(lamdaSense/(lamdaSim+lamdaSense+lamdaSoft))*gv.sensorrewardnormalization
        if prob_original_soft>0:
            reward+=(prob_updated_soft-prob_original_soft)*(lamdaSoft/(lamdaSim+lamdaSense+lamdaSoft))
        return reward
else:
    def computeRewardHOStatsNew1(prob_original_simulation, prob_updated_simulation,OS_sensor_pred,NS_model_pred,OS_model_pred,OS_lamda_sim,OS_lamda_sense):
        reward = 0
        modelPredNS = np.average(NS_model_pred)
        modelpredOS = np.average(OS_model_pred)
        if OS_lamda_sense<0:
            lamdaSense = 0
        else:
            lamdaSense = 1-OS_lamda_sense
        if OS_lamda_sim<0:
            lamdaSim = 0
        else:
            lamdaSim = OS_lamda_sim # this was done to ensure that the sim accuracy is low
        if prob_original_simulation>0:
            if lamdaSense>0:
                reward+=(prob_updated_simulation-prob_original_simulation)*(lamdaSim/(lamdaSim+lamdaSense))
            else:
                reward+=(prob_updated_simulation-prob_original_simulation)*(lamdaSim/(lamdaSim+lamdaSense))
        if not OS_sensor_pred<=0:
            reward-=(np.abs(OS_sensor_pred-modelPredNS)-np.abs(OS_sensor_pred-modelpredOS))*(lamdaSense/(lamdaSim+lamdaSense))*gv.sensorrewardnormalization

        return reward

