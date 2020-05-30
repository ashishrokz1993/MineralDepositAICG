'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module conntains the functions for validating histograms
'''
import numpy as np
import globalVars as gv
import matplotlib.pyplot as plt
import os
import pandas as pd
def plothist(data,bins=20, color=None, linestyle = None,linewidth=None, label=None):
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    plt.plot(x,y, color=color, linewidth=linewidth, label=label,linestyle=linestyle)
    return y,x
def validateHistogram(GridDictionary,name=None):
    bins = np.linspace(0,gv.ActionMaxRange,100)
    Y = []
    Name = []
    Name.append('Bins')
    if name=='Initial':
        hard,x = plothist(GridDictionary.hardData[:-1],bins=bins, color='black',linestyle='solid',linewidth=3,label='Data')
        Y.append(x)
        Y.append(hard)
        Name.append('Hard')
    else:
        hardData = GridDictionary.hardData[GridDictionary.hardData!=gv.NoDataValue]
        softData = GridDictionary.softData[GridDictionary.softData!=gv.NoDataValue]
        softup = softData+softData*(1-GridDictionary.softDataAccuracy[GridDictionary.softDataAccuracy!=gv.NoDataValue])
        softlow = softData-softData*(1-GridDictionary.softDataAccuracy[GridDictionary.softDataAccuracy!=gv.NoDataValue])
        hard,x  = plothist(GridDictionary.hardData[:-1],bins=bins, color='black',linestyle='solid',linewidth=3,label='Data')
        soft,x1  = plothist(softData,bins=bins, color='red',linestyle='solid',linewidth=3,label='Blasthole Tool Data')
        #softup,x2 = plothist(softup,bins=bins, color='blue',linestyle='dashed',linewidth=3,label='Blasthole Tool Data Upper')
        #softlow,x2 = plothist(softlow,bins=bins, color='green',linestyle='dashed',linewidth=3,label='Blasthole Tool Data Lower')
        Y.append(x)
        Y.append(hard)
        Name.append('Hard')
        Y.append(soft)
        Name.append('Soft')
        sensorPredicActual = GridDictionary.generateSensorPrediction(GridDictionary.locationSensorData)
        withoutNegativeSensorPrediction = sensorPredicActual[sensorPredicActual!=gv.NoDataValue]
        #sensor, x2 = plothist(withoutNegativeSensorPrediction,bins=bins,color='green',linestyle='solid',linewidth=3,label='Sensor')
        #Name.append('Sensor')
        #Y.append(sensor)
    for simnumber in range(gv.numSimTest):
        if simnumber==gv.numSimTest-1:
            if name=='Initial':
                sim,x  = plothist(GridDictionary.simulationData[simnumber][:-1],bins,'gray','solid',1,'Simulation')
            else:
                sim,x  = plothist(GridDictionary.simulationUpdating[simnumber][:-1],bins,'gray','solid',1,'Simulation')
        else:
            if name=='Initial':
                sim,x  =plothist(GridDictionary.simulationData[simnumber][:-1],bins,'gray','solid',1)
            else:
                sim,x  =plothist(GridDictionary.simulationUpdating[simnumber][:-1],bins,'gray','solid',1)
        Y.append(sim)
        Name.append('Sim')
    plt.title('Histogram: '+name+' Simulation',fontsize=25)
    plt.xlabel('Grade',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.legend(loc=0,fontsize=15)
    if not os.path.isdir('./Output/Validation/'+'Section'+str(gv.sectionalZ)+'/Update'+str(gv.updateNumber)):
        os.makedirs('./Output/Validation/'+'Section'+str(gv.sectionalZ)+'/Update'+str(gv.updateNumber)+'/Histogram')
    plt.savefig('./Output/Validation/'+'Section'+str(gv.sectionalZ)+'/Update'+str(gv.updateNumber)+'/Histogram/'+name+'simulation.png')
    plt.clf()
    OuputResult = pd.DataFrame(np.transpose(Y))
    OuputResult.columns = Name
    OuputResult.to_csv('./Output/Validation/'+'Section'+str(gv.sectionalZ)+'/Update'+str(gv.updateNumber)+'/Histogram/'+name+'.csv',index=False)
def outputSimAlloneFile(GriDictionary,updatenumber, name=None):
    print('Saving the simulation files to folder Update', gv.updateNumber)
    filename = 'Output/Update'+str(updatenumber)+'/Section'+str(gv.sectionalZ)+'/SimAll'+'.csv'
    xyz = np.array(GriDictionary.Tree.data)
    allvals = np.concatenate((xyz,np.array(GriDictionary.simulationData[:,:-1]).T),axis=1)
    OuputResult = pd.DataFrame(allvals)
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