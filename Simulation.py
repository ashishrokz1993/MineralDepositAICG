import numpy as np
import globalVars as gv
import ResultPlotting
from progressbar import *
from copy import deepcopy
from multiprocessing import Process
from joblib import Parallel, delayed
widgets = ['Completed: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
           ' ', ETA()] #see docs for other options
def runSimulationFunction(simnumber,simulationPath,hardDataDictionarySimulation,GridNode,DictionaryTISimulation,scaler):
    np.random.shuffle(simulationPath)
    TotalPoint = len(simulationPath)
    counter=0
    count = 0
    pbar = ProgressBar(widgets=widgets, maxval=TotalPoint)
    pbar.start()
    DataConditional = deepcopy(hardDataDictionarySimulation)
    for nodes in simulationPath:
        XYZ = (GridNode[nodes].x, GridNode[nodes].y, GridNode[nodes].z)
        cord,samples = DataConditional.cordandValNearSamplesRadius(XYZ,gv.SearchRadius)
        template,tolerance = DataConditional.templateconfig(XYZ,cord)
        if gv.UseTrainingImageSimulation:
            sampledVal = DictionaryTISimulation.samplecpdf(template,samples,tolerance,XYZ)
        else:
            sampledVal = hardDataDictionarySimulation.samplecpdf(template,samples,tolerance,XYZ)
        if sampledVal==-9999:
            sampledVal = np.average(samples)
            GridNode[nodes].simVal = sampledVal
            count+=1
        else:
            GridNode[nodes].simVal = sampledVal
            DataConditional.addpoint(XYZ,sampledVal)
        counter+=1
        pbar.update(counter)
    pbar.finish()
    print('Number of points simulated with average', count)
    print('Total points simulated', counter-count)
    ResultPlotting.outputSchedule(GridNode,gv.updateNumber,simnumber,scaler)

