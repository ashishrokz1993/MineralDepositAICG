'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module constains the functions for generating the GIF to visualize the trianing and testing
'''
import globalVars as gv
import imageio
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import numpy as np
def generateTIF():
    imagesInitial = []
    concatimage = []
    imagesUpdated = []
    imagesEval = []
    kargs = { 'duration': 0.2 }
    for simnumber in range(gv.numSimTrain):
        evalIndex = 0
        for index in range(1,gv.TotalImages):
            if index%gv.ImagesGifindex==0:
                evalIndex+=1           
                OriginalSim = 'Output/TrainResult/InitialSection'+str(gv.sectionalZ)+'Sim'+str(simnumber)+'.png'
                softData ='Output/TrainResult/Section'+str(gv.sectionalZ)+'Soft.png'        
                SensorPredicitonInitial = 'Output/TrainResult/InitialSection'+str(gv.sectionalZ)+'Sim'+str(simnumber)+'Sensor.png'
                UpdatedSimTrain = 'Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(index)+'Sim'+str(simnumber)+'.png'
                UpdatedSimEval = 'Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(evalIndex)+'UpdatedSim'+str(simnumber)+'.png'       
                SensorPredicitonTrain = 'Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(index)+'Sim'+str(simnumber)+'Sensor.png'
                SensorPredicitonEval = 'Output/TrainResult/Section'+str(gv.sectionalZ)+'Iter'+str(evalIndex)+'UpdatedSim'+str(simnumber)+'Sensor.png'
                HardData = 'Output/TrainResult/Section'+str(gv.sectionalZ)+'Hard.png' 
                imagesInitial.append([HardData,OriginalSim,SensorPredicitonInitial])
                imagesUpdated.append([softData,UpdatedSimTrain,SensorPredicitonTrain])
                imagesEval.append([softData,UpdatedSimEval,SensorPredicitonEval])

    for j in range(len(imagesInitial)):
        print('Working on image', j)
        initialImgs = [Image.open(i) for i in imagesInitial[j]]
        trainImgs = [Image.open(i) for i in imagesUpdated[j]]
        evalImgs = [Image.open(i) for i in imagesEval[j]]
        min_shape_initial = sorted( [(np.sum(i.size), i.size ) for i in initialImgs])[0][1]
        min_shape_train = sorted( [(np.sum(i.size), i.size ) for i in trainImgs])[0][1]
        min_shape_eval = sorted( [(np.sum(i.size), i.size ) for i in evalImgs])[0][1]

        initialCombImgs = np.hstack((np.asarray(i.resize(min_shape_initial)) for i in initialImgs))
        initial_comb_image = Image.fromarray(initialCombImgs)

        trainCombImgs = np.hstack((np.asarray(i.resize(min_shape_train)) for i in trainImgs))
        train_comb_image = Image.fromarray(trainCombImgs)

        evalCombImgs = np.hstack((np.asarray(i.resize(min_shape_eval)) for i in evalImgs))
        eval_comb_image = Image.fromarray(evalCombImgs)

        gifImage = [initial_comb_image,train_comb_image,eval_comb_image]
        min_shape_gif = sorted( [(np.sum(i.size), i.size ) for i in gifImage])[0][1]
        gifCombImgs = np.vstack( (np.asarray( i.resize(min_shape_gif) ) for i in gifImage ) )
        gif_comb_Image = Image.fromarray(gifCombImgs)
        gif_comb_Image.save('Output/Visualization/Combined'+str(j)+'.png')
        concatimage.append(gifCombImgs)
    imageio.mimsave('Output/Visualization/Video.gif',concatimage, 'GIF', **kargs)


