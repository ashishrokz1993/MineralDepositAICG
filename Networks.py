'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module contains parts of the actor and critic agnet networks
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import globalVars as gv

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk
def cnn_model(X):
    poolSize = 2
    numPools =2
    filter1= 32
    filter2 = 32
    xshape = int(gv.numBlocksX*2+1)
    yshape = int(gv.numBlocksY*2+1)
    zshape = int(gv.numBlockZ*2+1)
    lastShapex = xshape
    lastShapey = yshape
    kernelSize = 3
    denselayersize = 128
    with tf.device('/gpu:0'):
        if gv.working2D:
            input_layer = tf.reshape(X,[-1,xshape,yshape, 1])
            conv1 = tf.layers.conv2d(inputs=input_layer, filters=8, kernel_size=5,strides=1, activation=tf.nn.relu, padding='valid')
            conv2 = tf.layers.conv2d(inputs=conv1,filters=8,kernel_size=3,strides=1, padding='valid', activation=tf.nn.relu)
            flat = tf.reshape(conv2,[-1,conv2.shape[1].value*conv2.shape[2].value*conv2.shape[3].value])
            fc = tf.layers.dense(flat,denselayersize,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            input_layer = tf.reshape(X,[-1,xshape,yshape,zshape])
            conv1 = tf.layers.conv3d(inputs=input_layer, filters=8, kernel_size=5,strides=1, activation=tf.nn.relu, padding='valid')
            conv2 = tf.layers.conv3d(inputs=conv1,filters=8,kernel_size=3,strides=1, padding='valid', activation=tf.nn.relu)
            flat = tf.reshape(conv2,[-1,conv2.shape[1].value*conv2.shape[2].value*conv2.shape[3].value]) # might need some fixes haven't checked for 3d yet
            fc = tf.layers.dense(flat,denselayersize,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
    return fc

@register("CNN1Input")
def cnn_Network1(X):
    return cnn_model(X)

@register("CNN2Input")
def cnn_Network2(X1,X2):
    with tf.device('/gpu:0'):
        Input1Flat = cnn_model(X1)
        Input2Flat = cnn_model(X2)
        FinalFlat = tf.concat([Input1Flat,Input2Flat],1)
    return FinalFlat

@register("CNN3Input")
def cnn_Network3(X1,X2,X3):
    with tf.device('/gpu:0'):
        Input1Flat = cnn_model(X1)
        Input2Flat = cnn_model(X2)
        Input3Flat = cnn_model(X3)
        FinalFlat = tf.concat([Input1Flat,Input2Flat, Input3Flat],1)
    return FinalFlat
@register("CNN4Input")
def cnn_Network4(X1,X2,X3,X4):
    with tf.device('/gpu:0'):
        Input1Flat = cnn_model(X1)
        Input2Flat = cnn_model(X2)
        Input3Flat = cnn_model(X3)
        Input4Flat = cnn_model(X4)
        FinalFlat = tf.concat([Input1Flat,Input2Flat, Input3Flat,Input4Flat],1)
    return FinalFlat

@register('MLP2Input')
def mlp2Input(x1,x2,num_layers=4,num_hidden=800):
    with tf.device('/gpu:0'):
        inputLayer = tf.concat(x1,x2)
    return inputLayer


def fc(input,num_layer = 2):

    return 0


def get_network_builder(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))