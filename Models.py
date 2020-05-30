'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module contains the classes and function for actor and critic agent
'''
import numpy as np
import tensorflow as tf
from Networks import get_network_builder
import globalVars as gv
# The structure of the code is followed form openAI github repository. https://github.com/openai/baselines/tree/master/baselines/ddpg

class Model(object):
    def __init__(self, name, network='CNN1Input'):
        self.name = name
        self.network_builder = get_network_builder(network)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def trainableVars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def perturbableVars(self):
        return [var for var in self.trainableVars() if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions=1,nb_hidden=100,nb_hidden2=50, name='actor',network=None):
        super().__init__(name=name, network=network)
        self.nb_actions = nb_actions
        self.nb_hidden = nb_hidden
        self.nb_hidden2 = nb_hidden2
        self.networkName = network
    if gv.UseSoftDataUpdating:
        def __call__(self, SimtZero,obscnnSoft,additionalObs, reuse=False):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                with tf.device('/gpu:0'):
                    x = self.network_builder(SimtZero,obscnnSoft)
                    x= tf.concat([x,additionalObs],1)
                    x = tf.layers.dense(x, self.nb_hidden, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_hidden2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_actions,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                return x
    else:
        def __call__(self, SimtZero,additionalObs, reuse=False):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                with tf.device('/gpu:0'):
                    x = self.network_builder(SimtZero)
                    x= tf.concat([x,additionalObs],1)
                    x = tf.layers.dense(x, self.nb_hidden, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_hidden2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_actions,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                return x
class Critic(Model):
    def __init__(self,nb_hidden=100,nb_hidden2=50,name='critic',network=None):
        super().__init__(name=name, network=network)
        self.nb_hidden = nb_hidden
        self.nb_hidden2 = nb_hidden2
        self.nb_output =1
        self.networkName = network
    if gv.UseSoftDataUpdating:
        def __call__(self, SimtZero,action,obscnnSoft,additionalObs, reuse=False):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
               with tf.device('/gpu:0'):
                    x = self.network_builder(SimtZero,obscnnSoft)
                    x= tf.concat([x,additionalObs,action],1)
                    x = tf.layers.dense(x, self.nb_hidden, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_hidden2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_output,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),name='output')
               return x
    else:
        def __call__(self, SimtZero,action,additionalObs, reuse=False):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
               with tf.device('/gpu:0'):
                    x = self.network_builder(SimtZero)
                    x= tf.concat([x,additionalObs,action],1)
                    x = tf.layers.dense(x, self.nb_hidden, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_hidden2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.dense(x,self.nb_output,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),name='output')
               return x

    def outputVars(self):
        output_vars = [var for var in self.trainableVars if 'output' in var.name]
        return output_vars
