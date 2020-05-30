'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
This module contains the class and functions for DDPG
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import log_File
import globalVars as gv
from copy import copy, deepcopy
from functools import reduce
import Adam
import os
import time
##########################################################################################################################
# Set up the target updates
def getTargetUpdates(vars, target_vars, tau):
    print('Setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        print('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)
##########################################################################################################################
# Computer the variance
def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

##########################################################################################################################
# Compute the standard deviation
def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

##########################################################################################################################
# Get pertubed actor weights
def getPerturbedActorUpdates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars()) == len(perturbed_actor.vars())
    assert len(actor.perturbableVars()) == len(perturbed_actor.perturbableVars())
    updates = []
    for var, perturbed_var in zip(actor.vars(), perturbed_actor.vars()):
        if var in actor.perturbableVars():
            print('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            print('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars())
    return tf.group(*updates)
##########################################################################################################################
# shape adjustment for feeding into tf placeholders
def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters:
        placeholder     tensorflow input placeholder

        data            input data to be (potentially) reshaped to be fed into placeholder

    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]
    return np.reshape(data, placeholder_shape)
##########################################################################################################################
# Main class of Deep Deterministic Policy Gradient - It performs all the necessary operations related to finding a policy with DDPG
class DDPG(object):
    def __init__(self,actor, critic, memory,observation_shape, addtional_observation_shape,
                 action_shape, param_noise=None,action_noise=None,gamma=0.99,tau=0.001,
                 normalize_observation=False, normalize_additional_observation=False,
                 batch_size=gv.batchSize, observation_range=(-1,1), additional_observation_range = (-1,1),
                 action_range = (0,gv.ActionMaxRange), critic_l2_reg = 0., actor_lr=1e-4, critic_lr = 1e-3,
                 clip_norm=None,clip_obs = None, reward_scale=1.,return_range=(-np.inf,np.inf)):
        self.obs0= tf.placeholder(tf.float32, shape=(None,)+observation_shape,name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,)+observation_shape,name='obs1')
        self.obs0_additional = tf.placeholder(tf.float32, shape=(None,)+addtional_observation_shape,name='obs0_additional')
        self.obs1_additional = tf.placeholder(tf.float32, shape=(None,)+addtional_observation_shape,name='obs1_additional')
        if gv.UseSoftDataUpdating:
            self.obs_soft = tf.placeholder(tf.float32,shape=(None,)+observation_shape, name='obs_soft')
        self.rewards = tf.placeholder(tf.float32,shape=(None,1), name='rewards')
        self.actions = tf.placeholder(tf.float32,shape=(None,)+action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32,shape=(None,1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32,shape=(), name='param_noise_stddev')

        #Parameters
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normlize_observation = normalize_observation
        self.normalize_additional_observation = normalize_additional_observation
        self.action_noise =action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.clip_obs = clip_obs
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.return_range = return_range

        # Create target networks
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Normalize Observation
        '''
        This is not implemented yet because all the values in the observation are between -1 and 1.
        Also the rewards are between -1 and 1. 
        The actions needs to be clipped to make sure it only output between 0 and 1
        if required implement normalized fucntion with clip
        '''
        # Clip Observation
        clipped_obs_0 = tf.clip_by_value(self.obs0,self.observation_range[0], self.observation_range[1])
        clipped_obs_1 = tf.clip_by_value(self.obs1,self.observation_range[0],self.observation_range[1])
        if gv.UseSoftDataUpdating:
            clipped_obs_soft = tf.clip_by_value(self.obs_soft,self.observation_range[0],self.observation_range[1])
        clipped_obs0_additional = tf.clip_by_value(self.obs0_additional, self.observation_range[0], self.observation_range[1])
        clipped_obs1_additional = tf.clip_by_value(self.obs1_additional, self.observation_range[0], self.observation_range[1])
        # Create networks and core TF parts
        if gv.UseSoftDataUpdating:
            self.actor_tf = actor(clipped_obs_0, clipped_obs_soft,clipped_obs0_additional)
            self.critic_tf = critic(clipped_obs_0,self.actions,clipped_obs_soft,clipped_obs0_additional)
            self.critic_with_actor_tf = critic(clipped_obs_0,self.actor_tf,clipped_obs_soft,clipped_obs0_additional,reuse=True)
            Q_obs1 = target_critic(clipped_obs_1,target_actor(clipped_obs_1,clipped_obs_soft,clipped_obs1_additional),clipped_obs_soft,clipped_obs1_additional)
            self.target_Q = self.rewards+gamma*Q_obs1
        else:
            self.actor_tf = actor(clipped_obs_0, clipped_obs0_additional)
            self.critic_tf = critic(clipped_obs_0,self.actions,clipped_obs0_additional)
            self.critic_with_actor_tf = critic(clipped_obs_0,self.actor_tf,clipped_obs0_additional,reuse=True)
            Q_obs1 = target_critic(clipped_obs_1,target_actor(clipped_obs_1,clipped_obs1_additional),clipped_obs1_additional)
            self.target_Q = self.rewards+gamma*Q_obs1

        # Set up noise and optimizer
        if self.param_noise is not None:
            if gv.UseSoftDataUpdating:
                self.setupParamNoise(self.obs0,self.obs_soft,self.obs0_additional)
            else:
                self.setupParamNoise(self.obs0,self.obs0_additional)
        self.setupActorOptimizer()
        self.setupCriticOptimizer()
        self.setupTargetNetworkUpdates()

    #####################################################################################################################
    # Set up the updates for target network
    def setupTargetNetworkUpdates(self):
        actorInitialUpdates, actorSoftUpdates = getTargetUpdates(self.actor.vars(),self.target_actor.vars(), self.tau)
        criticInitialUdpates, criticSoftUpdates = getTargetUpdates(self.critic.vars(),self.target_critic.vars(),self.tau)
        self.targetInitialUpdates = [actorInitialUpdates,criticInitialUdpates]
        self.targetSoftUpdates = [actorSoftUpdates,criticSoftUpdates]
    if not gv.UseSoftDataUpdating:
        #####################################################################################################################
        # Set up the parametric noise for the weights of the neural network instead of the action noise
        def setupParamNoise(self,obs0, obs0_additional):
            assert self.param_noise is not None

            # Configure pertubed actor
            param_noise_actor = copy(self.actor)
            param_noise_actor.name = 'param_noise_actor'
            self.pertubed_actor_tf = param_noise_actor(obs0,obs0_additional)
            print('Setting up paramateric noise')
            self.pertub_policy_ops = getPerturbedActorUpdates(self.actor,param_noise_actor,self.param_noise_stddev)

            #Configure copy for stddev adaptation
            adaptive_param_noise_actor = copy(self.actor)
            adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
            adaptive_actor_tf = adaptive_param_noise_actor(obs0,obs0_additional)
            self.perturb_adaptive_policy_ops = getPerturbedActorUpdates(self.actor,adaptive_param_noise_actor,self.param_noise_stddev)
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf-adaptive_actor_tf)))
    else: 
        #####################################################################################################################
        # Set up the parametric noise for the weights of the neural network instead of the action noise
        def setupParamNoise(self,obs0,obs_soft, obs0_additional):
            assert self.param_noise is not None

            # Configure pertubed actor
            param_noise_actor = copy(self.actor)
            param_noise_actor.name = 'param_noise_actor'
            self.pertubed_actor_tf = param_noise_actor(obs0,obs_soft,obs0_additional)
            print('Setting up paramateric noise')
            self.pertub_policy_ops = getPerturbedActorUpdates(self.actor,param_noise_actor,self.param_noise_stddev)

            #Configure copy for stddev adaptation
            adaptive_param_noise_actor = copy(self.actor)
            adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
            adaptive_actor_tf = adaptive_param_noise_actor(obs0,obs_soft,obs0_additional)
            self.perturb_adaptive_policy_ops = getPerturbedActorUpdates(self.actor,adaptive_param_noise_actor,self.param_noise_stddev)
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf-adaptive_actor_tf)))
    #####################################################################################################################
    # Set up the optimizer for actor network
    def setupActorOptimizer(self):
        print('Setting up actor optimizer')
        self.actor_loss=-tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape(). as_list() for var in self.actor.trainableVars()]
        actor_nb_params = sum([reduce(lambda x, y: x*y, shape) for shape in actor_shapes])
        print('Actor shapes: {}'.format(actor_shapes))
        print('Actor parameters: {}'.format(actor_nb_params))
        self.actor_grads = Adam.flatgrad(self.actor_loss, self.actor.trainableVars(), clip_norm=self.clip_norm)
        self.actor_optimizer = Adam.Adam(var_list=self.actor.trainableVars())
    #####################################################################################################################
    # Set up the optimizer for the critic network
    def setupCriticOptimizer(self):
        print('Setting up critic optimizer')
        critic_target_tf = tf.clip_by_value(self.critic_target, self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_tf-critic_target_tf))
        if self.critic_l2_reg>0.:
            critic_reg_vars = [var for var in self.critic.trainableVars() if var.name.endswith('/kernel:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                print('Regularizing: {}'.format(var.name))
            print('Applying l2 regularizaion with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(tc.layers.l2_regularizer(self.critic_l2_reg),weights_list=critic_reg_vars)
            self.critic_loss+=critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainableVars()]
        critic_nb_params = sum([reduce(lambda x, y: x*y, shape) for shape in critic_shapes])
        print('Critic shapes: {}'.format(critic_shapes))
        print('Critic params: {}'.format(critic_nb_params))
        self.critic_grads = Adam.flatgrad(self.critic_loss, self.critic.trainableVars(), clip_norm=self.clip_norm)
        self.critic_optimizer = Adam.Adam(var_list=self.critic.trainableVars())
    #####################################################################################################################
    if gv.UseSoftDataUpdating:
        # Takes an action 
        def step(self,obs0, obs_soft,obs0_additional, apply_noise=True, compute_Q=True):
            if self.param_noise is not None and apply_noise:
                actor_tf = self.pertubed_actor_tf
            else:
                actor_tf = self.actor_tf
            feed_dict = {self.obs0:adjust_shape(self.obs0, [obs0]), self.obs0_additional:adjust_shape(self.obs0_additional,[obs0_additional]),self.obs_soft:adjust_shape(self.obs_soft,[obs_soft])}
            if compute_Q:
                action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
            else:
                action = self.sess.run(actor_tf, feed_dict=feed_dict)
                q = None
            if self.action_noise is not None and apply_noise:
                noise = self.action_noise()
                assert noise.shape==action[0].shape
                action+=noise
            action=np.clip(action, self.action_range[0], self.action_range[1])
            return action,q
        #####################################################################################################################
        # Store the transition in the memory
        def storeTransition(self,obs0, action, obs_soft,reward, obs0_additional, obs1,obs1_additional):
            #reward*=self.reward_scale
            B = obs0.shape[0]
            for b in range(B):
                self.memory.append(obs0[b],obs0_additional[b],action[b],obs_soft[b],reward[b],obs1[b], obs1_additional[b])
        #####################################################################################################################
    else: 
        def step(self,obs0, obs0_additional, apply_noise=True, compute_Q=True):
            if self.param_noise is not None and apply_noise:
                actor_tf = self.pertubed_actor_tf
            else:
                actor_tf = self.actor_tf
            feed_dict = {self.obs0:adjust_shape(self.obs0, [obs0]), self.obs0_additional:adjust_shape(self.obs0_additional,[obs0_additional])}
            if compute_Q:
                action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
            else:
                action = self.sess.run(actor_tf, feed_dict=feed_dict)
                q = None
            if self.action_noise is not None and apply_noise:
                noise = self.action_noise()
                assert noise.shape==action[0].shape
                action+=noise
            action=np.clip(action, self.action_range[0], self.action_range[1])
            return action,q
        #####################################################################################################################

        # Store the transition in the memory
        def storeTransition(self,obs0, action, reward, obs0_additional, obs1,obs1_additional):
            #reward*=self.reward_scale
            B = obs0.shape[0]
            for b in range(B):
                self.memory.append(obs0[b],obs0_additional[b],action[b],reward[b],obs1[b], obs1_additional[b])
        #####################################################################################################################
    # Train the agent with the batch
    def train(self):
        batch = self.memory.sample(batch_size=self.batch_size)
        if gv.UseSoftDataUpdating:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1:batch['obs1'],
                self.rewards:batch['rewards'],
                self.obs1_additional:batch['obs1_additional'],
                self.obs_soft:batch['obs_soft']
                })
            ops = [self.actor_grads,self.actor_loss,self.critic_grads,self.critic_loss]
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
                self.obs0:batch['obs0'], 
                self.obs0_additional:batch['obs0_additional'],
                self.obs_soft:batch['obs_soft'],
                self.actions:batch['actions'],
                self.critic_target:target_Q})
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1:batch['obs1'],
                self.rewards:batch['rewards'],
                self.obs1_additional:batch['obs1_additional']
                })
            ops = [self.actor_grads,self.actor_loss,self.critic_grads,self.critic_loss]
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
                self.obs0:batch['obs0'], 
                self.obs0_additional:batch['obs0_additional'],
                self.actions:batch['actions'],
                self.critic_target:target_Q})
        self.actor_optimizer.update(actor_grads,stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads,stepsize=self.critic_lr)
        return critic_loss,actor_loss
    #####################################################################################################################
    # Initilize the agents and netowrk graphs with a session
    def initialize(self,sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.targetInitialUpdates)
    #####################################################################################################################
    # Initilize the saver
    def initilizeSaver(self,saver):
        self.saver =saver
    #####################################################################################################################
    # Update the target network Q'
    def updateTargetNet(self):
        self.sess.run(self.targetSoftUpdates)
    #####################################################################################################################
    # Adapt the parametric noise 
    def adaptParamNoise(self):
        if self.param_noise is None:
            return 0
        batch=self.memory.sample(batch_size=self.batch_size)
        if gv.UseSoftDataUpdating:
            self.sess.run(self.perturb_adaptive_policy_ops,feed_dict={self.obs0:batch['obs0'],  self.obs0_additional:batch['obs0_additional'], self.obs_soft:batch['obs_soft'],self.param_noise_stddev:self.param_noise.current_stddev})
            distance = self.sess.run(self.adaptive_policy_distance,feed_dict={self.obs0:batch['obs0'], self.obs0_additional:batch['obs0_additional'],self.obs_soft:batch['obs_soft'],self.param_noise_stddev:self.param_noise.current_stddev})
            self.param_noise.adapt(distance)
        else:
            self.sess.run(self.perturb_adaptive_policy_ops,feed_dict={self.obs0:batch['obs0'],  self.obs0_additional:batch['obs0_additional'], self.param_noise_stddev:self.param_noise.current_stddev})
            distance = self.sess.run(self.adaptive_policy_distance,feed_dict={self.obs0:batch['obs0'], self.obs0_additional:batch['obs0_additional'],self.param_noise_stddev:self.param_noise.current_stddev})
            self.param_noise.adapt(distance)
        return distance
    #####################################################################################################################
    # Reset the action noise and paramateric noise state
    def reset(self):
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.pertub_policy_ops, feed_dict={self.param_noise_stddev:self.param_noise.current_stddev})
    #####################################################################################################################
    # Load the saved network
    def load(self,saver,noisetype):
        saver.restore(self.sess,'./Output/TrainedModel/'+noisetype+'/'+self.actor.networkName+'/MyModel')
        print('Model loaded sucessfully')
    #####################################################################################################################
    # Save the networks
    def save(self,noisetype):
        if not os.path.isdir('./Output/TrainedModel/'+noisetype+'/'+self.actor.networkName):
            os.makedirs('./Output/TrainedModel/'+noisetype+'/'+self.actor.networkName)
        savepath = self.saver.save(self.sess,'./Output/TrainedModel/'+noisetype+'/'+self.actor.networkName+'/MyModel')
        print("Model saved in path: %s" % savepath)




