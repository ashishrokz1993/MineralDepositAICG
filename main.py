'''
Copyright (c) 2020, Ashish Kumar All rights reserved. 
'''
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import globalVars as gv
from ddpg_learner import learn, act

#log_File.configure(dir="tmp/testlogging")
TestTrainedPolicy = False
if gv.trainPolicy:
    with tf.Session() as sess:
        if TestTrainedPolicy:
            act(sess=sess)
        else:
            agent = learn(sess=sess)
else:
    with tf.Session() as sess:
        act(sess=sess)


