# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:28:16 2021

"""

import glob
import os
import time

import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# dataset path
dataset_path = os.path.join(ROOT_DIR, "saved_trajectoires")

# Inti the lists to store all the needed data
state_joints = []
action_joints = []
state_action_joints = []

print("[INFO] Parsing the Pandas states and actions  ...")

time.sleep(1)

exps = ['state', 'action']

for data_type in exps:
    # Looking inside the dataset for the state_joints and action_joints
    for exp in glob.glob(dataset_path + '/' + data_type + '_array_sin_*'):

        numpy_exp_array = np.load(exp)
        if data_type == 'state':
            for i in range(len(numpy_exp_array)):
                state_joints.append(numpy_exp_array[i])
        elif data_type == 'action':
            for i in range(len(numpy_exp_array)):
                action_joints.append(numpy_exp_array[i])

print("[INFO] The Parsing is done!")

# Convert the state_joints list to a numpy array then save it for training
state_joints = np.array(state_joints)

np.save('all_state_joints.npy', state_joints)

print("[INFO] State joints numy array is saved in the main folder!")

# Convert the action_joints list to a numpy array then save it for training
action_joints = np.array(action_joints)
np.save('all_action_joints.npy', action_joints)

print("[INFO] Action joints numy array is saved in the main folder!")

print("[INFO] State, action joints numpy array is saved in the main folder!")
