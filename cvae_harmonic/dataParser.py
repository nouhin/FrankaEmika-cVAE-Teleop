# Import the necessary libraries
import csv
import os

import numpy as np
from tqdm import tqdm

print("\n[INFO] Loading data...\n")

# Set root path
ROOT_DIR = os.path.abspath("./")

# Data folder
DATA_FOLDER = os.path.join(ROOT_DIR, "Dataset/", "harmonic_0.5.0")

# List of participant folder names
participants = list(filter(lambda x: os.path.isdir(os.path.join(DATA_FOLDER, x)), os.listdir(DATA_FOLDER)))

# List of (state,action) pairs
state_action = []

# Stores the number of loaded demonstrations
loaded = 0

for p in tqdm(participants):

    # If you don't want to load all the dataset, please uncomment next line and set the number of runs to load
    # if loaded>=20: break
    for r in os.listdir(os.path.join(DATA_FOLDER, p, "run")):
        loaded += 1
        with open(os.path.join(DATA_FOLDER, p, "run", r, "text_data", "joint_positions.csv"), newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            # Ignore header
            next(reader, None)
            for row in reader:
                temp = row[3:11] + row[11:19]

                temp = list(map(float, temp))

                # Ignore (state,action) pair if all velocities are zero
                if np.mean([abs(t) for t in temp[8:16]]) == 0:
                    continue

                # To ignore some outliers present in HARMONIC dataset
                if np.min(temp[0:8]) < -10 or np.max(temp[0:8]) > 10 or np.min(temp[8:16]
                                                                              ) < -1 or np.max(temp[8:16]) > 1:
                    continue
                temp_ = [s / 10 for s in temp[0:8]] + temp[8:16]
                state_action.append(temp_)

# Split and save data (train 80%, validation 10%, test 10%)
np.save("train.npy", np.array(state_action[0:int(0.8 * len(state_action))]))

np.save("val.npy", np.array(state_action[int(0.8 * len(state_action)):int(0.9 * len(state_action))]))

np.save("test.npy", np.array(state_action[int(0.9 * len(state_action)):-1]))

print("\n[INFO] Data successfully loaded!\n Train :",len(state_action[0:int\
        (0.8*len(state_action))]),"\n Val :",len(state_action[int(0.8*len\
        (state_action)):int(0.9*len(state_action))]),"\n Test :",\
         len(state_action[int(0.9*len(state_action)):-1]))
