# Import the necessary libraries
from model import  CVAE
import torch
import torch.nn.functional as F
import os
import numpy as np


# Sets device to gpu if available , else cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = 8                  # size of each input to reconstract
HIDDEN_DIM = [200,100]         # size of hidden layers
LATENT_DIM = 2                 # latent vector dimension

# Load test data
test_dataset=np.load("test.npy")


# loads checkpoint present in root path
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])    
    return model


# root path
ROOT_DIR = os.path.abspath("./")

# Checkpoint (trained model) name
checkpoint="epoch_3_val_loss_0.0019807365232642053_checkpoint.pth.tar"

# The path to the trained model
checkpoint_path = os.path.join(ROOT_DIR, checkpoint)
            
# Instantiate a model
model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# Load weights of model to evaluate
model=load_checkpoint(model, checkpoint_path)


# set the evaluation mode
model.eval()

test_loss=0

# we don't need to track the gradients, since we are not updating the parameters 
# during evaluation
with torch.no_grad():
    for i in range(test_dataset.shape[0]):
        
        # extract actions from data
        action = test_dataset[i][-8:16].reshape(1,8)
        action = torch.Tensor(action)   
        action = action.float()
        action = action.to(device)

        # extract states from data
        state = test_dataset[i][0:8].reshape(1,8)
        state = torch.Tensor(state)
        state = state.float()
        state = state.to(device)

        # forward pass
        reconstructed_action, z_mu, z_var = model(action, state)
        
        # mse loss
        loss = F.mse_loss(reconstructed_action, action)
        test_loss+=loss.item()
        
# Display test loss
test_loss/=np.mean(test_dataset.shape[0])
print(f' Test Loss: {test_loss:.5f}')







