# Import the necessary libraries
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
#from torchsummary import summary
from model import CVAE
from torch.utils.data import DataLoader

# Sets device to gpu if available , else cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64  # number of data in each batch
N_EPOCHS = 20  # number of runs on complete data
INPUT_DIM = 8  # size of each input to reconstract
HIDDEN_DIM = [200, 100]  # size of hidden layers
LATENT_DIM = 2  # latent vector dimension
lr = 1e-3  # learning rate

# To save the trained model with as name the training configuration
experiment_name = "2_hiddenl_LR_" + str(lr) + "_batch_size_" + str(BATCH_SIZE)

ROOT_DIR = os.path.abspath("./")

# The path to the experiment csv file
csv_path = os.path.join(ROOT_DIR, "models", experiment_name)

# The path to checkpoints (at each epoch)
check_path = os.path.join(ROOT_DIR, "models", experiment_name, "checkpoints")

# Create folders if not found
os.makedirs(csv_path, exist_ok=True)
os.makedirs(check_path, exist_ok=True)


# Load data
def npy_loader(path):
    return torch.from_numpy(np.load(path))


train_dataset = npy_loader("train.npy")
val_dataset = npy_loader("val.npy")

# Make batches and convert to tensors (for pytorch)
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Instantiate a model
model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


def calculate_loss(action, reconstructed_action, mean, std):

    # Kullback–Leibler divergence
    KLD = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

    # mse
    recon_loss = F.mse_loss(reconstructed_action, action)
    wheight = 0.001

    return recon_loss + wheight * KLD


######################### Train #####################################


def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, x in enumerate(train_iterator):

        # extract actions from data
        action = x[:, -8:16]
        action = action.float()
        action = action.to(device)

        # extract states from data
        state = x[:, 0:8]
        state = state.float()
        state = state.to(device)

        # update the gradients to zero:
        # necessary because pytorch accumulates the gradients
        optimizer.zero_grad()

        # forward pass
        reconstructed_action, z_mu, z_var = model(action, state)

        # loss
        loss = calculate_loss(action, reconstructed_action, z_mu, z_var)

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss


######################### Validation ################################


def test(e):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating
    # the parameters during evaluation / testing
    with torch.no_grad():
        for i, x in enumerate(val_iterator):

            # extract actions from data
            action = x[:, -8:16]
            action = action.float()
            action = action.to(device)

            # extract states from data
            state = x[:, 0:8]
            state = state.float()
            state = state.to(device)

            # forward pass
            reconstructed_action, z_mu, z_var = model(action, state)

            if e == N_EPOCHS - 1:
                print("x\n", action[0], "reconstructed\n", reconstructed_action[0])

            # Calculate mse loss
            loss = F.mse_loss(reconstructed_action, action)
            test_loss += loss.item()

    return test_loss


# Saves the trained model
def save_checkpoint(model, optimizer, save_path, epoch):

    # checkpoint name
    filename = 'epoch_'+str(epoch) + '_val_loss_' + str(val_loss) +\
                                                '_checkpoint.pth.tar'
    # checkpoint saving path
    save_path = os.path.join(save_path, filename)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path
    )


if __name__ == "__main__":

    best_val_loss = np.inf
    loss_for_display = []

    # Print model information
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(), "\n")

    # train and validation loss at each epoch will be saved in csv file
    csv_path = os.path.join(csv_path, str(experiment_name) + '.csv')
    header = ['Epoch', 'lr', 'train_loss', 'val_loss']
    with open(csv_path, 'wt') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

    # To perform early stopping
    patience_counter = 0

    for e in range(N_EPOCHS):

        # run model for training and evaluation
        train_loss = train()
        val_loss = test(e)

        # loss
        train_loss /= len(train_iterator)
        val_loss /= len(val_iterator)

        loss_for_display.append(val_loss)

        # save epoch information
        with open(csv_path, 'a') as f:
            csv.writer(f).writerow([e, lr, train_loss, val_loss])

        # Save checkpoint (model after epoch e)
        save_checkpoint(model, optimizer, check_path, e)

        # Display train and val loss
        print(f'Epoch {e}, Train Loss: {train_loss:.5f}, Test Loss: {val_loss:.5f}')

        # Early stopping strategy
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 20:
            break

x = np.arange(N_EPOCHS)
plt.plot(x, loss_for_display)
plt.ylim(0, 0.005)
plt.xlabel("Epoch")
plt.ylabel("mse_loss")
plt.grid()
