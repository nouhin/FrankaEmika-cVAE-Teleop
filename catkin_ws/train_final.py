import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 1000)
        self.e2 = nn.Linear(1000, 500)

        self.mean = nn.Linear(500, latent_dim)
        self.log_std = nn.Linear(500, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 500)
        self.d2 = nn.Linear(500, 1000)
        self.d3 = nn.Linear(1000, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], dim=1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)

        log_std = self.log_std(z)  #.clamp(-4, 15)

        std = torch.exp(log_std)

        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z, clip=1):

        if clip is not None:
            z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat((state, z), dim=1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        return self.max_action * torch.tanh(a)


class VAEModule(object):
    def __init__(self, *args, **kwargs):
        self.vae = VAE(*args, **kwargs).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        self.min_loss = 1000.

    def train(self, my_dataset, test_dataset, batch_size=100, epochs=100):  #folder_name,
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': []}

        for i in range(epochs):
            my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            epc_loss, vae_loss, recon_loss, KL_loss = self.train_step(my_dataloader, batch_size)
            logs['vae_loss'].append(vae_loss)
            logs['recon_loss'].append(recon_loss)
            logs['kl_loss'].append(KL_loss)

            test_loss = self.test(test_dataloader)
            if test_loss < self.min_loss:
                self.min_loss = test_loss
                self.save('./cvae_model/1_sin_cvae.pth')
                print('[INFO] Model saved loss : ' + '{:.4}'.format(epc_loss), "\n")
                #self.save('./models/'+str(i+1)+'_cvae_{:.4}.pth'.format(test_loss))
            print('[INFO] Epoch ' + str(i + 1) + ' loss : ' + '{:.4}'.format(epc_loss))
            print('       Testing loss : {:.4}'.format(test_loss), "\n")

        return logs

    def train_step(self, dataset, batch_size=100):
        phase = 'train'
        running_loss = 0.0
        for batch_idx, (state, action) in enumerate(dataset):
            self.vae.train()

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)

            self.vae_optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                recon, mean, std = self.vae(state, action)
                recon_loss = F.mse_loss(recon, action)

                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + 0.01 * KL_loss

                vae_loss.backward()

                self.vae_optimizer.step()
            running_loss += vae_loss.item() * action.size(0)

        epoch_loss = running_loss / len(dataset.dataset)

        return epoch_loss, vae_loss.cpu().data.numpy(), recon_loss.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def test(self, dataset, batch_size=100, epochs=1):
        phase = 'test'
        running_loss = 0.0
        for batch_idx, (state, action) in enumerate(dataset):
            self.vae.eval()

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)

            self.vae_optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                recon, mean, std = self.vae(state, action)
                recon_loss = F.mse_loss(recon, action)

            #print("[INFO] action :",action[0],"recon :" ,recon[0])
            #print()

            running_loss += recon_loss.item() * action.size(0)

        epoch_loss = running_loss / len(dataset.dataset)

        return epoch_loss

    #def predict_action(self, z, state):

    def save(self, path):
        torch.save(self.vae.state_dict(), path)

    def load(self, path):
        self.vae.load_state_dict(torch.load(path, map_location=device))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 1
vae_trainer = VAEModule(state_dim=7, action_dim=7, latent_dim=latent_dim, max_action=1.)
#vae_trainer = VAEModule(8, 8, 2, 1, vae_lr=1e-4)


def npy_loader(path):
    return torch.from_numpy(np.load(path))


#train_dataset=npy_loader("all_state_action_joints.npy")
#train_dataset = train_dataset[0:300000]

state = np.load("all_state_joints.npy")
#max_array = np.max(abs(state))
state = state / 10.

action = np.load("all_action_joints.npy")
#max_array = np.max(abs(action))
action = action / 10.

#state = state[0:50000]
#action = action[0:50000]

state_test = state[0:int(0.2 * len(state))]
action_test = action[0:int(0.2 * len(state))]

state_train = state[int(0.2 * len(state)) + 1:len(state)]
action_train = action[int(0.2 * len(state)) + 1:len(state)]

tensor_x = torch.Tensor(state_train)
tensor_y = torch.Tensor(action_train)

tensor_x_test = torch.Tensor(state_test)
tensor_y_test = torch.Tensor(action_test)

my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

# Train
logs = vae_trainer.train(my_dataset, test_dataset, batch_size=256, epochs=100)
