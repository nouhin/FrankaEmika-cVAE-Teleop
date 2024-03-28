import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    ''' This is the encoder part of CVAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim:  An integer indicating the size of input part to reconstruct
            hidden_dim: A list indicating the size of hidden layers.
            latent_dim: An integer indicating the latent size.
        '''
        super().__init__()

        self.e1 = nn.Linear(input_dim + 8, hidden_dim[0])
        self.e2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.mu = nn.Linear(hidden_dim[1], latent_dim)
        self.var = nn.Linear(hidden_dim[1], latent_dim)

    def forward(self, x):

        x = F.relu(self.e1(x.float()))

        x = F.relu(self.e2(x))

        # Learned distribution for latent space
        mean = self.mu(x)

        log_var = self.var(x)

        return mean, log_var


class Decoder(nn.Module):
    ''' This is the decoder part of CVAE
    '''
    def __init__(self, latent_dim, hidden_dim, output_dim):
        '''
        Args:
            latent_dim: An integer indicating the latent size.
            hidden_dim: A list indicating the size of hidden layers.
            output_dim: An integer indicating the size of output
        '''
        super().__init__()

        self.d1 = nn.Linear(latent_dim + 8, hidden_dim[1])
        self.d2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.hidden_to_out = nn.Linear(hidden_dim[0], output_dim)

    def forward(self, x):

        x = F.relu(self.d1(x))

        x = F.relu(self.d2(x))

        # tanh activation function allows reconstruction of actions of negative values
        generated_x = torch.tanh(self.hidden_to_out(x))

        return generated_x


class CVAE(nn.Module):
    ''' This is the CVAE, composed of an encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: An integer indicating the size of input part to reconstruct
            hidden_dim: A list indicating the size of hidden layers.
            latent_dim: An integer indicating the latent size.  
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, action, state):

        # Concatenate state and action
        state_action = torch.cat((state, action), dim=1)

        # Learn distribution for latent space
        z_mu, z_var = self.encoder(state_action)

        # Sample from the distribution having latent parameters z_mu, z_var ,
        # then reparameterize (sample*std+mean)
        std = torch.exp(z_var)
        z_sample = z_mu + std * torch.randn_like(std)

        # Limiting latent action range
        z_sample = z_sample.clamp(-1, 1)

        # Concatenate state and latent vector
        z = torch.cat((state, z_sample), dim=1)

        # decode action from latent action and state
        generated_action = self.decoder(z)

        return generated_action, z_mu, z_var
