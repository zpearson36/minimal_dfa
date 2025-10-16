import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchrl.data import ReplayBuffer, ListStorage

device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")

LATENT_STATE_SPACE_SIZE = 40

class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = None
        self.optimizer = None

        self.encoder = nn.Sequential(
                nn.Linear(5, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, LATENT_STATE_SPACE_SIZE),
                nn.SoftMax(),
                )

        self.decoder = nn.Sequential(
                nn.Linear(2, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                )

        self.next_state = nn.Sequential(
                nn.Linear(150, LATENT_STATE_SPACE_SIZE),
                nn.SoftMax()
                )

        self.reward = nn.Sequential(
                nn.Linear(150, 1),
                nn.Sigmoid()
                )

        self.terminal = nn.Sequential(
                nn.Linear(150, 1),
                nn.Sigmoid()
                )

    def encode(self, state):
        state = self.encoder(state)
        return state

    def reparam(self, state):
        z = nn.functional.gumbel_softmax(state, hard=True)
        return z

    def decode(self, state, action):
        z = self.decoder(torch.cat((z, torch.Tensor([action]))))
        return self.next_state(z), self.reward(z), self.terminal(z)

    def forward(self, state, action):
        state = self.encode(state)
        state_hat = self.reparam(state)
        return self.generate_next_state(state_hat, action)

    def generate_next_state(self, state, action):
        pred_next_state, rew, term = self.decoder(torch.cat((state, torch.Tensor([action]))))
        return pred_next_state, rew, term

class Discriminator(nn.Module):

    def __init__(self):
        # Input = (Current_state, next_state, action, reward, term)
        # Input size = (40, 40, 1, 1, 1) -> 83
        super().__init__()

        self.loss = None
        self.optimizer = None

        self.layers = nn.Sequential(
                nn.Linear(2 * LATENT_STATE_SPACE_SIZE + 3, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 1),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.layers(x)

def train(vae, disc, replay):


if __name__ == "__main__":
    rb = ReplayBuffer(
            storage = ListStorage(max_size=10_000),
            batch_size=128
            )

    vae = VAE()
    disc = Discriminator()

    # populate replay_buffer

    # train latent space generator
    train(vae, disc, rb, epochs)

    # train policy
