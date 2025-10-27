import gymnasium as gym
from gymnasium import spaces
import mdp_autoencoder as mdp
import numpy as np
import torch

class CustomEnv(gym.Env):
    def __init__(self, env):
        super(CustomEnv, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = spaces.Box(low=np.zeros(shape=(mdp.LATENT_STATE_SPACE_SIZE)), high=np.ones(shape=(mdp.LATENT_STATE_SPACE_SIZE)))
        self.vae = mdp.VAE()
        self.disc = mdp.Discriminator()
        current_state, _ = self.env.reset()
        self.current_state = self.vae.encode(torch.Tensor(current_state))

    def step(self, action):
        self.current_state, reward, done = self.vae.decode(self.current_state, action)
        return self.current_state.detach().numpy(), reward, done, False, {}

    def reset(self, seed=42):
        current_state, _ = self.env.reset()
        self.current_state = self.vae.encode(torch.Tensor(current_state))
        return self.current_state.detach().numpy(), _
