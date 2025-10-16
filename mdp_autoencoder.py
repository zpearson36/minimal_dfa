import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchrl.data import ReplayBuffer, ListStorage

device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")

LATENT_STATE_SPACE_SIZE = 40

class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Linear(4, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, LATENT_STATE_SPACE_SIZE),
                nn.Softmax(),
                )

        self.decoder = nn.Sequential(
                nn.Linear(41, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                nn.Linear(150, 150),
                nn.ReLU(),
                )

        self.next_state = nn.Sequential(
                nn.Linear(150, LATENT_STATE_SPACE_SIZE),
                nn.Softmax()
                )

        self.reward = nn.Sequential(
                nn.Linear(150, 1),
                nn.Sigmoid()
                )

        self.terminal = nn.Sequential(
                nn.Linear(150, 1),
                nn.Sigmoid()
                )

        self.loss = nn.KLDivLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def encode(self, state):
        state = self.encoder(state)
        return state

    def reparam(self, state):
        z = nn.functional.gumbel_softmax(state, hard=True)
        return z

    def decode(self, state, action):
        z = self.decoder(torch.cat((state, torch.Tensor([action]))))
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

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters())

    def forward(self, x):
        return self.layers(x)

def train(vae, disc, replay, max_eps, action_space):
    for episode in range(max_eps):
        choice = np.random.randint(0, high=len(replay))
        state, action, reward, next_state, term = replay[choice]

        # Encode state and next_state
        encoded_state = vae.encode(torch.Tensor(state))
        encoded_next_state = vae.encode(torch.Tensor(next_state))

        # Generate predicted next_state
        pred_next_state, pred_reward, pred_term = vae.decode(
            encoded_state, action)

        # Get Simulated Step
        random_action =np.random.choice(action_space)
        sim_next_state, sim_reward, sim_term = vae.decode(
            pred_next_state,random_action)

        real = (encoded_state, action, reward, encoded_next_state, term)
        fake = (encoded_state, action, pred_reward, pred_next_state, pred_term)
        sim = (pred_next_state, random_action, sim_reward, sim_next_state, sim_term)

        # update Discriminator
        real_guess = disc.forward(
                torch.cat((
                    real[0],
                    real[3],
                    torch.Tensor([real[2]]),
                    torch.Tensor([real[1]]),
                    torch.Tensor([real[4]])))
                )
        fake_guess = disc.forward(
                torch.cat((
                    fake[0],
                    fake[3],
                    torch.Tensor([fake[2]]),
                    torch.Tensor([fake[1]]),
                    torch.Tensor([fake[4]])))
                )
        sim_guess = disc.forward(
                torch.cat((
                    sim[0],
                    sim[3],
                    torch.Tensor([sim[2]]),
                    torch.Tensor([sim[1]]),
                    torch.Tensor([sim[4]])))
                )
        disc.zero_grad()
        lm = 1 # hyperparam to normalize loss
        real_loss = disc.loss(torch.Tensor([real_guess]), torch.Tensor([1]))
        fake_loss = disc.loss(torch.Tensor([fake_guess]), torch.Tensor([0]))
        sim_loss   = disc.loss(torch.Tensor([sim_guess]), torch.Tensor([1]))
        disc_loss = np.log(real_loss) + lm*(np.log(1-fake_loss) + np.log(1-sim_loss))
        disc_loss.requires_grad=True
        disc_loss.backward()
        disc.optimizer.step()

        # update VAE
        vae.zero_grad()
        vae_loss = vae.loss(
                torch.cat((
                    torch.Tensor(pred_next_state),
                    torch.Tensor([pred_reward]),
                    torch.Tensor([pred_term])))[0],
                torch.cat((
                    torch.Tensor(encoded_next_state),
                    torch.Tensor([reward]),
                    torch.Tensor([term])))[0]
                )
        vae_loss = vae_loss - disc_loss
        vae_loss.backward()
        vae.optimizer.step()
        if episode % 100 == 0: print(episode, disc_loss, vae_loss)

def fill_initial_replay(env):
    rb = []
    state, _ = env.reset()
    while len(rb) < 10000:
        action = np.random.choice([0,1])
        n_state, reward, term, trunc, _ = env.step(action)
        rb.append([state.tolist(), action, reward, n_state.tolist(), term])
        state = n_state
        if term:
            state, _ = env.reset()

    return rb

if __name__ == "__main__":
    # populate replay_buffer
    env = gym.make("CartPole-v1")
    rb = fill_initial_replay(env)

    # train latent space generator
    vae = VAE()
    disc = Discriminator()
    train(vae, disc, rb, 10000, [0,1])

    # train policy
