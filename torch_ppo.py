import tensordict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
import torch
import torchrl
from torchrl.envs import Compose, DoubleToFloat, TransformedEnv, ObservationNorm, StepCounter
from torchrl.envs.libs.gym import GymEnv

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000

total_frames = 50_000

sub_batch_size = 64
num_epochs = 10

clip_epsilon = (0.2)
gamma = 0.99
lmbda = 0.95
enropy_eps = 1e-4

base_env = GymEnv("CartPole-v0", device=device)

env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
            )
        )

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

rollout = env.rollout(3)
print(rollout)
print(rollout.batch_size)

actor_net = torch.nn.Sequential(
        torch.nn.LazyLinear(num_cells, device=device),
        torch.Tanh(),
        torch.nn.LazyLinear(num_cells, device=device),
        torch.Tanh(),
        torch.nn.LazyLinear(num_cells, device=device),
        torch.Tanh(),
        torch.nn.LazyLinear(2*env.action_spec.shape[-1], device=device),
        NormalParamExtractor()
        )

policy_module = TensorDictModule(
        actor_net, in_keys["observation"], out_keys=["loc", "scale"]
        )
