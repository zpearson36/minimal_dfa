import custom_env
import gymnasium as gym
from stable_baselines3 import PPO

env = custom_env.CustomEnv(gym.make("CartPole-v1"))
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10_000)

#vec_env = model.get_env()
vec_env = custom_env.CustomEnv(gym.make("CartPole-v1",  render_mode='human'))
obs, _ = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print()
    print(action)
    print()
    obs, reward, done, trunc, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
