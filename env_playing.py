import gym
import numpy as np


env = gym.make("CarRacing-v0")

s = env.reset()
print(s.shape)
