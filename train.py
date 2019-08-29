import gym
from matplotlib import pyplot as plt
import torch

from agents import ReinforceAgent

env = gym.make("CarRacing-v0")
# env.verbose = 0

state = env.reset()

agent = ReinforceAgent(gamma=0.95, entropy_coef=0.01)
done = False

rewards = []
log_action_probs = []

loss = []
entropy = []

for episode in range(100):
    i = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action.data.numpy())
        rewards.append(reward)
        log_action_probs.append(agent.get_log_probs(action))
        state = next_state

    l, e = agent.update(torch.stack(log_action_probs), rewards)
    loss.append(l)
    entropy.append(e)
    state = env.reset()
    rewards = []
    log_action_probs = []
    print(f"episode {episode}, loss = {l}")

plt.subplot(211)
plt.plot(loss)
plt.subplot(212)
plt.plot(entropy)

