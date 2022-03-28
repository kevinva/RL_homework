import gym
from envs.gridworld_env import CliffWalkingWapper

env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)
n_states = env.observation_space.n
n_actions = env.action_space.n
print(f'状态数：{n_states}, 动作数：{n_actions}')

state = env.reset()
print(f'初始状态：{state}')