"""
Test all envs implemented over small number of steps
"""

import gymnasium as gym
import gym_moving_dot

ENVS = ["MovingDotDiscrete-v0",
        "MovingDotDiscreteNoFrameskip-v0",
        "MovingDotContinuous-v0",
        "MovingDotContinuousNoFrameskip-v0"]

for env_name in ENVS:
    print("=== Test: {} ===".format(env_name))

    env = gym.make(env_name, render_mode="human", random_start=True, step_size=2)

    observation, info = env.reset()

    for i in range(5):
        a = env.action_space.sample()
        o, r, terminated, truncated, info = env.step(a)
        print("Obs shape: {}, Action: {}, Reward: {}, Terminated: {}, Truncated: {}, Info: {}".format(o.shape, a, r, terminated, truncated, info))
    
    env.close()
    del env
