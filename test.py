import gym
import rsoccer_gym

# Using VSS Single Agent env

# kp, kd = [2.04119102, 0.08515453] # melhor kp e kd para so kp e kd
kp, kd = [1.64670453, 0.6981731]


# env = gym.make('VSSGA-v0', kp=kp, kd=kd, norm_max_speed=1)
env = gym.make('VSSGoTo-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
    print(reward)