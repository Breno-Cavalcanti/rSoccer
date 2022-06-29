import gym
import rsoccer_gym
import time
import ipdb
import numpy as np

# Using VSS Single Agent env
env = gym.make('VSS3v3-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1077, 1377):
    obs_list = list()
    done = False
    obs = env.reset()
    a = time.time()
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, acts, done, _ = env.step(action)
        # env.render()

        obs_list.append(np.concatenate((next_state, np.array(acts)), axis=0))
    obs_arr = np.array(obs_list)
    if i % 100 == 0:
        print(i)
        print(time.time() - a)

    # save the array as txt
    np.savetxt('./data/positions-' +
               str(i) + '.txt', obs_arr, delimiter=',')
    # np.save('./data/vss_data/positions-' + str(i), obs_arr)
    # ipdb.set_trace()
