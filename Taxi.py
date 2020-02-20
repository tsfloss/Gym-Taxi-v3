import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3").env

learning_rate = 0.1
discount = 0.6
epsilon = 0.1
frames = []
episodes = 100001

q_table = np.random.uniform(low = 0.0, high=1.0, size=([500] + [env.action_space.n]))
for episode in range(episodes):
    current_state = env.reset()
    steps = 0
    penalties = 0
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
#            print("Random: {}".format(action))
        else:
            action = np.argmax(q_table[current_state])
#            print("Q-action: {}".format(action))
            
        new_state, reward, done, _ = env.step(action)
        
        current_q = np.max(q_table[current_state])
        max_future_q = np.max(q_table[new_state])
        new_q = (1-learning_rate)*current_q + learning_rate * (reward + discount * max_future_q)
        q_table[current_state, action] = new_q
        current_state = new_state
        steps += 1
        if reward == -10:
            penalties += 1
    if episode % 100 ==0:
        print("Episode {}: we made it in {} steps with {} penalties".format(episode, steps, penalties))
print("Done training {} episodes!".format(episodes))

all_penalties = []
all_steps = []
episodes2 = 100
for episode in range(episodes2):
    current_state = env.reset()
    done = False
    steps = 0
    penalties = 0
    while not done:
        action = np.argmax(q_table[current_state])         
        new_state, reward, done, _ = env.step(action)
        print(reward)
        current_state = new_state
        steps += 1
        if steps >= 500:
            print("Had to reset!")
            current_state = env.reset()
            steps = 0
            penalties = 0
        if reward == -10:
            penalties += 1
    all_penalties.append(penalties)
    all_steps.append(steps)
env.close()
print("Average amount of steps: {}".format(np.mean(all_steps)))
print("Average amount of penalties: {}".format(np.mean(all_penalties)))

plt.subplot(1,2,1)
plt.plot(all_steps, label='steps')
plt.legend()
plt.subplot(1,2,2)
plt.plot(all_penalties, label='penalties')
plt.legend()
plt.show()