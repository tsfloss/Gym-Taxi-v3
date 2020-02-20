import gym
import numpy as np
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, env, lr, gamma, epsilon):
        self.env = env
        self.counter = 0
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.uniform(low = 0.0, high=1.0, size=([500] + [env.action_space.n]))
        
    def train(self,train_episodes):
        for episode in range(1,train_episodes+1):
            current_state = env.reset()
            steps = 0
            penalties = 0
            done = False
            while not done:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
        #            print("Random: {}".format(action))
                else:
                    action = np.argmax(self.q_table[current_state])
        #            print("Q-action: {}".format(action))
                    
                new_state, reward, done, _ = env.step(action)
                
                current_q = np.max(self.q_table[current_state])
                max_future_q = np.max(self.q_table[new_state])
                new_q = (1-self.learning_rate)*current_q + learning_rate * (reward + self.gamma * max_future_q)
                self.q_table[current_state, action] = new_q
                current_state = new_state
                steps += 1
                if reward == -10:
                    penalties += 1
            self.epsilon *= 0.995
            print("Episode {}: we made it in {} steps with {} penalties".format(episode, steps, penalties))
        print("Done training {} episodes!".format(train_episodes))
    
    def test(self, test_episodes):
        all_penalties = []
        all_steps = []
        for episode in range(1,test_episodes+1):
            current_state = self.env.reset()
            done = False
            steps = 0
            penalties = 0
            while not done:
                action = np.argmax(self.q_table[current_state])         
                new_state, reward, done, _ = self.env.step(action)
                current_state = new_state
                steps += 1
                if reward == -10:
                    penalties += 1
            all_penalties.append(penalties)
            all_steps.append(steps)
        print("Average amount of steps: {}".format(np.mean(all_steps)))
        print("Average amount of penalties: {}".format(np.mean(all_penalties)))
    
env = gym.make("Taxi-v3").env

learning_rate = 0.1
gamma = 0.6
epsilon = 0.1
agent = QAgent(env,learning_rate, gamma, epsilon)



    