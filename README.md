# Gym-Taxi-v3
My solution to the Gym environment Taxi-v3 using the Q learning algorithm.

The code is written to be executed in an IPython console.

## Training
Once the code is executed the model can be trained for a number of training_episodes by:
```
agent.train(train_episodes)
```
The model trains until all episodes have passed.

## Testing
Once the model is trained, It can be tested for a number of test_episodes by:
```
agent.test(test_episodes)
```

## Technical Information
The environment is solved using a Q Learning implementation. The model performs random actions decreasingly often as a means of exploration.

After 100000 episodes the model is certainly done training and subsequent test results are:

```
Average amount of steps: 13.07659
Average amount of penalties: 0.0
```
