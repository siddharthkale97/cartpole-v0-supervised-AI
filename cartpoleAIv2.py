import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import rmsprop
LR = 1e-3

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
score_requirement = 50
initial_games = 10000


def initial_population():

    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


# initial_population()


def neural_network_model(input_size):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_size, ), activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop(lr=0.001, decay=1e-6), metrics=['accuracy'])

    return model


def train_model(training_data, model=False):
    X = [i[0] for i in training_data]
    X = np.array(X)
    Y = [i[1] for i in training_data]
    print(X.shape)
    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit(np.array(X), np.array(Y), batch_size=None, epochs=3, shuffle=True)

    return model


training_data = initial_population()
model = train_model(training_data)

# model.save('cartpole_model.model')

##############
scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(np.array(prev_obs.reshape(-1, len(prev_obs))))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        prev_obs = np.array(prev_obs)
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)


print('Average Score ', mean(scores))
print('Choice 1 {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))




