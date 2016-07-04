import time
import king_pong as env
import multicnet as percept
import random
import numpy as np
from collections import deque
import sys
import os

class DeepLearningAgent:
    def __init__(self, input_width, input_height, nactions):
        self.memory = deque()
        self.environment = env.GameState()
        self.iteration = 0
        self.perception = percept.MultilayerConvolutionalNetwork(
            input_width, input_height, 4, nactions)

        self.logs_directory = 'logs'
        self.percepts_directory = 'percepts'
        self.networks_directory = 'networks'

        if not os.path.exists(self.logs_directory): os.makedirs(self.logs_directory)
        if not os.path.exists(self.percepts_directory): os.makedirs(self.percepts_directory)
        if not os.path.exists(self.networks_directory): os.makedirs(self.networks_directory)

        self.save_every = 100000
        self.nactions = nactions
        self.gamma = 0.90
        self.observe = 100000.
        self.explore = self.observe*10
        self.final_epsilon = 0.01
        self.initial_epsilon = .40
        self.epsilon = self.initial_epsilon
        self.replay_memory = 50000
        self.num_dreams = 20
        self.dream_len = 50
        self.dream_every = self.replay_memory / 5


    def save_progress(self, stack):
        with open(self.logs_directory + '/readout.txt', 'w+') as a_file:
            with open(self.logs_directory + '/hidden.txt', 'w+') as h_file:
                self.perception.save_variables(a_file, h_file, stack)

        for i in range(stack.shape[2]):
            current_percept_path = self.percepts_directory + '/frame' + \
                                   str(self.iteration) + '-' + str(i) +'.png'
            self.perception.save_percepts(current_percept_path, stack[:,:,i])

        self.perception.save_network(self.networks_directory, self.iteration)

    def load_progress(self):
        file_loaded = self.perception.attempt_restore(self.networks_directory)
        if file_loaded:
            print('Loaded successfully', file_loaded)
        else:
            print("Didn't find any saved network")

    def select_action(self, x_t = None):
        a_t = np.zeros([self.nactions])
        do_action = 0
        if x_t is None or random.random() <= self.epsilon:
            # print('random action -------------')
            do_action = random.randrange(self.nactions)
        else:
            # print('greedy action ------------')
            do_action = self.perception.select_best_action(x_t)

        a_t[do_action] = 1

        if self.epsilon > self.final_epsilon and self.iteration > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        return a_t

    def remember(self, sars):
        self.memory.append(sars)
        if len(self.memory) > self.replay_memory:
            self.memory.popleft()

    def dream_and_learn_maybe(self):
        if self.iteration <= self.observe:
            return

        if self.iteration % self.dream_every != 0:
            return

        rewards = np.array([abs(m[2]) for m in self.memory])
        if rewards.sum() == 0:
            return

        memories = list(self.memory)
        selected_memories = np.random.choice(len(memories),
                                    self.num_dreams,
                                    replace=True,
                                    p=rewards/rewards.sum())

        value_batch = []
        action_batch = []
        state_batch = []
        for i in selected_memories:
            initial_memory = random.randint(np.max([0, i - self.dream_len/2]),
                                            i)
            last_memory = random.randint(i,
                                         np.min([i + self.dream_len/2, len(memories)])) + 1
            last_memory = self.replay_memory if last_memory > self.replay_memory else last_memory
            initial_memory = self.replay_memory - 10 if initial_memory >= self.replay_memory - 5 else initial_memory
            print initial_memory, i, last_memory

            # get the memories subbatch state prime and reward to
            # calculate updated values
            memories_batch = memories[initial_memory:last_memory]
            state_prime_batch = np.array([m[3] for m in memories_batch])
            readout_batch = self.perception.readout_act(state_prime_batch)
            rewards_batch = np.array([m[2] for m in memories_batch])
            for j in range(last_memory - initial_memory):
                value_batch.extend([rewards_batch[j] + self.gamma * np.max(readout_batch[j])])

            state_batch.extend(np.array([m[0] for m in memories_batch]))
            action_batch.extend(np.array([m[1] for m in memories_batch]))


        print 'Preparing lessons'
        print len(value_batch), len(action_batch), len(state_batch)
        lesson_dict = {
            self.perception.y : value_batch,
            self.perception.a : action_batch,
            self.perception.input_image : state_batch
        }

        self.perception.train(lesson_dict)

    def act(self, action_selected, percept_stack):
        new_percept, reward = self.environment.frame_step(action_selected)
        new_percept = self.perception.preprocess_percepts(new_percept)
        new_percept_stack = np.append(new_percept, percept_stack[:, :, :3], axis = 2)
        return new_percept_stack, reward

    def exist(self, percept_stack, can_die = False):

        while True:

            start = time.time()
            action_selected = self.select_action(percept_stack)
            new_percept_stack, reward = self.act(action_selected, percept_stack)
            # print("Act and percieve",time.time() - start)

            start = time.time()
            self.remember((percept_stack, action_selected, reward, new_percept_stack))
            # print("Remembering",time.time() - start)
            start = time.time()
            self.dream_and_learn_maybe()
            # print("Learning",time.time() - start)

            percept_stack = new_percept_stack
            self.iteration += 1

            start = time.time()
            if self.iteration % self.save_every == 0:
                self.save_progress(percept_stack)
            # print("Save progress",time.time() - start)

            if reward != 0:
                print("timestep", self.iteration, "epsilon", self.epsilon, \
                      "action_selected", action_selected, "reward", reward)

            if can_die and self.environment.game_over():
                break


def main(argv):
    npixels = 80
    nactions = 3
    agent = DeepLearningAgent(npixels, npixels, nactions)
    noop = np.zeros(nactions)
    noop[0] = 1 # 0 action is do nothing, 1 is up, 2 is down
    raw_percept, _ = agent.environment.frame_step(noop)
    x_t = agent.perception.preprocess_percepts(raw_percept, False)
    first_percept_stack = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    agent.load_progress()
    agent.exist(first_percept_stack)

if __name__ == '__main__':
    from sys import argv
    main(argv)
