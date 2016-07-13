#!/usr/bin/python
from __future__ import print_function
import logging as log
import argparse
import time
import king_pong as env
import multicnet as percept
import random
import numpy as np
from collections import deque
import os


class DeepLearningAgent:
    """
    Agent class that uses a deep network
    and a game and learns how to play it
    """

    def __init__(self, input_width, input_height, nactions, nimages, reset=False, train=False):
        """
        Sets up the variables, initializes the game
        and the convolutional network
        recreates the files if needed
        """
        self.train = train
        self.memory = deque()
        self.environment = env.GameState()
        self.environment.print_scores = not train
        self.step = 0
        self.perception = percept.MultilayerConvolutionalNetwork(
            input_width, input_height, nimages, nactions)

        self.logs_directory = 'logs'
        self.percepts_directory = 'percepts'
        self.networks_directory = 'networks'

        if reset:
            log.info('cleaning up directories')
            import shutil
            shutil.rmtree(self.logs_directory)
            shutil.rmtree(self.percepts_directory)
            shutil.rmtree(self.networks_directory)

        if not os.path.exists(self.logs_directory): os.makedirs(self.logs_directory)
        if not os.path.exists(self.percepts_directory): os.makedirs(self.percepts_directory)
        if not os.path.exists(self.networks_directory): os.makedirs(self.networks_directory)

        self.memory_max_len = 50000
        self.save_interval = 100000
        self.nactions = nactions
        self.gamma = 0.99
        self.observe = int(self.memory_max_len * 1.25)
        self.initial_epsilon = 0.60 if self.train else 0.00
        self.final_epsilon = 0.01 if self.train else 0.00
        self.epsilon = self.initial_epsilon
        self.explore = self.observe * 10 # step when epsilon reaches final epsilon value
        self.batch_size = 60

        log.debug(vars(self))

    def save_progress(self, stack):
        """
        Save the current progress of the agent,
        that is the readout values, the hidden layer values,
        the images in the current stack of the agent
        and the current neural network
        """
        log.info('saving current stack')
        for i in range(stack.shape[2]):
            current_percept_path = self.percepts_directory + '/frame' + \
                                   str(self.step) + '-' + str(i) +'.png'
            self.perception.save_percepts(current_percept_path, stack[:,:,i])

        if not self.train:
            log.debug('no need to save network - we are not training')
            return

        log.info('saving percepts')
        with open(self.logs_directory + '/readout.txt', 'w+') as a_file:
            with open(self.logs_directory + '/hidden.txt', 'w+') as h_file:
                self.perception.save_variables(a_file, h_file, stack)

        log.info('saving current network')
        self.perception.save_network(self.networks_directory, self.step)

    def load_progress(self):
        """
        Loads the progress from the agent deep network
        """
        file_loaded = self.perception.attempt_restore(self.networks_directory)
        if file_loaded:
            log.info('loaded successfully => ' + str(file_loaded))
        else:
            log.info("didn't find any saved network")

    def select_action(self, x_t = None):
        """
        Selects either an epsilon random action
        or just the best action according to the
        current state of the neural network
        """
        a_t = np.zeros([self.nactions])
        do_action = 0
        if x_t is None or random.random() <= self.epsilon:
            log.debug('random action selected with epsilon ' + str(self.epsilon))
            do_action = random.randrange(self.nactions)
        else:
            log.debug('greedy action selected with epsilon ' + str(self.epsilon))
            do_action = self.perception.select_best_action(x_t)

        if self.epsilon > self.final_epsilon and self.step > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        log.debug('epsilon was updated to ' + str(self.epsilon))

        a_t[do_action] = 1
        log.debug('action distribution looks like ' + str(a_t))
        return a_t

    def remember(self, sars):
        """
        Inserts a state action reward new_state
        observation into the memory bank
        and pops the oldest memory if limit was
        a limit reached
        """

        self.memory.append(sars)
        log.debug('new sars observation inserted')
        log.debug('before memory size ' + str(len(self.memory)))

        if len(self.memory) > self.memory_max_len:
            log.debug('memory reached the max size. Removing oldest memory')
            self.memory.popleft()

        log.debug('after memory size ' + str(len(self.memory)))

    def learn_maybe(self):
        """
        This is the main training loop.
        The agent would leave early if the train
        attribute is not set to True and if the
        observe period hasn't been completed.

        It basically grabs a self.batch_size sample from the
        memory bank, extrans the sars observations and it
        batch trains the deep neural network
        """

        if not self.train or self.step <= self.observe:
            log.debug('No training the network. Train is set to ' + str(self.train) +
                      '. Current step is ' + str(self.step) +
                      ' and observation period will end after ' + str(self.observe))
            return

        minibatch = random.sample(self.memory, self.batch_size)
        log.debug('minibatch created. Total sample of ' + str(len(minibatch)))

        state_batch = [m[0] for m in minibatch]
        action_batch = [m[1] for m in minibatch]
        rewards_batch = [m[2] for m in minibatch]
        state_prime_batch = [m[3] for m in minibatch]
        log.debug('sars have been separated and extracted')

        value_batch = []
        readout_batch = self.perception.readout_act(state_prime_batch)
        log.debug('values of the next state have been queried')

        for i in range(len(minibatch)):
            if abs(rewards_batch[i]) == 1.0:
                value_batch.append(rewards_batch[i])
                log.debug('current memory was calculated as a terminal observation')
            else:
                value_batch.append(rewards_batch[i] + self.gamma * np.max(readout_batch[i]))
                log.debug('current memory was calculated as a non-terminal observation')
            log.debug('calculated value of ' + str(value_batch[i]))

        log.debug('training network with ' + str(len(value_batch)) + ' samples')
        self.perception.train(value_batch, action_batch, state_batch)
        log.debug('end of training')

    def act_and_perceive(self, action_selected, percept_stack):
        """
        Acts in the environment. That is, it moves the game
        one step further by passing the selected action,
        it then reads what the environment had to say about that
        and finally preprocesses the image into the format and
        size used by the network and appends the new image into the front
        of the image stack
        """
        log.debug('acting with ' + str(action_selected))
        new_percept, reward = self.environment.frame_step(action_selected)
        log.debug('got reward of ' + str(reward))

        new_percept = self.perception.preprocess_percepts(new_percept)
        log.debug('got the new image')

        new_percept_stack = np.append(new_percept, percept_stack[:, :, :3], axis=2)
        log.debug('modified the stack to include the new image')
        return new_percept_stack, reward

    def exist(self, percept_stack):
        """
        Main agent loop that selects and action, acts, remembers
        what happen, learns, updates, saves the progress and
        decides if it should die
        """
        log.debug('entering main agent loop')
        while True:

            start = time.time()
            action_selected = self.select_action(percept_stack)
            new_percept_stack, reward = self.act_and_perceive(action_selected, percept_stack)
            log.debug("act and percieve took " + str(time.time() - start))

            start = time.time()
            self.remember((percept_stack, action_selected, reward, new_percept_stack))
            log.debug("remembering took " + str(time.time() - start))

            start = time.time()
            self.learn_maybe()
            log.debug("learning took " + str(time.time() - start))

            log.debug('updating the image stack')
            percept_stack = new_percept_stack
            self.step += 1

            start = time.time()
            if self.step % self.save_interval == 0:
                self.save_progress(percept_stack)
            log.debug("save progress took " + str(time.time() - start))

            timestep_info = "timestep", self.step, "epsilon", self.epsilon, \
                            "action_selected", action_selected, "reward", reward
            if reward != 0:
                log.info(timestep_info)
            else:
                log.debug(timestep_info)

            if self.environment.score_last_changed() and not self.train:
                best_of, matches_per_game = self.environment.first_to
                cpu_games, agent_games = self.environment.games
                cpu_score, agent_score = self.environment.score
                current_game = cpu_games + agent_games
                title = ' SCOREBOARD '

                log.info('=' * 20 + title + '=' * 20)
                log.info('best of ' + str(best_of) + ' games')
                log.info('each game goes to ' +
                         str(matches_per_game) +
                         ' points')
                log.info('-' * (40 + len(title)))
                log.info('game # ' + str(current_game) + ' score:')
                log.info('-' * (40 + len(title)))
                log.info('cpu ' + str(cpu_score))
                log.info('agent ' + str(agent_score))
                log.info('-' * (40 + len(title)))
                log.info('overall score:')
                log.info('-' * (40 + len(title)))
                log.info('cpu ' + str(cpu_games))
                log.info('agent ' + str(agent_games))
                log.info('=' * (40 + len(title)))

            if not self.train and self.environment.game_over():
                log.info('killing the agent. Train was ' +
                         str(self.train) +
                         ' and game over was ' +
                         str(self.environment.game_over()))
                cpu_games, agent_games = self.environment.games
                won = agent_games > cpu_games
                log.info(('agent won ' if won else 'cpu won ') +
                         '... final score ' + str(cpu_games) +
                         ' to ' + str(agent_games))
                break


def main(args):
    """
    Sets up the environment, loads the agent, prepares the
    first action and moves the game a single frame
    and then enters the agents main loop
    """
    log.info('Verbose output enabled ' + str(log.getLogger().getEffectiveLevel()))
    log.debug(args)

    npixels, nactions, nimages = 80, 3, 4
    agent = DeepLearningAgent(npixels, npixels, nactions, nimages, args.reset, args.train)
    log.info('agent loaded successfully')

    agent.environment.first_to = [args.ngames, args.nmatches]
    log.info('we are playing ' + str(args.ngames) +
             ' games of ' + str(args.nmatches) +
             ' matches each - Good luck!!!')

    noop = np.zeros(nactions)
    noop[0] = 1 # 0 action is do nothing, 1 is up, 2 is down
    raw_percept, _ = agent.environment.frame_step(noop)
    log.info('pumping first frame on the game state')

    x_t = agent.perception.preprocess_percepts(raw_percept, False)
    first_percept_stack = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    log.info('first stack created successfully')

    agent.load_progress()
    log.info('agent load progress completed')

    agent.exist(first_percept_stack)
    log.info('agent dies')


if __name__ == '__main__':
    """
    Loads the script and parses the arguments
    """
    from sys import argv
    parser = argparse.ArgumentParser(
        description='A Deep Reinforcement Learning agent that plays pong like a King.'
    )
    parser.add_argument(
        '-v',
        help='logging level set to ERROR',
        action='store_const', dest='loglevel', const=log.ERROR,
    )
    parser.add_argument(
        '-vv',
        help='logging level set to INFO',
        action='store_const', dest='loglevel', const=log.INFO,
    )
    parser.add_argument(
        '-vvv',
        help='logging level set to DEBUG',
        action='store_const', dest='loglevel', const=log.DEBUG,
    )
    parser.add_argument(
        '-g', '--games',
        help='number of games for the agent to play. (default: 100) '
             'NOTE: if you enable training, this variable will not be used',
        dest='ngames', type=int, default=100,
    )
    # TODO: Enable players to be set left or right
    """
    parser.add_argument(
        '-l', '--left-player',
        help='agent to play on left. (default: CPU) '
             'OPTIONS: CPU, AGENT, HUMAN',
        dest='lplayer', type=str, default='CPU',
    )
    parser.add_argument(
        '-r', '--right-player',
        help='agent to play on left. (default: AGENT) '
             'OPTIONS: CPU, AGENT, HUMAN',
        dest='rplayer', type=str, default='AGENT',
    )
    """
    parser.add_argument(
        '-m', '--matches',
        help='number of matches for each game. (default: 5) '
             'NOTE: if you enable training, this variable will not be used',
        dest='nmatches', type=int, default=5,
    )
    parser.add_argument(
        '-t', '--train',
        help='allows the training of the deep neural network. '
             'NOTE: leave disabled to only see the current agent behave',
        dest='train', action='store_true')
    parser.add_argument(
        '-c', '--clear',
        help='clears the folders where the state of the agent is saves. '
             'NOTE: use this to retrain the agent from scratch',
        dest='reset', action='store_true')

    args = parser.parse_args()
    if args.loglevel:
        log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=args.loglevel)
    else:
        log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=log.CRITICAL)

    main(args)
