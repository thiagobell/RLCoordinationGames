# -*- coding: utf-8 -*-
'''
This module implements the UCB class implementing the UCB1 algorithm.
'''

import math
import operator

import random
class UCB1():
    def __init__(self, num_players, num_actions,initorder):
        #self.k = k
        #self.numdrivers=len(drivers)
        self.num_players = num_players
        self.num_actions = num_actions
        self.init_order = initorder
        self.round = [0]*self.num_players
        self.actions = {}
     	self.number_plays = []
        self.means = []
        for d in range(self.num_players):
            self.number_plays.append([0]*num_actions)
            self.means.append([0.0]*num_actions)
            self.actions[d] = []

    def reset(self):
        self.round = [0]*self.num_players
        self.actions = {}
     	self.number_plays = []
        self.means = []
        for d in range(self.num_players):
            self.number_plays.append([0]*self.num_actions)
            self.means.append([0.0]*self.num_actions)
            self.actions[d] = []
        

    ##returns the route id the driver choosed
    def chooseActionDriver(self, dInx):
        self.round[dInx] += 1
        #initialization
        if (self.round[dInx] <= self.num_actions):
            if(self.init_order == 2):
                self.number_plays[dInx][self.round[dInx]-1] += 1
                return self.round[dInx]-1
            ##plays each arm once in a random order
            possible_actions = []
            for k in range(self.num_actions):
                if k not in self.actions[dInx]:
                    possible_actions.append(k)
            choice = random.choice(possible_actions)
            self.number_plays[dInx][choice] += 1
            return choice
        else:  # regular case
            for a in range(self.num_actions):
              assert(self.number_plays[dInx][a] > 0)
            choice_value = [0.0] * self.num_actions
            for i, u in enumerate(self.means[dInx]):
                choice_value[i] = u + math.sqrt((2.0 * math.log(self.round[dInx])) / self.number_plays[dInx][i])
            ##chooses action with highest value
            index, v = max(enumerate(choice_value), key=operator.itemgetter(1))
            self.number_plays[dInx][index] += 1  ## does not update mean
            return index

    ##updates the means os the specified agent with the reward of the last round
    def set_reward(self, dInx, action, reward):
        plays = self.number_plays[dInx][action]
        self.actions[dInx].append(action)
        self.means[dInx][action] = (self.means[dInx][action] * (plays - 1) + reward) / plays
