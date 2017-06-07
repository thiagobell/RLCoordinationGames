# -*- coding: utf-8 -*-
'''
This module implements the UCB1Window class implementing the Sliding Window UCB1  algorithm as decribed
on the paper On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems by Garivier and Moulines.
'''

import math
import operator
import random
import sys
class UCB1Window():
    def __init__(self, num_player, num_action, discount_factor, window_size, rewardbound,initorder):
        self.k = num_action
        self.init_order = initorder
        self.numdrivers = num_player
        self.discount_factor = discount_factor
        self.window_size = window_size
        self.rewards = []
        self.rewardUpperBound = rewardbound##upper bound on rewards. needed for algorithm
        self.xi = 2.0
        self.episode = [0] * self.numdrivers
        self.rewards ={} #indexed by driver id. value is list of rewards ordered in chronological order
        self.actions = {} #same as above for actions taken

        for dinx in range(self.numdrivers):
            self.rewards[dinx] = []
            self.actions[dinx] = []

    def reset(self):
        self.rewards = []
        self.episode = [0] * self.numdrivers
        self.rewards ={} #indexed by driver id. value is list of rewards ordered in chronological order
        self.actions = {} #same as above for actions taken

        for dinx in range(self.numdrivers):
            self.rewards[dinx] = []
            self.actions[dinx] = []


    def set_reward(self, dInx, action, reward):
      self.rewards[dInx].append(reward)
      self.actions[dInx].append(action)


    ##returns the route id the driver chose
    def chooseActionDriver(self, dInx):
        self.episode[dInx]+=1
        if (len(self.actions[dInx]) < self.k):
            if self.init_order == 1:
                ##plays each arm once in a random order
                possible_actions = []
                for k in range(self.k):
                    if k not in self.actions[dInx]:
                        possible_actions.append(k)
                return random.choice(possible_actions)
            elif self.init_order == 2: #play arm sequentially
                return len(self.actions[dInx])
            else:
                raise "invalid init order"
        else:  # regular case
            log_size = self.xi * math.log(min(self.window_size,self.episode))
            Xs = [0.0]*self.k
            Ns = [0.0]*self.k
            begin_window = self.episode[dInx] - min(self.window_size, self.episode[dInx]-1)
            itercount = 0
            n = 0.0
            ##calculate X and N for every action by iterating over previous rewards
            for i in range(begin_window, self.episode[dInx]):
                discountedFactor = math.pow(self.discount_factor, self.episode[dInx] - i)
                Ns[self.actions[dInx][i-1]] += discountedFactor
                Xs[self.actions[dInx][i-1]] += discountedFactor * self.rewards[dInx][i-1]
                itercount += 1
                n += Ns[self.actions[dInx][i-1]]
            #n = sum(Ns)

            Cs = [0.0]*self.k
            choice_value = []
            for kinx in range(self.k):
                if Ns[kinx] > 0:
                    Xs[kinx] = Xs[kinx] / Ns[kinx]
                    c = self.rewardUpperBound
                    c *= math.sqrt(log_size/Ns[kinx])
                    Cs[kinx] = c
                choice_value.append(Cs[kinx]+Xs[kinx])

            ##chooses action with highest value
            index, v = max(enumerate(choice_value), key=operator.itemgetter(1))

            return index

