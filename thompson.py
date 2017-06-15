# -*- coding: utf-8 -*-
'''
This module implements the Thompson class implementing the Thompson Sampling algorithm.
'''


import numpy as np
import warnings
import random

class Thompson:
    def __init__(self, num_players, num_actions, optimistic = False):
        self.num_actions = num_actions
        self.num_drivers = num_players
        self.observations = [] # key: agent{ key:arm, value:[reward]}
        self.optimistic = optimistic
        for i in range(self.num_drivers):
            self.observations.append([])
            for j in range(self.num_actions):
                self.observations[i].append([])

        self.episode = [0] * self.num_drivers
        self.parameter_update_interval = 20 ## interval between updates on the parameters for the distributions
        self.sd = [] #std deviation values for each agent's observations of each route
        self.av = [] #avg values for each agent's observations of each route

        for dInx in range(self.num_drivers):
            self.sd.append([0.0]*num_actions)
            self.av.append([0.0]*num_actions)

    def reset(self):
        self.observations = []  # key: agent{ key:arm, value:[reward]}

        for i in range(self.num_drivers):
            self.observations.append([])
            for j in range(self.num_actions):
                self.observations[i].append([])

        self.episode = [0] * self.num_drivers
        self.parameter_update_interval = 20  ## interval between updates on the parameters for the distributions
        self.sd = []  # std deviation values for each agent's observations of each route
        self.av = []  # avg values for each agent's observations of each route

        for dInx in range(self.num_drivers):
            self.sd.append([0.0] * self.num_actions)
            self.av.append([0.0] * self.num_actions)

    def chooseActionDriver(self,dInx):
        warnings.simplefilter("error")

        if self.episode[dInx] < self.num_actions * 2:
            self.episode[dInx] += 1
            return  self.episode[dInx] % self.num_actions


        else:
            epsilon = 0.0001
            #updates parameters every x episodes or at the first episode after initialization
            if (self.episode[dInx] % self.parameter_update_interval == 0) or (self.episode[dInx] == self.num_actions * 2):
                for i in range(self.num_actions):
                    self.sd[dInx][i] = np.std(self.observations[dInx][i],ddof=1) + epsilon
                    self.av[dInx][i] = np.average(self.observations[dInx][i])


            thetas = []
            for i in range(self.num_actions):
                if(self.optimistic):
                    vl = max(self.av[dInx][i], np.random.normal(self.av[dInx][i], self.sd[dInx][i]))
                else:
                    vl = np.random.normal(self.av[dInx][i], self.sd[dInx][i])
                thetas.append(vl)
            self.episode[dInx] += 1
            return int( np.argmax(thetas))


    def set_reward(self, dInx, action, value):
        self.observations[dInx][action].append(value)
