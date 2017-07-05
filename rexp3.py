# -*- coding: utf-8 -*-
'''
This module implements the Rexp3 class implementing the Rexp3  algorithm as decribed
on the paper Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards by Gur, Yonatan
;Zeevi, Assaf;Besbes, Omar
'''

import math
import operator
import sys
import numpy.random

class Rexp3():
    def __init__(self, num_players, num_actions, epoch_size, gamma):
        self.k = num_actions
        self.numdrivers=num_actions
        self.episode = 0
        self.epoch_size = epoch_size
        self.current_batch = 1
        assert(type(gamma)==float)
        assert(gamma >= 0.0 and gamma <= 1.0)
        self.gamma = gamma
        self.w = {}
        self.p = {}
        for i in range(self.numdrivers):
            self.w[i] = [1.0]*self.k
            self.p[i] = [0.0]*self.k

    def reset(self):
        self.current_batch = 1
        self.episode = 0
        self.w = {}
        self.p = {}
        for i in range(self.numdrivers):
            self.w[i] = [1.0] * self.k
            self.p[i] = [0.0] * self.k

    ##returns the route id the driver choosed
    def chooseActionDriver(self, dInx):
        wsum = sum(self.w[dInx])

        for kinx in range(self.k):
            self.p[dInx][kinx]= (1 - self.gamma) * (self.w[dInx][kinx]/wsum) + self.gamma/self.k
        return int(numpy.random.choice(range(self.k), 1, p=self.p[dInx])[0])


    def set_reward(self, dInx, action, reward):
        for kinx in range(self.k):
            if(action == kinx):
                x = reward/self.p[dInx][action]
            else:
                x = 0.0
            self.w[dInx][kinx] *= math.exp((self.gamma*x)/self.k)
