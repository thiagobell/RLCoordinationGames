##game rewards
#row player at first first level of lists
game = [[10,0],[0,10]]
from ucb1discounted import UCB1Discounted
import random


results = [0 for x in range(100)]
for trial in range(100):
    ucb = UCB1Discounted(2,2,0.65)
    iteres = []
    for ite in range(100):
        act1 = ucb.choseActionDriver(1)
        act0 = ucb.choseActionDriver(0)
        if trial == 0:
            print "%d %d" % (act0, act1)
        reward = game[act0][act1]
        ucb.set_reward(0,act0,reward)
        ucb.set_reward(1,act1,reward)
        #print reward
        if reward == 10:
            results[ite] +=1
        iteres.append(reward)
    #print iteres
print results
