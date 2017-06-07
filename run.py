##game rewards
#row player at first first level of lists
from ucb1discounted import UCB1Discounted
from ucb1 import UCB1
from ucb1window import UCB1Window
import random
from game import Game


def play_game(game, rl, episodes, repetitions):

    results = [0.0 for x in range(100)]
    for trial in range(repetitions):
        rl.reset()
        for ite in range(episodes):        
            act1 = rl.chooseActionDriver(1)
            act0 = rl.chooseActionDriver(0)
#            if ite < 3 and (trial == 0 or trial == 1):
#                print "s%d %ds" % (act0, act1)
            reward = game.reward(act0,act1)
            rl.set_reward(0,act0,reward)
            rl.set_reward(1,act1,reward)
            #results[ite] += game.ratio_of_maximum(reward)
            if(reward == game.max_reward):
                results[ite] += 1
    return map(lambda x: x/repetitions,results)

game = Game('game1.txt')
assert(game.num_actions_col == game.num_actions_row)
num_actions = game.num_actions_col   
#init order
randominit = 1
sequentialinit = 2
ucb = UCB1(2,num_actions,1)
ucbdiscount = UCB1Discounted(2,num_actions,0.8, sequentialinit) 
ucbwin = UCB1Window(2, game.num_actions_col, 0.99,50,game.max_reward,2)
print 'ucb1:', play_game(game, ucb, 100, 100)
print 'ucb1discount:', play_game(game, ucbdiscount, 100, 10)
print 'ucb1window:', play_game(game, ucbwin, 100, 100)


#print ["%.2f"%x for x in ]
