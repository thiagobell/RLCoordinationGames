##game rewards
#row player at first first level of lists
from ucb1discounted import UCB1Discounted
from ucb1 import UCB1
from ucb1window import UCB1Window
from thompson import Thompson
from rexp3 import Rexp3
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
#                results[ "s%d %ds" % (act0, act1)
            reward = game.reward(act0,act1)
            rl.set_reward(0,act0,reward)
            rl.set_reward(1,act1,reward)
            #results[ite] += game.ratio_of_maximum(reward)
            if(reward == game.max_reward):
                results[ite] += 1
    return map(lambda x: x/repetitions,results)

game = Game('game2.txt')
assert(game.num_actions_col == game.num_actions_row)
num_actions = game.num_actions_col   
#init order
randominit = 1
sequentialinit = 2
ucb_seq = UCB1(2,num_actions,sequentialinit)
ucb_ran = UCB1(2,num_actions,randominit)

ucbdiscount_seq = UCB1Discounted(2,num_actions,0.8, sequentialinit)
ucbdiscount_ran = UCB1Discounted(2,num_actions,0.8, randominit)
ucbwin_seq = UCB1Window(2, game.num_actions_col, 0.99,50,game.max_reward,sequentialinit)
ucbwin_ran = UCB1Window(2, game.num_actions_col, 0.99,50,game.max_reward, randominit)
thomp_non_opt = Thompson(2, game.num_actions_col,False)
thomp_optimistic= Thompson(2, game.num_actions_col, True)
####### Rexp3(num_players, num_actions, epoch_size, gamma)
rexp3 = Rexp3(2, game.num_actions_col, 101, 0.99)
results =[]

results.append(['ucb1 sequential init:'] + play_game(game, ucb_seq, 100, 100))
results.append(['ucb1 random init:'] + play_game(game, ucb_ran, 100, 100))
results.append(['ucb1discount sequential init:'] + play_game(game, ucbdiscount_seq, 100, 10))
results.append(['ucb1discount random init:'] + play_game(game, ucbdiscount_ran, 100, 10))
results.append(['ucb1window sequential init:'] + play_game(game, ucbwin_seq, 100, 100))
results.append(['ucb1window random init:'] + play_game(game, ucbwin_ran, 100, 100))
results.append(['thompson non optimistic']+ play_game(game, thomp_non_opt, 100, 100))
results.append(['thompson optimistic:'] +play_game(game, thomp_optimistic, 100, 100))
results.append(['rexp3 gamma=0.99:'] +play_game(game, rexp3, 100, 100))

output = open("out.csv",'w')
for ep in range(101):
    epres = []
    for ex in range(len(results)):
        epres.append('"'+str(results[ex][ep])+'"')
    output.write(",".join(epres)+'\n')
output.close()

