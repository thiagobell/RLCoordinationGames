class Game:

    def __init__(self, path):
        f = [x.replace('\n','') for x in open(path).readlines()]
        if(len(f) == 0):
            raise "invalid game file"
        

        self.num_actions_col = None
        self.max_reward = None
        self.game_matrix = []
        for line in f:
            values = map(int,line.split(' '))
            #validating game matrix 
            if self.num_actions_col == None:
                self.num_actions_col = len(values)
            elif self.num_actions_col != len(values):
                raise "number of plays for column player does not match"
            self.max_reward = max(self.max_reward, max(values))
            self.game_matrix.append(values)
        self.num_actions_row = len(f)

    def reward(self,action_row, action_col):
        return self.game_matrix[action_row][action_col]
    
    def ratio_of_maximum(self,reward):
        return float(reward)/self.max_reward
