import numpy as np
from collections import defaultdict
class stateConfig:
    def __init__(self,num_cols,num_rows,col_air):
        self.num_cols=num_cols
        self.num_rows=num_rows
        self.col_air=col_air
        

        move_up = lambda row, col: (max(row - 1 -self.col_air[col], 0), col)
        move_down = lambda row, col: (min(row + 1-self.col_air[col], num_rows - 1), col)
        move_left = lambda row, col: (max(row-self.col_air[col],0) , max(col - 1, 0))
        move_right = lambda row, col: (max(row-self.col_air[col],0), min(col + 1, num_cols - 1))
        # air_state = lambda row,col: (max( row-self.col_air[col] , 0 ),col)

        self.action_defs = {0: move_up, 1: move_right,
                            2: move_down, 3: move_left}

        # Number of states/actions
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        self.isd=np.zeros(nS)
        self.iqd=np.zeros(nS,nA)




        self.grid2state_dict = {(s // num_cols, s % num_cols): s
                                for s in range(nS)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols)
                                for s in range(nS)}

        
        # Gold state
        gold_cell = (3,7)
        gold_state = self.grid2state_dict[gold_cell]
        # trap_states = [self.grid2state_dict[(r, c)]
        #                for (r, c) in trap_cells]
        self.terminal_states = [gold_state] # + trap_states
        print(f"terminal state: {self.terminal_states}")


                # Build the transition probability
        P = defaultdict(dict) # how does it work
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            P[s] = defaultdict(list)
            for a in range(nA):
                action = self.action_defs[a]
                next_s = self.grid2state_dict[action(row, col)]

                # Terminal state
                if self.is_terminal(next_s):
                    r = (1.0 if next_s == self.terminal_states[0] # goal state
                         else -1.0)
                else:
                    r = 0.0
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                P[s][a] = [(1.0, next_s, r, done)]

        # Initial state distribution
        isd = np.zeros(nS)
        # isd[0] = 1.0


        # super().__init__(nS, nA, P, isd)


air_col=[0,0,0,1,1,1,2,2,0]
agent_1=stateConfig(7,10,air_col)

action=np.random.unifor,
traj_1= crnt_state,action