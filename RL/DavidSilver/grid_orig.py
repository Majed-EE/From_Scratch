# coding: utf-8

import numpy as np
from gym.envs.toy_text import discrete
from collections import defaultdict
import time
import pickle
import os
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
CELL_SIZE = 100
MARGIN = 10


def get_coords(row, col, loc='center'):
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE//2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = xc - half_size, xc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE//3
        x2, y2 = xc + CELL_SIZE//3, yc - CELL_SIZE//3
        x3, y3 = xc - CELL_SIZE//3, yc - CELL_SIZE//3
        return [(x1, y1), (x2, y2), (x3, y3)]


def draw_object(coords_list):
    if len(coords_list) == 1:  # -> circle
        obj = rendering.make_circle(int(0.45*CELL_SIZE))
        obj_transform = rendering.Transform()
        obj.add_attr(obj_transform)
        obj_transform.set_translation(*coords_list[0])
        obj.set_color(0.2, 0.2, 0.2)  # -> black
    elif len(coords_list) == 3:  # -> triangle
        obj = rendering.FilledPolygon(coords_list)
        obj.set_color(0.9, 0.6, 0.2)  # -> yellow
    elif len(coords_list) > 3:  # -> polygon
        obj = rendering.FilledPolygon(coords_list)
        obj.set_color(0.4, 0.4, 0.8)  # -> blue
    return obj


class GridWorldEnv(discrete.DiscreteEnv):
    def __init__(self, num_rows=4, num_cols=6, delay=0.05,col_air=[0,0,0,1,1,1,2,2,1,0]):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.col_air = col_air
        self.delay = delay
        self.path = [] # agent path
        self.image_save_path = "agent_paths" 
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

        move_down = lambda row, col: (min(max(row - 1 +self.col_air[col], 0), num_rows-1), col)
        move_up = lambda row, col: (min( max(row + 1+self.col_air[col] ,0 ), num_rows - 1), col)
        move_left = lambda row, col: (max(row-self.col_air[col],0) , max(col - 1, 0))
        move_right = lambda row, col: (max(row-self.col_air[col],0), min(col + 1, num_cols - 1))

        self.action_defs = {0: move_up, 1: move_right,
                            2: move_down, 3: move_left}

        # Number of states/actions
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        self.grid2state_dict = {(s // num_cols, s % num_cols): s
                                for s in range(nS)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols)
                                for s in range(nS)}


        # Gold state
        gold_cell = (3,7)

        # Trap states 
        trap_cells =[]


        gold_state = self.grid2state_dict[gold_cell]
        trap_states = [] 
        # [self.grid2state_dict[(r, c)]
        #                for (r, c) in trap_cells]
        self.terminal_states = [gold_state] + trap_states
        print(self.terminal_states)

        # Build the transition probability
        P = defaultdict(dict) # how does it work
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            P[s] = defaultdict(list)
            for a in range(nA):
                action = self.action_defs[a]
                # print(f"state: {s} (row= {row},col= {col}) and action: {a} ")
                next_s = self.grid2state_dict[action(row, col)]

                # Terminal state
                if self.is_terminal(next_s):
                    r = (2.0 if next_s == self.terminal_states[0] # goal state
                         else -2.0)
                else:
                    r = -1.0 # -1 for each step
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                P[s][a] = [(1.0, next_s, r, done)] # what is 1 here

        # Initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        super().__init__(nS, nA, P, isd)

        self.viewer = None
        self._build_display(gold_cell, trap_cells)

    def is_terminal(self, state):
        return state in self.terminal_states

    def _build_display(self, gold_cell, trap_cells):

        screen_width = (self.num_cols + 2) * CELL_SIZE
        screen_height = (self.num_rows + 2) * CELL_SIZE
        self.viewer = rendering.Viewer(screen_width,
                                       screen_height)

        all_objects = []

        # List of border points' coordinates
        bp_list = [
            (CELL_SIZE - MARGIN, CELL_SIZE - MARGIN),
            (screen_width - CELL_SIZE + MARGIN, CELL_SIZE - MARGIN),
            (screen_width - CELL_SIZE + MARGIN,
             screen_height - CELL_SIZE + MARGIN),
            (CELL_SIZE - MARGIN, screen_height - CELL_SIZE + MARGIN)
        ]
        border = rendering.PolyLine(bp_list, True)
        border.set_linewidth(5)
        all_objects.append(border)

        # Vertical lines
        for col in range(self.num_cols + 1):
            x1, y1 = (col + 1) * CELL_SIZE, CELL_SIZE
            x2, y2 = (col + 1) * CELL_SIZE, \
                     (self.num_rows + 1) * CELL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        # Horizontal lines
        for row in range(self.num_rows + 1):
            x1, y1 = CELL_SIZE, (row + 1) * CELL_SIZE
            x2, y2 = (self.num_cols + 1) * CELL_SIZE, \
                     (row + 1) * CELL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        # Traps: --> circles
        for cell in trap_cells:
            trap_coords = get_coords(*cell, loc='center')
            all_objects.append(draw_object([trap_coords]))

        # Gold:  --> triangle
        gold_coords = get_coords(*gold_cell,
                                 loc='interior_triangle')
        all_objects.append(draw_object(gold_coords))

        # Agent --> square or robot
        if (os.path.exists('robot-coordinates.pkl') and CELL_SIZE == 100):
            agent_coords = pickle.load(
                open('robot-coordinates.pkl', 'rb'))
            starting_coords = get_coords(0, 0, loc='center')
            agent_coords += np.array(starting_coords)
        else:
            agent_coords = get_coords(0, 0, loc='interior_corners')
        agent = draw_object(agent_coords)
        self.agent_trans = rendering.Transform()
        agent.add_attr(self.agent_trans)
        all_objects.append(agent)

        for obj in all_objects:
            self.viewer.add_geom(obj)



    def save_path_image(self, step):
        """Save the current path as an image."""
        fig, ax = plt.subplots(figsize=(self.num_cols, self.num_rows))
        ax.set_xlim(0, self.num_cols)
        ax.set_ylim(0, self.num_rows)
        ax.set_xticks(range(self.num_cols + 1))
        ax.set_yticks(range(self.num_rows + 1))
        ax.grid(True)

        # Draw the path
        if self.path:
            path_x, path_y = zip(*self.path)
            ax.plot(path_x, path_y, marker="o", color="blue", label="Path")

        # Save the image
        image_file = os.path.join(self.image_save_path, f"step_{step}.png")
        plt.savefig(image_file)
        plt.close(fig)



    def render(self, mode='human', done=False):
        if done:
            sleep_time = 1
        else:
            sleep_time = self.delay
        x_coord = self.s % self.num_cols
        y_coord = self.s // self.num_cols
        x_coord = (x_coord + 0) * CELL_SIZE
        y_coord = (y_coord + 0) * CELL_SIZE
        self.agent_trans.set_translation(x_coord, y_coord)
        rend = self.viewer.render(
            return_rgb_array=(mode == 'rgb_array'))
        time.sleep(sleep_time)

        self.path.append((x_coord + 0.5, y_coord + 0.5))  # Append the agent's position
        # Save the path image at each step
        # self.save_path_image(len(self.path))

        return rend

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = GridWorldEnv(7, 10)
    q_table=np.random.rand(70,4)
    
    
    alpha = 0.5
    gamma = 0.1
    epsilon = 0.1 
    episodes=100
    all_reward=[]
    for ep in range(episodes):
        s = env.reset()
        ep_r=0
        env.render(mode=None, done=False)
        print(f"episode: {ep}")
        a_next= np.argmax(q_table[s])
        while True:
            
            # time step t
            action = a_next if (np.random.randn()>epsilon) else np.random.choice(env.nA) 
            # a_next #np.argmax(q_table[s]) 
            res = env.step(action)
            
            # print('Action ', env.s, action, ' -> ', res) # P[s][a] = [(1.0, next_s, r, done)] # what is 1 here
            s_next= res[0]
            
            R=res[1]
            ep_r+=R
            a_next = np.argmax(q_table[s_next])
            print(f"current state: {env.state2grid_dict[s]}/nCurrent Action: {action}\nnext_state: {env.state2grid_dict[s_next]}\nq_table[s]")
              
            # env.render(mode=None, done=res[2])
            if res[2]:
                print(f"###############terminating episode{ep} ###########\nTotal Reward:: {ep_r}")
                q_table[s][action] = q_table[s][action] + alpha*( R  )
                break       
            else: q_table[s][action] = q_table[s][action] + alpha*( R + gamma* (q_table[s_next][a_next] - q_table[s][action]) )

                
        all_reward.append(ep_r)

    env.close()

    all_reward=np.asarray(all_reward)
    x=np.linspace(0,all_reward.shape[0],all_reward.shape[0])
    plt.plot(all_reward,x)
    plt.show()