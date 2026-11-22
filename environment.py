import numpy as np
import gymnasium as gym
from gymnasium import spaces                

class RatEnv(gym.Env):                      # gym.Env cria e padroniza o ambiente (class) de \RL           

    def __init__(self):                     #  definindo as propriedades do ambiente 
        self.size = 50                      # tamanho máximo

        A = 5
        sigma = 0.5

        x = np.linspace(-1, 1, self.size)                                              # np.linspace(start, stop, count) => it gives a NumPy array
        X, Y = np.meshgrid(x, x) 

        self.height = A * np.exp(-((X**2) + (Y**2))/(sigma**2))
        self.start = (0,0)
        self.dist = 100
        self.goal = ((self.size - 1),(self.size - 1))
        self.goal_bonus = 100
        self.climb = 10
        self.base = 1
        self.moves = [(-1, 0), (1,0), (0, -1), (0,1)]               # index = 0, 1, 2, 3
        self.max_steps = 1000                                        # limite seguro para cada episodio 

        self.action_space = spaces.Discrete(4)              # the action space defines how many actions the rat can choose, and discrete() especify the number (0-3 -> 4 ações)
        self.observation_space = spaces.Discrete(self.size * self.size)             # the observation space defines how many states the rat can be in, and the discrete especify the number of this existent states

    def _to_state(self, pos):               # function that helps know where the rat is in the map by row and col
        row, col = pos
        state = row*(self.size) + col
        
        return (state)
    
    def reset(self, seed=None, options=None):                    
        super().reset(seed=seed)
        self.pos = self.start
        self.steps = 0
        return self._to_state(self.pos), {}


    def step(self, action):
        row, col = self.pos 
        d_row, d_col = self.moves[action]
        new_row = row + d_row
        new_col = col + d_col

        if new_col > (self.size - 1) or new_col < 0 or new_row > (self.size - 1) or new_row < 0: 
            new_col = col
            new_row = row     
        
        h_from = self.height[row, col]
        h_to = self.height[new_row, new_col]

        dH = h_to - h_from
        
        dProximo = ((self.size - 1) - new_row) + ((self.size - 1) - new_col)
        dAtual = ((self.size - 1) - row) + ((self.size - 1) - col)

        if dH < 0: 
            energy = self.base 
        else:
            energy = self.base + self.climb*dH                                                    
            
        PotencialShaping =  0.99 * (dAtual - dProximo) 

        self.pos = new_row, new_col
        self.steps = self.steps + 1

        if self.pos == self.goal: 
            reward = - energy + self.goal_bonus + PotencialShaping
            terminated = True
            truncated = False
        else: 
            reward = - energy + PotencialShaping
            if(self.steps == self.max_steps): 
                truncated = True
                terminated = False
            else: 
                truncated = False
                terminated = False
    
        return self._to_state(self.pos), reward, terminated, truncated, {} 
