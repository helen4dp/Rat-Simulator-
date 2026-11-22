
import numpy as np
import matplotlib.pyplot as plt

# ========= Building the world =============

A = 5
sigma = 0.5

x = np.linspace(-1, 1, 50)                                              # np.linspace(start, stop, count) => it gives a NumPy array
X, Y = np.meshgrid(x, x)                                                # X = columns, Y = rows         # meshgrid returns two things, and Python lets you catch them in two variables at once

height = A * np.exp(-((X**2) + (Y**2))/(sigma**2))

start = (0, 0)
goal = (49, 49)

# ======================== Setting the energy of wich move =======

base = 1
climb = 10 
down = 1

# ====================== Moves ========

moves = { 
    "UP": (- 1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# =================== defining the energy of one step =====

def energy (h_from, h_to):                                     # From e to da altura 
    dH = h_to - h_from                  #height change

    if dH < 0:
        return base 
    else:
        return base + climb*dH

# ====================== defining the step ===========
def step(pos, action):                                  # ths function receives two inputs: pos, that is the current posion of the rat, and action, that is one of the previous defined moves (UP, DOWN..)
    row, col = pos
    d_row, d_col = moves[action]                         # delta_row, delta_col -> action gives how much the rat has to walk at wich direction 
    new_row = row + d_row
    new_col = col + d_col

    if new_row < 0 or new_row > 49 or new_col < 0 or new_col > 49:              # checking to see if the move is possible (if it dosent step out of the map)
        return (row, col), base
    else:
        h_from = height(row, col)               # calculando a altura do ponto atual
        h_to = height(new_row, new_col)             # calculando a altura do ponto que queremos atingir 

        cost = energy(h_from, h_to)            # calculando o custo para ir do ponto a para o ponto b 

        return (new_row, new_col), cost      # return the new posision (x, y) and the coast to active the new posision


