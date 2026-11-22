import numpy as np
import matplotlib.pyplot as plt
from environment import RatEnv

np.random.seed(42)

env = RatEnv()
q_table = np.zeros((env.observation_space.n, env.action_space.n))

episodes = 2000                  # number of episodes the rat is gonna do, so it dosent try infinitos caminhos
E = 0.2
alfa = 0.1
gama = 0.99

def choose_action(q_table, state, E):
    
    n = np.random.random()
    if n >= E: r = np.argmax(q_table[state])
    else: r = np.random.randint(4)
    
    return(r) 


for ep in range(episodes):              #loop 

    state, info = env.reset()
    done = False                        #variable to stop the inner loop when terminated or truncated is true
    total = 0                           # counter for the sum of all the reward in this episode
    

    while not done:
        action = choose_action(q_table, state, E)

        new_state, reward, terminated, truncated, info = env.step(action)

        total += reward 
        done = terminated or truncated              # if terminated or truncated is true, done is true

        # Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
        # target  = reward + gamma * max(Q[new_state])

        if terminated == False: target = (reward + gama * np.max(q_table[new_state])) 
        else: target = reward

        q_table[state, action] = q_table[state, action] + alfa * (target - q_table[state, action])

        state = new_state

    if ep % 100 == 0: print(f"Episode: {ep}, total: {total}")                #to see how the rat improves at each episode
    E = max(0.01, E*0.999) 
    
state, info = env.reset()
path = [env.pos] 
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    path.append(env.pos)
    done = truncated or terminated 

col = []
row = []
for pair in path:
    col.append(pair[1])
    row.append(pair[0])


plt.imshow(env.height)
plt.colorbar() 
plt.plot(col, row, c = "red", lw = 1, ms = 2)
plt.scatter (0, 0, c = "yellow", s = 100, marker = "o", label = "Inicial spot") 
plt.scatter(env.size -1, env.size - 1, c = "green", s = 100, marker = "o", label = "Goal")
plt.savefig("path.png")
