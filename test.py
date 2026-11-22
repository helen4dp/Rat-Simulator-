from environment import RatEnv
env = RatEnv()

state, info = env.reset()
total = 0
done = False
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
print("steps:", env.steps, "total reward:", total)