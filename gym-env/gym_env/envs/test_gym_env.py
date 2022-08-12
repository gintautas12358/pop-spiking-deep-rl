from PegInHole_CIC_env import PegInHole

env = PegInHole()

# obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)