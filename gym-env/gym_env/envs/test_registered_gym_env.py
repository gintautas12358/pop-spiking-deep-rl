import gym
import gym_env

# env = gym.make("PegInHole-v0")
# env = gym.make("PegInHole-rand")
# env = gym.make("PegInHole-rand_events")
# env = gym.make("PegInHole-rand_events_depth")

env = gym.make("PegInHole-rand_events_visual_servoing")





obs = env.reset()

done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    step += 1
    # print("obs", obs, "reward", reward, "done", done, "info", info)
    # print("obs", obs[0][6:])

    # print(step)

    if step > 100:
        done