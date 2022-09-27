import gym
import gym_env

# env = gym.make("PegInHole-test", sim_speed=1, headless=False)
# env = gym.make("PegInHole-v0")
# env = gym.make("PegInHole-rand")
# env = gym.make("PegInHole-rand_events")
# env = gym.make("PegInHole-rand_events_depth")
# env = gym.make("PegInHole-rand_events_visual_servoing")

# env = gym.make("PegInHole-rand_events_visual_servoing_guiding", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_corner_activity", sim_speed=1, headless=False, render_every_frame=True)

env = gym.make("PegInHole-rand_events_visual_servoing_guiding_vae", sim_speed=1, headless=False)








obs = env.reset()

done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    step += 1
    # print("obs", obs, "reward", reward, "done", done, "info", info)
    # print("obs", obs[0][6:])

    print(step)

    if step > 100:
        # done = True
        obs = env.reset()
        step = 0
