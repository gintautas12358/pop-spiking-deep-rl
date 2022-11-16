#
#BSD 3-Clause License
#
#
#
#Copyright 2022 fortiss, Neuromorphic Computing group
#
#
#All rights reserved.
#
#
#
#Redistribution and use in source and binary forms, with or without
#
#modification, are permitted provided that the following conditions are met:
#
#
#
#* Redistributions of source code must retain the above copyright notice, this
#
#  list of conditions and the following disclaimer.
#
#
#
#* Redistributions in binary form must reproduce the above copyright notice,
#
#  this list of conditions and the following disclaimer in the documentation
#
#  and/or other materials provided with the distribution.
#
#
#
#* Neither the name of the copyright holder nor the names of its
#
#  contributors may be used to endorse or promote products derived from
#
#  this software without specific prior written permission.
#
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#


import gym
import gym_env

# env = gym.make("PegInHole-test", sim_speed=1, headless=False)
# env = gym.make("PegInHole-v0")
# env = gym.make("PegInHole-rand")
# env = gym.make("PegInHole-rand_events")
# env = gym.make("PegInHole-rand_events_depth")
# env = gym.make("PegInHole-rand_events_visual_servoing")

# env = gym.make("PegInHole-rand_events_visual_servoing_guiding", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_corner_activity2", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_activity2", sim_speed=1, headless=False, render_every_frame=True)


# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_vae", sim_speed=1, headless=False)


# env = gym.make("PegInHole-rand_events_visual_servoing_guiding2_no_coord", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding2_no_coord_rand", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding2_rand", sim_speed=1, headless=False, render_every_frame=True)
env = gym.make("PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord_rand", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_activity2_rand", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord_rand", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_corner_activity2_rand", sim_speed=1, headless=False, render_every_frame=True)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord", sim_speed=1, headless=False)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord_rand", sim_speed=1, headless=False)
# env = gym.make("PegInHole-rand_events_visual_servoing_guiding_vae2_rand", sim_speed=1, headless=False)





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
