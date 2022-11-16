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


import torch
from tqdm import tqdm
import numpy as np
import gym
import gym_env
import math
import pickle
import cv2
import os

from sac_cuda_norm import SpikeActorDeepCritic
from replay_buffer_norm import ReplayBuffer

import matplotlib.pyplot as plt

# choosing the model
a = 0, 100



path = ""
# path = "19_09_2022/"
# path = "20_09_2022/"
# path = "21_09_2022/"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_activity2"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding2"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path +f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_corner_activity2"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_vae2"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#########################
#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding2_no_coord"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2_no_coord-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path +f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2_no_coord-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"


#########################
#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_activity2_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding2_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path +f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_corner_activity2_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_vae2_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"


#########################
#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding2_no_coord_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2_no_coord_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path +f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding2_no_coord_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
env_name = "PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord_rand"
param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord_rand"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord_rand-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord_rand-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"




use_cuda = True


num_test_episodes = 200
max_ep_len = 200

ac_kwargs = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=10,
                     decoder_pop_dim=10,
                     mean_range=(-3, 3),
                     std=math.sqrt(0.15),
                     spike_ts=5,
                     device=torch.device('cuda'))

replay_size = int(1e6)
norm_clip_limit = 3
norm_update = 50



# Set device
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


env = gym.make(env_name, sim_speed=32, headless=False, render_every_frame=True)



obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


ac = SpikeActorDeepCritic(env.observation_space, env.action_space, **ac_kwargs)
ac.popsan.load_state_dict(torch.load(param_path))
ac.to(device)



# replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
#                                  clip_limit=norm_clip_limit, norm_update_every=norm_update)    
   

 # Experience buffer        
 #self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,        
 #                            clip_limit=norm_clip_limit, norm_update_every=norm_update)        
#  self.dir_path = os.path.dirname(os.path.realpath(__file__))        
replay_buffer = pickle.load(open(rb_param_path, "rb"))        
#  self.ep_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,                                    
#                                     clip_limit=norm_clip_limit, norm_update_every=norm_update) #None


def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1,
                      deterministic)


def plot_data(agents_accuracy):

    min_run_length = 300
    max_run_length = 0
    for a in agents_accuracy:
        if len(a) < min_run_length:
            min_run_length = len(a)

        if len(a) > max_run_length:
            max_run_length = len(a)

    # for a in agents_accuracy:
    #     size = len(a)
    #     if size < max_run_length:
    #         for i in range(max_run_length - size):
    #             a.append(a[size-1])



    fig, axs = plt.subplots(2)

    plt.xlabel('episode steps')
    plt.ylabel('pixel error')

    for i, a in enumerate(agents_accuracy):
        axs[0].plot(range(len(a)), a, label = "line" + str(i))

    runs_mean = []
    for step in range(max_run_length):
        sum = 0
        for r in agents_accuracy:
            if step >= len(r):
                continue

            sum += r[step]
        
        runs_mean.append(1.0 * sum / len(agents_accuracy))

    axs[1].plot(runs_mean)

    axs[1].set_xlabel('episode steps')
    axs[1].set_ylabel('pixel error')

    axs[0].set_xlabel('episode steps')
    axs[0].set_ylabel('pixel error')

    plt.legend()
    plt.show()


def write_data_to_txt(agents_accuracy, file_path):
    with open(file_path, "w") as f:
        f.write("")

    with open(file_path, "a") as f:
        for a in agents_accuracy:
            for i, step in enumerate(a): 
                if i < len(a)-1:
                    f.write(str(step) + ",")
                else:
                    f.write(str(step))

            f.write("\n")




def test_agent_accuracy(env):
        ###
        # compuate the return mean test reward
        ###

        agents_accuracy = []
        agents_real_accuracy = []

        print("testing env...")
        test_reward_sum = 0
        reach_count = 0
        for j in tqdm( range(num_test_episodes) ):

            accuracy = []
            accuracy_real = []
            
            has_reached = False

            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = env.step(get_action(replay_buffer.normalize_obs(o), True))
                ep_ret += r
                ep_len += 1
                # print(r, ep_ret, o[2])
                accuracy.append(env.err)
                real_err = np.linalg.norm(env.controller.fk()[:2] - np.array(env.goal_coord))
                accuracy_real.append(real_err)

                if env.err <= 2:
                    has_reached = True

            test_reward_sum += ep_ret
            agents_accuracy.append(accuracy)
            agents_real_accuracy.append(accuracy_real)

            if has_reached:
                reach_count += 1

        print("reach statistics:", reach_count, num_test_episodes, reach_count * 1.0 / num_test_episodes)

        file_path_vision = "collected_data.txt"
        write_data_to_txt(agents_accuracy, file_path_vision)
        
        plot_data(agents_accuracy)

        # file_path_real = "collected_data_real.txt"
        # write_data_to_txt(agents_real_accuracy, file_path_real)  

        # plot_data(agents_real_accuracy)      
        
        print("done testing env")
        return test_reward_sum / num_test_episodes


test_agent_accuracy(env)
