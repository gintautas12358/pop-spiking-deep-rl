import torch
from tqdm import tqdm
import numpy as np
import gym
import gym_env
import math
import pickle
import cv2

from sac_cuda_norm import SpikeActorDeepCritic
from replay_buffer_norm import ReplayBuffer

# choosing the model
a = 0, 40

path = ""
# path = "19_09_2022/"
# path = "20_09_2022/"
# path = "21_09_2022/"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_activity"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_activity-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path +f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
# env_name = "PegInHole-rand_events_visual_servoing_guiding_corner_activity"
# param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
# rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_corner_activity-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

#+
env_name = "PegInHole-rand_events_visual_servoing_guiding_vae"
param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae-encoder-dim-10-decoder-dim-10/model{a[0]}_e{a[1]}.pt"
rb_param_path = path + f"params/spike-sac_sac-popsan-PegInHole-rand_events_visual_servoing_guiding_vae-encoder-dim-10-decoder-dim-10/replay_buffer{a[0]}_e{a[1]}.p"

use_cuda = True


num_test_episodes = 10
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


env = gym.make(env_name, sim_speed=1, headless=False, render_every_frame=True)



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


def test_agent(env):
        ###
        # compuate the return mean test reward
        ###
        print("testing env...")
        test_reward_sum = 0
        for j in tqdm( range(num_test_episodes) ):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = env.step(get_action(replay_buffer.normalize_obs(o), True))
                ep_ret += r
                ep_len += 1
                # print(r, ep_ret, o[2])
                # print(r)

                # print(ep_len, o, d)

            test_reward_sum += ep_ret

        print("done testing env")
        return test_reward_sum / num_test_episodes


def render_agent(env):
        ###
        # compuate the return mean test reward
        ###
        print("rendering env...")

        # height, width, layers = frame.shape

        video = cv2.VideoWriter(f'epoch_{env_name}_{a[0]}_{a[1]}.avi', 0, 60, (64,64))

    
        test_reward_sum = 0
        for j in tqdm( range(num_test_episodes) ):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = env.step(get_action(replay_buffer.normalize_obs(o), True))
                frame = env.render_frame()
                video.write(cv2.flip(frame, 0))

                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret

        video.release()

        print("done rendering env")
        return test_reward_sum / num_test_episodes

test_agent(env)
# render_agent(env)
