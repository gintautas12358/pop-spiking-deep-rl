import os
import os.path
import numpy as np
import logging
import argparse
# import pycuda.driver as cuda

import torch
import torchvision
import torchvision.transforms as transforms



import gym_env.envs.fsvae.global_v as glv
# import gym_env.envs.fsvae.fsvae_models.fsvae
import gym_env.envs.fsvae.fsvae_models.fsvae as fsvae
from PIL import Image

import cv2

class FSVAE:

    def __init__(self) -> None:

        # init network
        self.net = fsvae.FSVAE()

        #load best trained fsvae model 
        # checkpoint_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/fsvae/best.pth"
        checkpoint_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/fsvae/best_hole_run_0_dilated_1e-4_4_128.pth"
        
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint)  

        self.n_steps = 4

        # inference mode on
        self.net = self.net.eval()

        # init tansforms
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(148),
            # transforms.Resize((input_size,input_size)),
            # transforms.RandomAffine([-180, 180], [0.5, 0.5], [0.3, 1.1], fill=127),
            # transforms.RandomAffine([-180, 180], translate=[0.5, 0.5], fill=127),
            transforms.ToTensor(),
            SetRange
            ])

    def input_image(self, img):
        x_recon = None
        sampled_z = None
        real_img = self.transform(img)

        with torch.no_grad():
            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_steps) # (N,C,H,W,T)
            x_recon, q_z, p_z, sampled_z = self.net(spike_input, scheduled=True)

        # return self.t2n(x_recon[0]), self.t2n(sampled_z)
        return self.t2n(x_recon[0]), sampled_z.numpy().transpose(1, 2, 0)


    def t2n(self, t):
        t = (t+1)/2 
        tt = t.numpy().transpose(1, 2, 0)
        tt *= 255
        tt = tt.astype(np.uint8)

        return tt

