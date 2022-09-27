import os
import os.path
import numpy as np
import logging
import argparse
# import pycuda.driver as cuda

import torch
import torchvision
import torchvision.transforms as transforms



import global_v as glv

import fsvae_models.fsvae as fsvae
from PIL import Image

import cv2

def t2n(t):
    t = (t+1)/2 
    tt = t.numpy().transpose(1, 2, 0)
    tt *= 255
    tt = tt.astype(np.uint8)

    return tt


net = fsvae.FSVAE()

checkpoint_path = "best.pth"
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint)    

input_img_path = "data/hole/preproc/rim/00002703.png"
input_img = Image.open(input_img_path)

n_steps = 4
max_epoch = 0

mean_q_z = 0
mean_p_z = 0
mean_sampled_z = 0

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(148),
    # transforms.Resize((input_size,input_size)),
    # transforms.RandomAffine([-180, 180], [0.5, 0.5], [0.3, 1.1], fill=127),
    transforms.RandomAffine([-180, 180], translate=[0.5, 0.5], fill=127),
    transforms.ToTensor(),
    SetRange
    ])


while(True):

    net = net.eval()
    with torch.no_grad():


            real_img = transform(input_img)
            

            # direct spike input
            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)

            x_recon, q_z, p_z, sampled_z = net(spike_input, scheduled=True)

            tt = cv2.resize(t2n(sampled_z), (20, 720)) 
            cv2.imshow("latent z", tt)

            # input image show
            cv2.imshow("real_img", t2n(real_img))

            # output image show
            cv2.imshow("reconstructed_img", t2n(x_recon[0]))
            cv2.waitKey(0)


