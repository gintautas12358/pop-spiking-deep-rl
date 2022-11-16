
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
from scipy.linalg import sqrtm

# checkpoint_path = "best.pth"
checkpoint_path = "best_hole_run_0_dilated_1e-4_4_128.pth"
# checkpoint_path = "best_hole_run_2_dilated_1e-4_4_32_2.pth"
# checkpoint_path = "best_hole_run_5_dilated_1e-4_4_32_2.pth"





# latent_space_size = 32


# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def t2n(t):
    t = (t+1)/2 
    tt = t.numpy().transpose(1, 2, 0)
    tt *= 255
    tt = tt.astype(np.uint8)

    return tt


net = fsvae.FSVAE()




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


real_img_vector_batch = np.array([])
recon_img_vector_batch = np.array([])
for i in range(int(1e+3)):

    net = net.eval()
    with torch.no_grad():


        real_img = transform(input_img)
        

        # direct spike input
        spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)

        x_recon, q_z, p_z, sampled_z = net(spike_input, scheduled=True)

        # tt = cv2.resize(t2n(sampled_z), (20, 720)) 
        # cv2.imshow("latent z", tt)

        # input image show
        # cv2.imshow("real_img", t2n(real_img))

        # # output image show
        # cv2.imshow("reconstructed_img", t2n(x_recon[0]))
        # cv2.waitKey(0)

        real_img_vector = t2n(real_img).flatten()
        recon_img_vector = t2n(x_recon[0]).flatten()
        # print(real_img_vector.shape)

        if real_img_vector_batch.size == 0:
            real_img_vector_batch = real_img_vector
            recon_img_vector_batch = recon_img_vector
        else:
            real_img_vector_batch = np.vstack([real_img_vector_batch, real_img_vector])
            recon_img_vector_batch = np.vstack([recon_img_vector_batch, recon_img_vector])


# act1 = np.random.random(10*2048)
# act1 = act1.reshape((10,2048))
# act2 = np.random.random(10*2048)
# act2 = act2.reshape((10,2048))

# print(real_img_vector_batch.shape, len(real_img_vector_batch.shape))
# print(recon_img_vector_batch.shape, len(real_img_vector_batch.shape))
fid_score = calculate_fid(real_img_vector_batch, recon_img_vector_batch)
# fid_score = calculate_fid(real_img_vector_batch, real_img_vector_batch)
# fid_score = calculate_fid(act1, act2)


print(fid_score)




