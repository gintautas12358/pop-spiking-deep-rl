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


import mujoco
import mujoco_viewer
import os
import time
import numpy as np
from .controllers.full_impedance_controller import FullImpedanceController
# from utils.read_cfg import get_mjc_xml, get_jposes, get_cposes, get_cerr_lim

from .utils.kinematics import current_ee_position
from .utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat, quat2eul, quat2Vel

import gym
import cv2
import skimage.measure
import time
from scipy.spatial import distance_matrix

from .fsvae.fsvae import FSVAE
from .PegInHole_CIC_env_rand_events_visual_servoing_guiding_base import PegInHoleRandomEventsVisualServoingGuidingBase

class PegInHoleRandomEventsVisualServoingGuidingVAE2(PegInHoleRandomEventsVisualServoingGuidingBase):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True, running_events=False):
        super().__init__(sim_speed, headless, render_every_frame, running_events)
        # print("################# 2 ########################")
        self.env_reset()

        # gym spaces
        position_ac = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(position_ac,), dtype=np.float32)

        position_ob = 3
        vae = 128*4
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(position_ob + vae,), dtype=np.float32)

    def random_goal_coord(self):
        min_c = 4
        max_c = 29
        goal_coord = np.random.randint(min_c, max_c), np.random.randint(min_c, max_c)
        goal_coord = 5,5 # rezults with 5,5
        # print(goal_coord)

        # latent_img = np.ones((32, 32)) * 0
        # latent_img[goal_coord[0], goal_coord[1]] = 255

        # cv2.imshow("goal coord image", latent_img.astype(np.uint8))
        # cv2.waitKey(0)

        return goal_coord

    def get_pose(self, action):

        pose = self.current_pose.copy()
        ac_position = action[:2]
        pose[:2] +=  ac_position * self.ac_position_scale 

        self.action = action

        return pose

    def observe_0(self):
        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        if out is not None:
            # print("events created")
            e_img, e, num_e = out
            self.img  = self.preprocessing2(e_img.copy())

            # img2 = self.preprocessing2(e_img.copy())
            # print("debug", img2.shape)
            recons_img, latent_z = self.fsvae.input_image(self.img)
            self.latent_z = latent_z

            # cv2.imshow("input image 0", self.img)
            # cv2.imshow("recons image 0", recons_img)
            # cv2.waitKey(0)

    def observe(self): 
        # # events
        # fixedcamid = 1
        # timestamp = self.data.time  
        # out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        # if out is not None:
        #     # print("events created")
        #     e_img, e, num_e = out
        #     self.img  = self.preprocessing2(e_img.copy())

        #     # img2 = self.preprocessing2(e_img.copy())
        #     # print("debug", img2.shape)
        #     recons_img, latent_z = self.fsvae.input_image(self.img)
        #     self.latent_z = latent_z

        #     cv2.imshow("input image", self.img)
        #     cv2.imshow("recons image", recons_img)
        #     # cv2.waitKey(0)

        pose = self.controller.fk()

        ob0 = (pose[0] - self.current_pose[0]) / (self.ac_position_scale )
        ob1 = (pose[1] - self.current_pose[1]) / (self.ac_position_scale )
        ob2 = (pose[2] - self.current_pose[2]) / (self.ac_position_scale )
        ob3 = self.latent_z.flatten()

        # print("shape", ob2.shape)

        observation = np.array([])
        observation = np.append(observation, ob0)
        observation = np.append(observation, ob1)
        observation = np.append(observation, ob2)
        observation = np.append(observation, ob3)

        return observation

    def get_reward(self):
        dx = np.linalg.norm(self.action - self.old_a)
        self.old_a = self.action.copy()
        err = self.dist_metric(self.img)

        reward = 1/(0.01*err+0.01) - 1*dx

        return reward, err

    def env_reset(self):    
        self.img = np.ones((32, 32)) * 127
 
        self.activity_coord = 31, 31

        self.latent_z = np.zeros((128, 4))

        self.goal_coord = self.random_goal_coord()
 
        self.fsvae = FSVAE()

    def get_vision_error(self):
        return self.err


    def preprocessing2(self, img):

        img = cv2.flip(img, 0)

        # size crop
        res = 512
        half_size = int(res / 2)

        H, W, _ = img.shape
        mid_x = int(W / 2)
        img = img[H - res:H, mid_x-half_size:mid_x+half_size]

        img = np.where(img == 50, 255, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        # maxpool
        for i in range(4):
            img = skimage.measure.block_reduce(img, (2,2), np.max)

        # dilate
        grow_size = 3
        kernel = np.ones((grow_size,grow_size), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        # gray background

        img = np.where(img == 0, 127, img)
        img = np.where(img == 129, 0, img)

         # applied noise
        # img = self.apply_event_noise(img, 1, (32,32))

        # observe result. (debug camera view) 
        # cv2.imshow("resized image", img)
        # cv2.waitKey(0)

        return img

    def dist_metric(self, img):
        out = self.get_activity_coord(img)

        # img = cv2.resize(img, (512, 512)) 
        # cv2.imshow("seen image", img)
        
        if out is not None:
            x, y = out
            self.activity_coord = x, y 
        else:
            x, y = self.activity_coord

        v = np.array((x, y))

        g_out = self.goal_coord 
        g_v = np.array(g_out)

        latent_img = np.ones((32, 32)) * 0
        latent_img[int(x),int(y)] = 255

        # cv2.imshow("coord image", latent_img.astype(np.uint8))
        # cv2.waitKey(0)

        err = np.linalg.norm(g_v - v)
        
        return err

    def get_activity_coord(self, img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)

        if x.size == 0:
            return None

        x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()

        return x_mean, y_mean