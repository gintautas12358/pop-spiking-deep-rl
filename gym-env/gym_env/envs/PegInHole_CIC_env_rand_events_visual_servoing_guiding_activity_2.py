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

from .PegInHole_CIC_env_rand_events_visual_servoing_guiding_base import PegInHoleRandomEventsVisualServoingGuidingBase

class PegInHoleRandomEventsVisualServoingGuidingActivity2(PegInHoleRandomEventsVisualServoingGuidingBase):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True, running_events=True):
        super().__init__(sim_speed, headless, render_every_frame, running_events)

        self.env_reset()

        # gym spaces
        position_ac = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(position_ac,), dtype=np.float32)

        position_ob = 3
        img_err = 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(position_ob + img_err,), dtype=np.float32)

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
            e_img, e, num_e = out
            self.img  = self.preprocessing(e_img)

    def observe(self): 
        # # events
        # fixedcamid = 1
        # timestamp = self.data.time  
        # out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        # if out is not None:
        #     e_img, e, num_e = out
        #     self.img  = self.preprocessing(e_img)

        err = self.dist_metric(self.img)

        pose = self.controller.fk()

        observation = (pose[0] - self.current_pose[0]) / (self.ac_position_scale ), \
                      (pose[1] - self.current_pose[1]) / (self.ac_position_scale ), \
                      (pose[2] - self.current_pose[2]) / (self.ac_position_scale ), \
                      err / 45.0

        return observation

    def get_reward(self):
        dx = np.linalg.norm(self.action - self.old_a)
        self.old_a = self.action.copy()
        err = self.dist_metric(self.img)

        reward = 1/(0.01*err+0.04) - 1*dx

        return reward, err


    def env_reset(self):    
        self.img = np.ones((32, 32)) * 127
        self.activity_coord = 31, 31

        self.goal_coord = self.random_goal_coord()

    def change_to_shape(self, a):
        if a.shape == (1, 2):
            return a[0]

        return a

    def preprocessing(self, img):

        img = cv2.flip(img, 0)

        # size crop
        img = np.where(img == 50, 255, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # max_value = np.max(img)
        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        # maxpool
        for i in range(1):
            img = skimage.measure.block_reduce(img, (2,2), np.max)

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
        
        x, y = self.activity_coord
        if out is not None:
            x, y = out
            self.activity_coord = x, y 


        latent_img = np.ones((32, 32)) * 0
        latent_img[int(x),int(y)] = 255

        # cv2.imshow("coord image", latent_img.astype(np.uint8))
        # cv2.waitKey(0)

        v = np.array((x, y))

        g_out = self.goal_coord 
        g_v = np.array(g_out)

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
