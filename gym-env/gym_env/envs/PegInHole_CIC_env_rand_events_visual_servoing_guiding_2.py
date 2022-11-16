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

class PegInHoleRandomEventsVisualServoingGuiding2(PegInHoleRandomEventsVisualServoingGuidingBase):
    
    def __init__(self, sim_speed=32, headless=False, render_every_frame=True, running_events=True):
        super().__init__(sim_speed, headless, render_every_frame, running_events)

        self.env_reset()

        # gym spaces
        position_ac = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(position_ac,), dtype=np.float32)

        position_ob = 3
        img_err = 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(position_ob + img_err,), dtype=np.float32)


    def random_img_goal(self):
        img_files = ["00000437.png", "00001262.png", "00000531.png", "00002562.png"]
        
        index = np.random.randint(0,4)
        img_file = img_files[index]
        goal_img_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/goal_image/" + img_file

        min, max = 7, 55
        self.goal_coord = (np.random.randint(min,max), np.random.randint(min,max))
        self.goal_coord = 10, 10 # rezults with 52, 10
        # print(goal_coord)
        goal_img = self.relocate_img(cv2.imread(goal_img_path), self.goal_coord, 64)

        return goal_img

    def get_pose(self, action):

        pose = self.current_pose.copy()
        ac_position = action[:2]
        pose[:2] +=  ac_position * self.ac_position_scale 

        self.action = action

        return pose

    def observe_0(self): 
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        if out is not None:
            # print("events created")
            e_img, e, num_e = out
            self.img  = self.preprocessing(e_img)
            # self.img = self.apply_event_noise(self.img, 1, (32,32))

    def observe(self): 
        # fixedcamid = 1
        # timestamp = self.data.time  
        # out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        # if out is not None:
        #     # print("events created")
        #     e_img, e, num_e = out
        #     self.img  = self.preprocessing(e_img)

        err = self.dist_metric(self.goal_img, self.img)

        pose = self.controller.fk()

        observation = (pose[0] - self.current_pose[0]) / (self.ac_position_scale ), \
                      (pose[1] - self.current_pose[1]) / (self.ac_position_scale ), \
                      (pose[2] - self.current_pose[2]) / (self.ac_position_scale ), \
                      err / 45.0

        return observation

    def get_reward(self):
        dx = np.linalg.norm(self.action - self.old_a)
        self.old_a = self.action.copy()
        err = self.dist_metric(self.goal_img, self.img)

        reward = 1/(0.01*err+0.04) - 1* dx

        return reward, err

    def env_reset(self):    
        self.img = np.ones((32, 32)) * 127
        self.c = np.array([[32, 32]])

        goal_img = self.random_img_goal()
        self.goal_img = self.preprocessing(goal_img)

    def preprocessing(self, img):

        img = cv2.flip(img, 0)

        # size crop
        img = np.where(img == 50, 255, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    def dist_metric(self, goal_img, img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)
        c = np.append(x[:, None], y[:,None], axis=1)

        if x.size == 0:
            c = self.c
        else: 
            self.c = c 
        
        gpx, gpy = np.where(goal_img == 0)
        gnx, gny = np.where(goal_img == 255)
        gx = np.append(gnx, gpx)
        gy = np.append(gny, gpy)
        gc = np.append(gx[:, None], gy[:,None], axis=1)
        
        dist_mat = distance_matrix(c, gc)
        
        self_dist_mat = distance_matrix(gc, gc)
        
        dist = np.abs(dist_mat.flatten()).mean() - np.abs(self_dist_mat.flatten()).mean()
        self.dist = dist
        return dist


    def relocate_img(self, img, pos, res):
        size = 10
        # print("pos", pos)
        center_img = self.center_crop(img, size)
        new_img = self.translate_img_from_center(center_img, pos, size, res)

        return new_img

    def center_crop(self, img, size):
        half_size = int(size / 2)

        positions_y, positions_x, channel = np.where(img > 0)
        # print(positions_y, positions_x, channel)
        min_position_x = np.min(positions_x)
        max_position_x = np.max(positions_x)
        min_position_y = np.min(positions_y)
        max_position_y = np.max(positions_y)

        mid_x = int((min_position_x + max_position_x) / 2)
        mid_y = int((min_position_y + max_position_y) / 2)

        # print(min_position_x, max_position_x, min_position_y, max_position_y)
        crop_y0 = mid_y-half_size
        crop_y1 = mid_y+half_size
        crop_x0 = mid_x-half_size
        crop_x1 = mid_x+half_size
        # crop_img = img[mid_y-half_size:mid_y+half_size, mid_x-half_size:mid_x+half_size]
        # print(crop_y0, crop_y1, crop_x0, crop_x1)
        crop_img = img[crop_y0:crop_y1, crop_x0:crop_x1, :]
        return crop_img

    def translate_img_from_center(self, img, pos, size, res):
        # print(img.shape)
        half_size = int(size / 2)
        mid_x = int(res / 2)
        mid_y = int(res / 2)
        new_img = np.zeros((res,res, 3))
        x_offset = pos[0] - mid_x
        y_offset = pos[1] - mid_y
        # print("new", new_img.shape)

        # print("bug", mid_x, half_size, mid_x-half_size, mid_x+half_size)

        crop_y0 = mid_y-half_size
        crop_y1 = mid_y+half_size
        crop_x0 = mid_x-half_size
        crop_x1 = mid_x+half_size

        # print(crop_y0, crop_y1, crop_x0, crop_x1)
        # print(x_offset, y_offset)

        # new_img[mid_y-half_size + x_offset:mid_y+half_size + x_offset, mid_x-half_size + y_offset:mid_x+half_size + y_offset, :] = img[:,:,:]
        # new_img[crop_y0 + x_offset:crop_y1 + x_offset, crop_x0 + y_offset:crop_x1 + y_offset, :] = img[:,:,:]
        new_img[crop_x0 + x_offset:crop_x1 + x_offset, crop_y0 + y_offset:crop_y1 + y_offset, :] = img[:,:,:]


        new_img[:,:] = new_img[::-1,:]

        return new_img.astype(np.uint8)
