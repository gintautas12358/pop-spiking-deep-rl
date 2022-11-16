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
from .PegInHole_CIC_env_rand_events_visual_servoing_guiding_vae_2 import PegInHoleRandomEventsVisualServoingGuidingVAE2

class PegInHoleRandomEventsVisualServoingGuidingVAE2NC(PegInHoleRandomEventsVisualServoingGuidingVAE2):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True, running_events=False):
        super(PegInHoleRandomEventsVisualServoingGuidingVAE2, self).__init__(sim_speed, headless, render_every_frame, running_events)
        
        self.env_reset()

        # gym spaces
        position_ac = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(position_ac,), dtype=np.float32)

        position_ob = 3
        vae = 128*4
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(0 + vae,), dtype=np.float32)

    def observe(self): 
        # # events
        # fixedcamid = 1
        # timestamp = self.data.time  
        # out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        # if out is not None:
        #     # print("events created")
        #     e_img, e, num_e = out
        #     self.img  = self.preprocessing2(e_img.copy())

        #     img2 = self.preprocessing2(e_img.copy())
        #     # print("debug", img2.shape)
        #     recons_img, latent_z = self.fsvae.input_image(img2)
        #     self.latent_z = latent_z

        #     # cv2.imshow("recons image", recons_img)
        #     # cv2.waitKey(0)

        pose = self.controller.fk()

        # ob0 = (pose[0] - self.current_pose[0]) / (self.ac_position_scale )
        # ob1 = (pose[1] - self.current_pose[1]) / (self.ac_position_scale )
        # ob2 = (pose[2] - self.current_pose[2]) / (self.ac_position_scale )
        ob3 = self.latent_z.flatten()

        # print("shape", ob2.shape)

        # observation = np.array([])
        # observation = np.append(observation, ob0)
        # observation = np.append(observation, ob1)
        # observation = np.append(observation, ob2)
        # observation = np.append(observation, ob3)
        observation = ob3

        return observation

   