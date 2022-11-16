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

from .PegInHole_CIC_env_rand_events_visual_servoing_guiding_activity_2 import PegInHoleRandomEventsVisualServoingGuidingActivity2

class PegInHoleRandomEventsVisualServoingGuidingActivity2NC(PegInHoleRandomEventsVisualServoingGuidingActivity2):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True, running_events=True):
        super(PegInHoleRandomEventsVisualServoingGuidingActivity2, self).__init__(sim_speed, headless, render_every_frame, running_events)

        self.env_reset()

        # gym spaces
        position_ac = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(position_ac,), dtype=np.float32)

        position_ob = 3
        img_pos_ob = 2
        img_err = 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(img_pos_ob + img_err,), dtype=np.float32)

   
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

        observation = (self.activity_coord[0] - 16) / (32.0 ), \
                      (self.activity_coord[1] - 16) / (32.0 ), \
                      err / 45.0

        return observation
