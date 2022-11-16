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

class PegInHoleRandomEventsVisualServoingGuidingBase(gym.Env):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True, running_events=True):
        
        xml_path = "/home/palinauskas/Documents/mujoco-eleanor/kuka/envs/assets/full_kuka_INRC3_mounted_camera_hole.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # init first position
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, headless=headless, render_every_frame=render_every_frame, running_events=running_events)
        self.controller = FullImpedanceController(self.model, self.data) 

        # inti esim
        self.viewer.init_esim(contrast_threshold_negative=0.9, contrast_threshold_positive=0.9, refractory_period_ns=100)
        # print("################# 1 ########################")

        # simulation speed
        self.viewer.set_sim_speed(sim_speed)

        # in radians
        # self.data.qpos = np.array([-1.57,0,0,1.57,0,-1.57,0]) # init pose
        self.init_pose = np.array([-1.57,
                                    -0.68,
                                    0,
                                    1.85,
                                    0.0,
                                    -0.6,
                                    -1.57])

        self.data.qpos = self.init_pose

        self.init_qvel = self.data.qvel.copy()
        self.init_act = self.data.act.copy()
        self.init_qacc_warmstart = self.data.qacc_warmstart.copy()

        self.init_ctrl = self.data.ctrl.copy()
        self.init_qfrc_applied = self.data.qfrc_applied.copy()
        self.init_xfrc_applied = self.data.xfrc_applied.copy()
        self.init_qacc = self.data.qacc.copy()
        self.init_act_dot = self.data.act_dot.copy()

        self.init_time = self.data.time


        self.err = None


        # init hole position
        self.init_hole_pos = self.get_hole_pose()
        self.max_rand_offset = 0.01
        np.random.seed(int(time.time()))

        self.randomize_hole_position()

        self.current_pose = np.array([0.0, 0.43, 0.10, 3.14, 0, 0])
        # self.current_pose = np.array([0.0 ,0.6,0.10, 3.14, 0, 0])  # on top of the hole
        # self.current_pose = np.array([0.0 ,0.6,0.05, 3.14, 0, 0]) # inside the hole
        # self.goal_pose = np.array([0.0, 0.6, 0.02, 3.14, 0, 0])

        self.current_step = 0
        self.old_a = 0

        self.ac_position_scale = 0.3
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 10
        self.ob_orientation_scale = 10
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000
        self.ob_image_scale = 255.0
        self.ob_img_dist_scale = 1000

        
    def get_pose(self, action):
        pass

    def observe(self): 
        pass

    def get_reward(self):
        pass

    def env_reset(self):    
        pass

    def observe_0(self): 
        pass


    def step(self, action):

        # print("hole pos", self.get_hole_pose())
        
        # to check the cartesian position of the initialised joint position
        # print(self.controller.fk())

        # init step gym
        reward = 0
        done = False
        self.current_step += 1

        # ======== apply action ==========
        action = self.change_to_shape(action)

        pose = self.get_pose(action)
        self.controller.set_action(pose)

        for i in range(5):
            self.viewer.make_current()
            self.viewer.render(overlay_on=False)
            torque = self.controller.get_torque()
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
            self.observe_0()

        # ======== observation ==========
        observation = self.observe()

        # ======== reward ==========
        reward, err = self.get_reward()

        # ======== done condition ==========
        condition = self.controller.fk()[0] < self.current_pose[0]-0.15 or \
                    self.controller.fk()[0] > self.current_pose[0]+0.15 or \
                    self.controller.fk()[1] < self.current_pose[1]-0.05 or \
                    self.controller.fk()[1] > self.current_pose[1]+0.6 or \
                    self.controller.fk()[2] > self.current_pose[2]+0.1 or \
                    np.abs(self.controller.fk()[3]) > self.current_pose[3]+0.1 or \
                    np.abs(self.controller.fk()[3])  < self.current_pose[3]-0.1 or \
                    self.controller.fk()[4] > self.current_pose[4]+0.1 or \
                    self.controller.fk()[4] < self.current_pose[4]-0.1 
        if condition:
            # print(self.controller.fk()[0] < self.current_pose[0]-0.15 ,
            #         self.controller.fk()[0] > self.current_pose[0]+0.15 ,
            #         self.controller.fk()[1] < self.current_pose[1]-0.05 ,
            #         self.controller.fk()[1] > self.current_pose[1]+0.6 ,
            #         self.controller.fk()[2] > self.current_pose[2]+0.1 ,
            #         np.abs(self.controller.fk()[3]) > self.current_pose[3]+0.1 ,
            #         np.abs(self.controller.fk()[3])  < self.current_pose[3]-0.1 ,
            #         self.controller.fk()[4] > self.current_pose[4]+0.1 ,
            #         self.controller.fk()[4] < self.current_pose[4]-0.1  )
            reward = -1
            done = True

        self.err = err
        
        if err < 2:
        # if observation[3] * 45 < 2:
            # print("achieved goal ######################")
            reward = 1e+4
            done = True

        info = {}
        return observation, reward, done, info

    def reset(self):

        self.current_step = 0
        self.old_a = 0

        self.data.qpos = self.init_pose.copy()
        self.data.qvel = self.init_qvel.copy()
        self.data.act = self.init_act.copy()
        self.data.qacc_warmstart = self.init_qacc_warmstart.copy()

        self.data.ctrl = self.init_ctrl.copy()
        self.data.qfrc_applied = self.init_qfrc_applied.copy()
        self.data.xfrc_applied = self.init_xfrc_applied.copy()
        self.data.qacc = self.init_qacc.copy()
        self.data.act_dot = self.init_act_dot.copy()

        self.data.time = self.init_time


        self.viewer.make_current()
        self.viewer.render(overlay_on=False)

        # inti esim
        self.viewer.init_esim(contrast_threshold_negative=0.9, contrast_threshold_positive=0.9, refractory_period_ns=100)

        self.err = None

        self.env_reset()

        pose = self.current_pose
        self.controller.set_action(pose)

        self.randomize_hole_position()

        # ======== observation ==========

        observation = self.observe()

        return observation

    def randomize_hole_position(self):
        offset_pos = (np.random.rand(3) - 0.5) * self.max_rand_offset
        offset_pos[2] = 0.0                     # no offset in z

        offset_pos = np.zeros_like(offset_pos)

        self.set_hole_pose(offset_pos)


    def close(self):
        self.viewer.close()


    def change_to_shape(self, a):
        if a.shape == (1, 2):
            return a[0]

        return a

    def render_frame(self):
        return self.viewer.capture_frame(1)

    def get_hole_pose(self):
        # get body offset
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "test_box0")
        pos_offset = self.model.body_pos[body_id]

        return pos_offset

    def set_hole_pose(self, offset_pos):
        obj_names = ["test_box0", "test_box1", "test_box2", "test_box3", "test_box4"]

        # get body offset
        for name in obj_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.model.body_pos[body_id][:3] += offset_pos

    def apply_event_noise(self, img, n, res):

        backgorund_map = img == 127

        # event noise
        a = np.random.uniform(0, 100, res)
        ne = a < n/2.0
        pe = a > 100-n/2.0

        filtered_ne = ne * backgorund_map
        filtered_pe = pe * backgorund_map

        img1 = np.where(filtered_ne, 0, img)
        img2 = np.where(filtered_pe, 255, img1)
        
        return img2

    def apply_event_stream_noise(self, e_stream, e_img, n, res):
        
        backgorund_map = e_img == 127

        e_time_set = set(e_stream[:,2])

        # print("real")
        # print(e_stream)

        for t in e_time_set:


            # event noise
            a = np.random.uniform(0, 100, res)
            ne = a < n/2.0
            pe = a > 100-n/2.0

            filtered_ne = ne * backgorund_map
            filtered_pe = pe * backgorund_map

            ne_hits = np.where(filtered_ne)
            pe_hits = np.where(filtered_pe)

            # print(e_stream)


            for i in range(len(ne_hits[0])):
                e_stream = np.append(e_stream, [[ne_hits[0][i], ne_hits[1][i], t, -1]], axis = 0)

            for i in range(len(pe_hits[0])):
                e_stream = np.append(e_stream, [[pe_hits[0][i], pe_hits[1][i], t, 1]], axis = 0)

        e_stream = e_stream[e_stream[:, 2].argsort()]

        noisy_e_stream = e_stream.copy()

        # print("changed")
        # print(e_stream)

        return noisy_e_stream

