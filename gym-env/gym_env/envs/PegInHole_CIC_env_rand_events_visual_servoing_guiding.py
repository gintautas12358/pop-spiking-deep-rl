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

class PegInHoleRandomEventsVisualServoingGuiding(gym.Env):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True):
        
        xml_path = "/home/palinauskas/Documents/mujoco-eleanor/kuka/envs/assets/full_kuka_INRC3_mounted_camera_hole.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # init first position
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, headless=headless, render_every_frame=render_every_frame, running_events=True)
        self.controller = FullImpedanceController(self.model, self.data) 

        # inti esim
        
        self.viewer.init_esim(contrast_threshold_negative=0.9, contrast_threshold_positive=0.9, refractory_period_ns=100)
        

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

        self.data.qpos = self.init_pose.copy()
        self.init_qvel = self.data.qvel.copy()

        # init hole position
        self.init_hole_pos = self.get_hole_pose()
        self.max_rand_offset = 0.01
        np.random.seed(int(time.time()))
        offset_pos = (np.random.rand(3) - 0.5) * self.max_rand_offset
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        self.current_pose = np.array([0.0, 0.43, 0.10, 3.14, 0, 0])
        # self.current_pose = np.array([0.0 ,0.6,0.10, 3.14, 0, 0])  # on top of the hole
        # self.current_pose = np.array([0.0 ,0.6,0.05, 3.14, 0, 0]) # inside the hole
        # self.goal_pose = np.array([0.0, 0.6, 0.02, 3.14, 0, 0])

        self.img = np.ones((32, 32)) * 127
        self.c = np.array([[32, 32]])

        img_files = ["00001654.png", "00001262.png", "00001886.png", "00002562.png"]
        index = np.random.randint(0,4)
        img_file = img_files[index]
        goal_img_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/goal_image/" + img_file


        goal_img = self.relocate_img(cv2.imread(goal_img_path), (np.random.randint(26,38), np.random.randint(26,38)), 64)

        # self.goal_img = self.preprocessing(cv2.flip(cv2.imread(goal_img_path), 0))
        self.goal_img = self.preprocessing(goal_img)


        
        self.err_limit = 0.0005

        self.max_steps = 1
        self.current_step = 0
        self.old_a = 0

        # gym spaces
        position_ac = 2
        orientation__ac = 0
        acs = position_ac + orientation__ac
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(acs,), dtype=np.float32)

        position_ob = 3
        orientation_ob = 3
        contact_force_ob = 6
        state = position_ob + orientation_ob + contact_force_ob
        # self.state_observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(obs,), dtype=np.float32)


        img_shape = 32 , 32
        img_pixels = 32*32
        img_activity_coords = 2
        img_err = 1
        # self.image_observation_space = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(position_ob + img_err,), dtype=np.float32)

        # self.observation_space = gym.spaces.Tuple([self.state_observation_space, 
        #                                         self.image_observation_space])


        self.ac_position_scale = 0.3
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 10
        self.ob_orientation_scale = 10
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000
        self.ob_image_scale = 255.0
        self.ob_img_dist_scale = 1000
        

    def step(self, action):
        
        # print(self.controller.fk() )

        # print("hole pos", self.get_hole_pose())
        
        # to check the cartesian position of the initialised joint position
        # print(self.controller.fk())

        # init step gym
        reward = 0
        done = False
        self.current_step += 1

        self.viewer.make_current()
        self.viewer.render(overlay_on=False)

        # ======== apply action ==========
        action = self.change_to_shape(action)

        # print(action)
        pose = self.current_pose.copy()
        # print(pose)
        ac_position = action[:2]
        pose[:2] +=  ac_position * self.ac_position_scale 
        # print(pose)
        # ac_orientation = action[3:]
        # pose[3:] += ac_orientation * self.ac_orientation_scale
        # print(pose)
        self.controller.set_action(pose)
        torque = self.controller.get_torque()
        self.data.ctrl[:] = torque
        
        mujoco.mj_step(self.model, self.data)


        # ======== observation ==========

        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        if out is not None:
            # print("events created")
            e_img, e, num_e = out
            self.img  = self.preprocessing(e_img)

        observation = self.observe(self.img)

        # ======== reward ==========

        dx = np.linalg.norm(action - self.old_a)
        self.old_a = action.copy()
        err = self.dist_metric(self.goal_img, self.img)

        # reward = 1/(2*err+0.001)-5*(err+0.001)-5 - 5* dx

        # reward = 0.1/(0.00001*err+0.001) - 100* dx
        reward = 1/(0.01*err+0.01) - 1* dx

        # print(reward)

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
            #         np.abs(self.controller.fk()[3])  > self.current_pose[3]+0.1 ,
            #         np.abs(self.controller.fk()[3])  < self.current_pose[3]-0.1 ,
            #         self.controller.fk()[4] > self.current_pose[4]+0.1 ,
            #         self.controller.fk()[4] < self.current_pose[4]-0.1 )
            done = True

        info = {}
        return observation, reward, done, info

    def env_reset(self):    

        self.data.qpos = self.init_pose.copy()
        self.data.qvel = self.init_qvel.copy()

        self.img = np.ones((32, 32)) * 127

        # img_files = ["00001654.png", "00001262.png", "00001886.png", "00002562.png"]
        img_files = ["00001654.png", "00001262.png", "00002562.png"]

        index = np.random.randint(0,4)
        img_file = img_files[index]
        goal_img_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/goal_image/" + img_file


        goal_img = self.relocate_img(cv2.imread(goal_img_path), (np.random.randint(26,38), np.random.randint(26,38)), 64)

        # self.goal_img = self.preprocessing(cv2.flip(cv2.imread(goal_img_path), 0))
        self.goal_img = self.preprocessing(goal_img)

    def reset(self):
        self.env_reset()

        pose = self.current_pose.copy()
        self.controller.set_action(pose)

        # randomize hole position
        
        offset_pos = (np.random.rand(3) - 0.5) * self.max_rand_offset
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        # ======== observation ==========

        observation = self.observe(self.img)

        return observation

    def observe(self, img):

        err = self.dist_metric(self.goal_img, self.img)

        pose = self.controller.fk()

        observation = (pose[0] - self.current_pose[0]) / (self.ac_position_scale ), \
                      (pose[1] - self.current_pose[1]) / (self.ac_position_scale ), \
                      (pose[2] - self.current_pose[2]) / (self.ac_position_scale ), \
                      err / 45.0

        return observation

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



        # mat_offset = self.model.body_quat[body_id]
        # quat_offset = mat2Quat(np.array(mat_offset))

        # get end-effector pose
        # pos, quat = self.controller._fk()

        # get camera pose
        # cam_pos = pos + pos_offset
        # cam_quat = quatAdd(quat, quat2Vel(quat_offset))

        return pos_offset

    def set_hole_pose(self, offset_pos):
        obj_names = ["test_box0", "test_box1", "test_box2", "test_box3", "test_box4"]

        # get body offset
        for name in obj_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.model.body_pos[body_id][:3] += offset_pos

    def preprocessing(self, img):

        img = cv2.flip(img, 0)

        # size crop

        # res = 512
        # half_size = int(res / 2)

        # H, W, _ = img.shape
        # mid_x = int(W / 2)
        # img = img[H - res:H, mid_x-half_size:mid_x+half_size]

        img = np.where(img == 50, 255, img)

                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # max_value = np.max(img)
        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        # maxpool

        # for i in range(4):
        #     img = skimage.measure.block_reduce(img, (2,2), np.max)
        
        for i in range(1):
            img = skimage.measure.block_reduce(img, (2,2), np.max)


        # gray background

        img = np.where(img == 0, 127, img)
        img = np.where(img == 129, 0, img)


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
        new_img[crop_y0 + x_offset:crop_y1 + x_offset, crop_x0 + y_offset:crop_x1 + y_offset, :] = img[:,:,:]


        return new_img.astype(np.uint8)
