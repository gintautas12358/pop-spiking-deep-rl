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

class PegInHoleRandomEventsDepthActivityCenter(gym.Env):

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
                                    -0.95,
                                    0,
                                    1.4,
                                    0.0,
                                    -0.75,
                                    0])

        self.data.qpos = self.init_pose

        # init hole position
        self.init_hole_pos = self.get_hole_pose()
        offset_pos = (np.random.rand(3) - 0.5) * 0.05
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        self.current_pose = np.array([0.0, 0.63, 0.10, 3.14, 0, 0])
        # self.current_pose = np.array([0.0 ,0.6,0.20, 3.14, 0, 0])  # on top of the hole
        # self.current_pose = np.array([0.0 ,0.6,0.15, 3.14, 0, 0]) # inside the hole
        # self.goal_pose = np.array([0.0, 0.6, 0.12, 3.14, 0, 0])

        self.last_ac = [0, 0]
        self.old_a = 0

        self.err_limit = 0.0005

        self.max_steps = 1
        self.current_step = 0

        # gym spaces
        position_ac = 3
        orientation__ac = 3
        acs = position_ac + orientation__ac
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(acs,), dtype=np.float32)

        position_ob = 3
        orientation_ob = 3
        contact_force_ob = 6
        state = position_ob + orientation_ob + contact_force_ob
        # self.state_observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(obs,), dtype=np.float32)


        img_shape = 32 , 32
        img_pixels = 32*32
        activity_coords = 2
        # self.image_observation_space = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(state + activity_coords,), dtype=np.float32)

        # self.observation_space = gym.spaces.Tuple([self.state_observation_space, 
        #                                         self.image_observation_space])


        self.ac_position_scale = 0.1
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 10
        self.ob_orientation_scale = 10
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000
        self.ob_image_scale = 255.0
        self.ob_activity_coords_scale = 32

        # for i in range(5):
        #     self.viewer.make_current()
        #     self.viewer.render(overlay_on=False)
        #     time.sleep(1)

    def step(self, action):
        

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
        ac_position = action[:3]
        pose[:3] +=  ac_position * self.ac_position_scale 
        # print(pose)
        ac_orientation = action[3:]
        pose[3:] += ac_orientation * self.ac_orientation_scale
        # print(pose)
        self.controller.set_action(pose)
        torque = self.controller.get_torque()
        self.data.ctrl[:] = torque
        
        mujoco.mj_step(self.model, self.data)

        # ======== observation ==========

        observation = self.observe()

        # ======== reward ==========

        # err = np.linalg.norm(self.goal_pose[:3] - self.controller.fk()[:3])
        # reward = 1/(2*err+0.001)-5*(err+0.001)-5

        """
        # When above the hole (depth = 0.2) the average score is then:
        # -50 * (0.2 - 0.12) ** 2 + 1 = 0.68
        # So if the agent did not entered the hole the reward should be <680
        # """
        # depth = self.controller.fk()[2]
        # reward = -50 * (depth - 0.12) ** 2 + 1
        # if reward < 0:
        #     reward = 0


        depth = self.controller.fk()[2]


        # w2 = 10
        # depth_error = depth - 0.12

        # reward =  w2*(depth_error)


        err = abs(0.02 - depth )
        # reward = 1/(2*err+0.001)-5*(err+0.001)-5


        # err = np.linalg.norm(self.goal_pose[:3] - self.controller.fk()[:3])
        f = np.linalg.norm(observation[6:9])
        dx = np.linalg.norm(action - self.old_a)
        self.old_a = action.copy()
        reward = 1/(2*err+0.001)-5*(err+0.001)-5 -0.01 * f - 5* dx

        # print(reward)

        # ======== done condition ==========

        boundary_offset = 0.15
        condition = self.controller.fk()[0] < 0-boundary_offset or \
                    self.controller.fk()[0] > 0+boundary_offset or \
                    self.controller.fk()[1] < 0.6-boundary_offset or \
                    self.controller.fk()[1] > 0.6+boundary_offset or \
                    self.controller.fk()[2] > 0.20+boundary_offset
        if condition:
            # reward = -10
            done = True


        info = {}
        return observation, reward, done, info

    def reset(self):
        self.data.qpos = self.init_pose

        pose = self.current_pose
        self.controller.set_action(pose)

        # randomize hole position
        offset_pos = (np.random.rand(3) - 0.5) * 0.05
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        # ======== observation ==========

        observation = self.observe()

        return observation

    def observe(self):

        # state
        current_pose = self.controller.fk() 
        current_pose[:3] = current_pose[:3] / self.ob_position_scale
        current_pose[3:] = current_pose[3:] / self.ob_orientation_scale

        forces = self.controller.force_feedback()
        forces[:3] = forces[:3] / self.ob_linear_force_scale
        forces[3:] = forces[3:] / self.ob_rotation_force_scale 

        state_observation = np.append(current_pose, forces)

        
        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        img_observation = np.ones((32, 32)) * 127
        if out is not None:
            # print("events created")
            e_img, e = out
            img_observation = self.preprocessing(e_img)

        # img_observation = img_observation / (self.ob_image_scale / 2.0) - 1


        

        activity_coords = self.last_ac
        out2 = self.get_activity_coord(img_observation)
        if out is not None:
            activity_coords = np.array(out2)
            self.last_ac = (activity_coords / (self.ob_activity_coords_scale / 2.0) ) - 1

        # observation = [state_observation, img_observation]
        observation = np.append(state_observation, activity_coords)

        return observation.clip(-1, 1)

    def close(self):
        self.viewer.close()


    def change_to_shape(self, a):
        if a.shape == (1, 6):
            return a[0]

        return a

    def render_frame(self):
        return self.viewer.capture_frame(0)

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
        # cv2.imshow("cropped", img)
        # cv2.waitKey(0)

        return img

    def get_activity_coord(self, img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)

        if x.size == 0:
            return None

        x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()

        return x_mean, y_mean

    def get_activity_coord_robust(img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)

        x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()

        
        w = 1
        new_x, = np.where((x < x_mean + w * np.sqrt(x_var)) & (x > x_mean - w * np.sqrt(x_var)))
        new_y, = np.where((y < y_mean + w * np.sqrt(y_var)) & (y > y_mean - w * np.sqrt(y_var)))

        x_mean, y_mean, x_var, y_var = new_x.mean(), new_y.mean(), new_x.var(), new_y.var()
        print(x_mean, y_mean, x_var, y_var)

        return x_mean, y_mean, x_var, y_var



