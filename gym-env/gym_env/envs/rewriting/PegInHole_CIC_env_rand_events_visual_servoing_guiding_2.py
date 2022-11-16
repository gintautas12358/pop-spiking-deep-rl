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

from .PegInHole_CIC_env_rand_events_core import PegInHoleRandomEventsCore

class PegInHoleRandomEventsVisualServoingGuiding2(PegInHoleRandomEventsCore):

    def __init__(self, sim_speed=32, headless=False, render_every_frame=True):
        
        self.observation_space = self.__class__.observation_space
        self.action_space = self.__class__.action_space
        
        self.img = np.ones((32, 32)) * 127
        self.dist = 60

        goal_img_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/goal_image/00001654.png"
        self.goal_img = self.preprocessing(cv2.flip(cv2.imread(goal_img_path), 0))

    def get_pose(self, action):
        action = self.change_to_shape(action)

        pose = self.current_pose.copy()
        ac_position = action[:2]
        pose[:2] +=  ac_position * self.ac_position_scale 
        return pose

    def get_reward(self):
        err = self.dist_metric(self.goal_img, self.img)
        reward = 1/(0.01*err+0.01) 

        return reward

    def get_observation(self):
        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        if out is not None:
            # print("events created")
            e_img, e, num_e = out
            self.img  = self.preprocessing(e_img)

        err = self.dist_metric(self.goal_img, self.img)
        pose = self.controller.fk()
        observation = (pose[0] - self.current_pose[0]) / (self.ac_position_scale ), \
                      (pose[1] - self.current_pose[1]) / (self.ac_position_scale ), \
                      err / 45.0

        return observation


    def preprocessing(self, img):

        img = cv2.flip(img, 0)

        img = np.where(img == 50, 255, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        # maxpool
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
            return -10.0
        
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



