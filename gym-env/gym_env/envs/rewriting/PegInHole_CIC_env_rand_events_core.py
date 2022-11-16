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
import time
from scipy.spatial import distance_matrix

class PegInHoleRandomEventsCore(gym.Env):

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
                                    -0.80,
                                    -1.57])

        # for save for reset later
        self.data.qpos = self.init_pose.copy()
        self.init_qvel = self.data.qvel.copy()

        # init hole position
        self.init_hole_pos = self.get_hole_pose()
        self.max_rand_offset = 0.01
        np.random.seed(int(time.time()))
        self.move_hole_random()

        self.current_pose = np.array([0.0, 0.43, 0.10, 3.14, 0, 0])
        
        # gym spaces
        position_ac = 2
        orientation__ac = 0
        acs = position_ac + orientation__ac
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(acs,), dtype=np.float32)

        position_ob = 2
        orientation_ob = 3
        contact_force_ob = 6
        state = position_ob + orientation_ob + contact_force_ob

        img_err = 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(position_ob + img_err,), dtype=np.float32)

        self.ac_position_scale = 0.2
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 10
        self.ob_orientation_scale = 10
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000
        self.ob_image_scale = 255.0
        self.ob_img_dist_scale = 1000

    def get_pose(self, action):
        pass

    def get_reward(self):
        pass

    def get_observation(self):
        pass

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
        self.controller.set_action(self.get_pose(action))
        torque = self.controller.get_torque()
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

        # ======== observation ==========
        observation = self.get_observation()

        # ======== reward ==========
        reward = self.get_reward()

        # ======== done condition ==========
        condition = self.controller.fk()[0] < self.current_pose[0]-0.15 or \
                    self.controller.fk()[0] > self.current_pose[0]+0.15 or \
                    self.controller.fk()[1] < self.current_pose[1]-0.05 or \
                    self.controller.fk()[1] > self.current_pose[1]+0.6 or \
                    self.controller.fk()[2] > self.current_pose[2]+0.1 

        if condition:
            # print(self.controller.fk()[0] < self.current_pose[0]-boundary_offset,
            #         self.controller.fk()[0] > self.current_pose[0]+boundary_offset,
            #         self.controller.fk()[1] < self.current_pose[1]-0.05,
            #         self.controller.fk()[1] > self.current_pose[1]+boundary_offset,
            #         self.controller.fk()[2] > self.current_pose[2]+0.1)
            done = True

        info = {}
        return observation, reward, done, info

    def env_reset(self):    

        self.data.qpos = self.init_pose.copy()
        self.data.qvel = self.init_qvel.copy()

        pose = self.current_pose.copy()
        self.controller.set_action(pose)

    def reset(self):
        self.env_reset()

        # randomize hole position
        self.move_hole_random()

        # ======== observation ==========

        observation = self.observe(self.img)

        return observation


    def close(self):
        self.viewer.close()


    def change_to_shape(self, a):
        if a.shape == (1, 2):
            return a[0]

        return a

    def render_frame(self):
        return self.viewer.capture_frame(1)

    def move_hole_random(self):
        offset_pos = (np.random.rand(3) - 0.5) * self.max_rand_offset
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

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


