import mujoco
import mujoco_viewer
import os
import time
import numpy as np
from .controllers.full_impedance_controller import FullImpedanceController
# from utils.read_cfg import get_mjc_xml, get_jposes, get_cposes, get_cerr_lim

from .utils.kinematics import current_ee_position


import gym

class PegInHole(gym.Env):

    def __init__(self, sim_speed=16, headless=False, render_every_frame=False):
        
        xml_path = "/home/palinauskas/Documents/mujoco-eleanor/kuka/envs/assets/full_kuka_INRC3_mounted_camera_hole.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # init first position
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, headless=headless, render_every_frame=render_every_frame)
        self.controller = FullImpedanceController(self.model, self.data) 

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

        self.current_pose = np.array([0.0, 0.63, 0.20, 3.14, 0, 0])
        # self.current_pose = np.array([0.0 ,0.6,0.20, 3.14, 0, 0])  # on top of the hole
        # self.current_pose = np.array([0.0 ,0.6,0.15, 3.14, 0, 0]) # inside the hole
        self.goal_pose = np.array([0.0, 0.6, 0.12, 3.14, 0, 0])

        self.err_limit = 0.0005

        self.max_steps = 1
        self.current_step = 0
        self.old_a = 0

        # gym spaces
        position_ac = 3
        orientation__ac = 3
        acs = position_ac + orientation__ac
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(acs,), dtype=np.float32)

        position_ob = 3
        orientation_ob = 3
        contact_force_ob = 6
        state = position_ob + orientation_ob + contact_force_ob
        # self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(obs,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(state,), dtype=np.float32)


        self.ac_position_scale = 0.1
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 10
        self.ob_orientation_scale = 10
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000

    def step(self, action):
        
        # to check the cartesian position of the initialised joint position
        # print(self.controller.fk())

        # init step gym
        reward = 0
        done = False
        self.current_step += 1

        self.viewer.make_current()
        self.viewer.render()

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

        err = np.linalg.norm(self.goal_pose[:3] - self.controller.fk()[:3])
        f = np.linalg.norm(observation[6:9])
        dx = np.linalg.norm(action - self.old_a)
        self.old_a = action.copy()
        reward = 1/(2*err+0.001)-5*(err+0.001)-5 -0.01 * f - 5* dx

        # ======== done condition ==========
        boundary_offset = 0.1
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


    def observe(self):

        # state
        current_pose = self.controller.fk() 
        current_pose[:3] = current_pose[:3] / self.ob_position_scale
        current_pose[3:] = current_pose[3:] / self.ob_orientation_scale

        forces = self.controller.force_feedback()
        forces[:3] = forces[:3] / self.ob_linear_force_scale
        forces[3:] = forces[3:] / self.ob_rotation_force_scale 


        state_observation = np.append(current_pose, forces)
        

        # observation = [state_observation, img_observation]
        # observation = np.append(state_observation, img_observation.flatten())
        observation = state_observation

        return observation.clip(-1, 1)

    def reset(self):
        self.data.qpos = self.init_pose

        pose = self.current_pose
        self.controller.set_action(pose)

        # ======== observation ==========

        current_pose = self.controller.fk()
        forces = self.controller.force_feedback()

        observation = self.observe()

        return observation

    def close(self):
        self.viewer.close()


    def change_to_shape(self, a):
        if a.shape == (1, 6):
            return a[0]

        return a

    def render_frame(self):
        return self.viewer.capture_frame(0)