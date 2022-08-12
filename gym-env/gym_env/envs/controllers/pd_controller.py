import os

import numpy as np
import gym
import mujoco

import sys
sys.path.append("..")

from utils.mujoco_utils import kuka_subtree_mass, get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices
from .joint_controller import Joint_controller


class PDController(Joint_controller):

    '''
    A Proportional Derivative Controller.
    '''

    def __init__(self,
                 sim_model, sim_data,
                 action_scale=1.,
                 action_limit=1.,
                 controlled_joints=None,
                 kp=3.,
                 kd="auto",
                 set_velocity=False,
                 keep_finite=False,
                 gravity_comp_model_path=None):

        super(PDController, self).__init__(sim_model, sim_data,
                    action_scale,
                    action_limit,
                    controlled_joints,
                    kp,
                    kd,
                    set_velocity,
                    keep_finite)
        
        #TODO is it needed? or there is another way to reuse self.model
        if gravity_comp_model_path is not None:
            self.gravity_comp = True
            kuka_asset_dir="kuka"
            model_path = kuka_asset_dir + gravity_comp_model_path
            print(model_path)

            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.gravity_comp_sim_data = sim_data
            assert self.model.nv == self.sim_model.nv, \
                "the model for control and simulation must have the same number of DOF"
        else: self.gravity_comp =  False

        # correct PD parameters
        dt = self.sim_model.opt.timestep
        self.kd = np.minimum(kp * dt, self.kd) #kp * dt is needed otherwise it explodes

    def set_action(self, action):
        '''
        Sets the setpoints for the PD Controller.
        '''
        action = action * self.action_scale

        nu = len(self.sim_actuators_idx)
        self.sim_qpos_set = action[:nu]

        if self.set_velocity:
            self.sim_qvel_set = action[nu:2 * nu]


    def get_torque(self):
        '''
        Computes the torques from the setpoints and the current state.
        '''

        torque = self.kp * self.joint_error() + self.kd * self.joint_vel_error()

        # Add gravity compensation if necessary
        if self.gravity_comp:
            self.gravity_comp_sim_data.qpos[self.sim_qpos_idx] = self.sim_data.qpos[self.sim_qpos_idx].copy()
            self.gravity_comp_sim_data.qvel[self.sim_qvel_idx] = np.zeros_like(self.sim_qvel_idx)
            self.gravity_comp_sim_data.qacc[self.sim_qvel_idx] = np.zeros_like(self.sim_qvel_idx)
            mujoco.mj_inverse(self.model, self.gravity_comp_sim_data)
            torque += self.gravity_comp_sim_data.qfrc_inverse[self.sim_actuators_idx].copy()

        return torque


class RelativePDController(PDController):
    def set_action(self, action):
        action = action * self.action_scale

        nu = len(self.self_actuators_idx)
        self.qpos_setpoint = action[0:nu] + \
            self.sim_data.qpos[self.sim_qpos_idx]
        if self.set_velocity:
            self.qvel_setpoint = action[nu:2 * nu]


#register_controller(PDController, 'PDController')
#register_controller(RelativePDController, 'RelativePDController')
