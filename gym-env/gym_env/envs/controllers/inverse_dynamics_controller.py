import os
import numpy as np
from gym import spaces
import mujoco

import sys
sys.path.append("..")

from .joint_controller import Joint_controller
from utils.mujoco_utils import kuka_subtree_mass, get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices 


class InverseDynamicsController(Joint_controller):
    '''
    An inverse dynamics controller that used PD gains to compute a desired acceleration.
    '''

    def __init__(self,
                 sim_model, sim_data,
                #  model_path='full_kuka_no_collision_no_gravity.xml',
                 action_scale=1.0,
                 action_limit=1.0,
                 kp=1.0,
                 kd='auto',
                 controlled_joints=None,
                 set_velocity=False,
                 keep_finite=False):
        super(InverseDynamicsController, self).__init__(sim_model, sim_data,
                    action_scale,
                    action_limit,
                    controlled_joints,
                    kp,
                    kd,
                    set_velocity,
                    keep_finite)


    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        nq = len(self.sim_qpos_idx)
        nv = len(self.sim_qvel_idx)

        self.sim_qpos_set = self.action_scale * action[:nq]
        if self.set_velocity:
            self.sim_qvel_set = self.action_scale * action[nq:nv]

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''
        self.sim_data.qacc = self.kp * self.joint_error() + self.kd * self.joint_vel_error()
        mujoco.mj_inverse(self.sim_model, self.sim_data)
        id_torque = self.sim_data.qfrc_inverse[self.sim_actuators_idx].copy()

        # Sum the torques.
        return id_torque


class RelativeInverseDynamicsController(InverseDynamicsController):
    def set_action(self, action):
        nq = len(self.sim_qpos_idx)
        nv = len(self.sim_qvel_idx)

        # Set the setpoint difference from the current position.
        self.sim_qpos_set = self.sim_data.qpos[self.sim_qpos_idx] + self.action_scale * action[:nq]
        if self.set_velocity:
            self.sim_qvel_set = self.action_scale * action[nq:nv]
