from typing import Generator
from .base_controller import BaseController
from utils.mujoco_utils import kuka_subtree_mass, get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices
import numpy as np
from gym import spaces
import mujoco
from scipy.spatial.transform import Rotation as R


from utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat
from utils.kinematics import forwardKinSite, forwardKinJacobianSite


class JointPDController(BaseController):
    '''
    A base for all joint based controllers
    '''

    def __init__(self,
                    sim_model, sim_data,
                    action_scale=1.,
                    action_limit=1.,
                    site_name='ee_site',
                    controlled_joints=None,
                    kp=3.,
                    kd="auto",
                    set_velocity=False,
                    keep_finite=False):

        super(JointPDController, self).__init__(sim_model, sim_data)

        self.set_velocity = set_velocity
        self.site_name = site_name

        # Get the position, velocity, and actuator indices for the model.
        self.init_indices(controlled_joints)

        # gym
        self.gym_action_space(action_limit, action_scale, keep_finite)
        
        # PD parameters
        self.set_gains(kp, kd)

        # Initialize setpoint.
        self.sim_qpos_set = sim_data.qpos[self.sim_qpos_idx].copy()
        self.sim_qvel_set = np.zeros(len(self.sim_qvel_idx))
    
        
    def init_indices(self, controlled_joints): 
        if controlled_joints is not None:
            self.sim_qpos_idx = get_qpos_indices(self.sim_model, controlled_joints)
            self.sim_qvel_idx = get_qvel_indices(self.sim_model, controlled_joints)
            self.sim_actuators_idx = get_actuator_indices(self.sim_model, controlled_joints)
            self.sim_joint_idx = get_joint_indices(self.sim_model, controlled_joints)
        else:
            self.sim_qpos_idx = range(self.sim_model.nq)
            self.sim_qvel_idx = range(self.sim_model.nv)
            self.sim_actuators_idx = range(self.sim_model.nu)
            self.sim_joint_idx = range(self.sim_model.nu)

    def gym_action_space(self, action_limit, action_scale, keep_finite):
        self.action_scale = action_scale

        low = self.sim_model.jnt_range[self.sim_joint_idx, 0]
        high = self.sim_model.jnt_range[self.sim_joint_idx, 1]

        low[self.sim_model.jnt_limited[self.sim_joint_idx] == 0] = -np.inf
        high[self.sim_model.jnt_limited[self.sim_joint_idx] == 0] = np.inf
        
        if keep_finite:
            # Don't allow infinite bounds (necessary for SAC)
            low[not np.isfinite(low)] = -3.
            high[not np.isfinite(high)] = 3.

        low = low*action_limit
        high = high*action_limit

        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def set_gains(self, kp, kd):
        
        self.kp = kp
        if kd == 'auto':
            # calc kd for critically damped 
            # kd = 2 * sqrt(kp * m)
            mass = kuka_subtree_mass(self.sim_model)
            self.kd = 2 * np.sqrt(mass * kp)

        else:
            self.kd = kd

    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        # self.scale = 1
        # action = action * self.scale

        # dx = action[0:3].astype(np.float64)
        # dr = action[3:6].astype(np.float64)

        # self.pos_set = dx
        # self.quat_set = quatAdd(np.array([1., 0., 0., 0.]), dr)
        self.sim_qpos_set = action

    def get_torque(self):
        '''
        Update the PD setpoint and compute the torque.
        '''

        # calculate errors
        jerr = self.joint_error()
        djerr = self.joint_vel_error()

        # PD law
        torque = self.kp * jerr + self.kd * djerr

        # gravity compensation
        G = self.sim_data.qfrc_bias

        # Sum the torques.
        out_torque = torque + G
        self.sim_data.ctrl = out_torque

        pos, mat = forwardKinSite(self.sim_model, self.sim_data, self.site_name, recompute=False)
        quat = mat2Quat(mat)
        # eul = R.from_matrix(mat.reshape(3,3)).as_euler('xyz')
        
        # print("###Position###")
        # print(f"sim_qpos_set: {self.sim_qpos_set}")
        # print(f"qpos: {self.sim_data.qpos[self.sim_qpos_idx]}")
        # print(f"xpos: {pos} quat: {quat}")
        # print("###Velocity###")
        # print(f"sim_qvel_set: {self.sim_qvel_set}")
        # print(f"qvel: {self.sim_data.qvel[self.sim_qvel_idx]}")


        print("###Errors###")
        print(f"jerr: {jerr}")
        # print(f"djerr: {djerr}")
        # print(f"kp: {self.kp}")
        # print(f"kd: {self.kd}")

        # print("###Forces###")
        # print(f"qfrc_bias: {self.sim_data.qfrc_bias}")
        

        # print("###Output###")
        # print(f"G: {G}")
        # print(f"torque: {torque}")
        # print(f"out_torque: {out_torque}")

        
        return torque

        

    def joint_error(self):
        return self.sim_qpos_set - self.sim_data.qpos[self.sim_qpos_idx]

    def joint_vel_error(self):
        return self.sim_qvel_set - self.sim_data.qvel[self.sim_qvel_idx]

    def pose_error(self):
        # Compute the pose difference.
        pos, mat = forwardKinSite(self.sim_model, self.sim_data, self.site_name, recompute=False)
        quat = mat2Quat(mat)
        
        dx = self.pos_set - pos
        dr = subQuat(self.quat_set, quat) # Original
        dframe = np.concatenate((dx,dr))
        return dframe