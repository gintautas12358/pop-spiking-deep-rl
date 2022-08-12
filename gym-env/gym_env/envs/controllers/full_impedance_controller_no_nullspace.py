import os

import numpy as np
from gym import spaces
import mujoco

import sys
sys.path.append("..")

#from gym_kuka_mujoco.envs.assets import kuka_asset_dir
from utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat
from utils.kinematics import forwardKinSite, forwardKinJacobianSite
from .base_controller import BaseController
#from . import register_controller
from utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices 


class FullImpedanceController(BaseController):
    '''
    An inverse dynamics controller that used PD gains to compute a desired acceleration.
    '''

    def __init__(self,
                 sim_model, sim_data,
                 ## sim,
                 pos_scale=1.0,
                 rot_scale=1.0,
                 pos_limit=1.0,
                 rot_limit=1.0,
                 model_path='full_kuka_no_collision_no_gravity.xml',
                 site_name='ee_site',
                 stiffness=None,
                 damping='auto',
                 null_space_damping=10,
                 null_space_stiffness=1.0,
                 controlled_joints=None,
                 nominal_pos=None,
                 nominal_quat=None,
                 nominal_qpos=None):
        super(FullImpedanceController, self).__init__(sim_model, sim_data)

        mujoco.mj_forward(sim_model, sim_data)

        # Set the zero position and quaternion of the action space.
        if nominal_pos is None:
            self.nominal_pos = np.array([0.,0.,0.]) #TODO: use a real position
        else:
            self.nominal_pos = nominal_pos.copy()

        if nominal_quat is None:
            self.nominal_quat = np.array([1., 0., 0., 0.]) # TODO: use a real quaternion
        else:
            self.nominal_quat = nominal_quat.copy()


        if nominal_qpos is None:
            self.nominal_qpos = np.zeros(7) # TODO: use a real pose
        else:
            self.nominal_qpos = nominal_qpos.copy()

        self.gym_action_space(pos_limit, rot_limit)

        # Controller parameters.
        self.scale = np.ones(6)
        self.scale[:3] *= pos_scale
        self.scale[3:6] *= rot_scale

        self.site_name = site_name
        self.pos_set = self.nominal_pos.copy()
        self.quat_set = self.nominal_quat.copy()

        # Default stiffness and damping.
        if stiffness is None:
            #self.stiffness = np.array([10000.0, 10000.0, 10000.0, 10000.3, 10000.3, 10000.3])
            self.stiffness = np.array([300.0, 300.0, 300.0, 200.0, 200.0, 200.0])
            #self.stiffness = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        else:
            self.stiffness = np.ones(6)*stiffness

        if damping=='auto':
            self.damping = 2*np.sqrt(self.stiffness)
        else:
            self.damping = 2*np.sqrt(self.stiffness)*damping

        self.null_space_damping = null_space_damping
        self.null_space_stiffness = null_space_stiffness

        self.init_indices(controlled_joints)

    def set_action(self, action):
        '''
        Set the setpoint.
        '''
        action = action * self.scale

        dx = action[0:3].astype(np.float64)
        dr = action[3:6].astype(np.float64)

        self.pos_set = self.nominal_pos + dx
        self.quat_set = quatAdd(self.nominal_quat, dr)
        
        print("POS SET: "+str(self.pos_set))
        print("QUAT SET: "+str(self.quat_set))

    def get_torque(self):
        '''
        Update the impedance control setpoint and compute the torque.
        '''
        self.sim_data.qacc = self.impedance_controller()

        mujoco.mj_inverse(self.sim_model, self.sim_data)
        id_torque = self.sim_data.qfrc_inverse[self.sim_actuators_idx].copy()
        
        print("GET TORQUE: "+str(id_torque))

        return id_torque

    def pose_error(self):
        # Compute the pose difference.
        pos, mat = forwardKinSite(self.sim_model, self.sim_data, self.site_name, recompute=False)
        quat = mat2Quat(mat)
        
        dx = self.pos_set - pos
        dr = subQuat(self.quat_set, quat) # Original
        dframe = np.concatenate((dx,dr))
        return dframe

    def Jac(self):
        jpos, jrot = forwardKinJacobianSite(self.sim_model, self.sim_data, self.site_name, recompute=False)
        J = np.vstack((jpos, jrot)) # full jacobian
        return J

    def right_pseudo_Jac(self, eps=0):
        J = self.Jac()
        
        pJ = J.T @ np.linalg.inv(J @ J.T + eps*np.eye(6)) 
        return pJ

    def left_pseudo_Jac(self, eps=0):
        J = self.Jac()
        
        pJ = np.linalg.inv(J.T @ J + eps*np.eye(7)) @ J.T
        return pJ

    def null_space_proj_m(self):
        J = self.Jac()

        # p = 1 - J^T * (J^+)^T
        projection_matrix = np.eye(7) - J.T @ self.left_pseudo_Jac(eps=1e-6).T

        return projection_matrix

    def impedance_controller(self):
        # desired behaviour 
        J = self.Jac()
        cartesian_acc_des = self.stiffness*self.pose_error() - self.damping * (J @ self.sim_data.qvel[self.sim_qvel_idx])

        # impedance control
        impedance_control = self.right_pseudo_Jac(eps=1e-6) @ cartesian_acc_des
        return impedance_control

    def null_space_controller(self):
        # ns = kn * (qn - q) -  dn * dq
        null_space_control = self.null_space_stiffness * (self.nominal_qpos - self.sim_data.qpos[self.sim_qpos_idx]) - self.null_space_damping*self.sim_data.qvel[self.sim_qvel_idx]
        return null_space_control

    def gym_action_space(self, pos_limit, rot_limit):
        # Construct the action space.
        high_pos = pos_limit*np.ones(3)
        low_pos = -high_pos
        high_rot = rot_limit*np.ones(3)
        low_rot = -high_rot

        high = np.concatenate((high_pos, high_rot))
        low = np.concatenate((low_pos, low_rot))
        self.action_space = spaces.Box(low, high, dtype=np.float32)


    def init_indices(self, controlled_joints): 
            # Get the position, velocity, and actuator indices for the model.
        if controlled_joints is not None:
            self.sim_qpos_idx = get_qpos_indices(self.sim_model, controlled_joints)
            self.sim_qvel_idx = get_qvel_indices(self.sim_model, controlled_joints)
            self.sim_actuators_idx = get_actuator_indices(self.sim_model, controlled_joints)
            self.sim_joint_idx = get_joint_indices(self.sim_model, controlled_joints)
        else:
            assert self.sim_model.nv == self.sim_model.nu, "if the number of degrees of freedom is different than the number of actuators you must specify the controlled_joints"
            self.sim_qpos_idx = range(self.sim_model.nq)
            self.sim_qvel_idx = range(self.sim_model.nv)
            self.sim_actuators_idx = range(self.sim_model.nu)
            self.sim_joint_idx = range(self.sim_model.nu)