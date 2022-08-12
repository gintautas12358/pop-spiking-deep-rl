import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

identity_quat = np.array([1., 0., 0., 0.])

def mat2Quat(mat):
    '''
    Convenience function for mju_mat2Quat.
    '''
    res = np.zeros(4)
    mujoco.mju_mat2Quat(res, mat.flatten())
    return res

def quat2Mat(quat):
    '''
    Convenience function for mju_quat2Mat.
    '''
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, quat)
    res = res.reshape(3,3)
    return res

def quat2Vel(quat):
    '''
    Convenience function for mju_quat2Vel.
    '''
    res = np.zeros(3)
    mujoco.mju_quat2Vel(res, quat, 1.)
    return res

def axisAngle2Quat(axis, angle):
    '''
    Convenience function for mju_quat2Vel.
    '''
    res = np.zeros(4)
    mujoco.mju_axisAngle2Quat(res, axis, angle)
    return res

def subQuat(qb, qa):
    '''
    Convenience function for mju_subQuat.
    '''
    # Allocate memory
    qa_t = np.zeros(4)
    q_diff = np.zeros(4)
    res = np.zeros(3)

    # Compute the subtraction
    mujoco.mju_negQuat(qa_t, qa)
    mujoco.mju_mulQuat(q_diff, qb, qa_t)
    mujoco.mju_quat2Vel(res, q_diff, 1.)

    return res

def mulQuat(qa, qb):
    res = np.zeros(4)
    mujoco.mju_mulQuat(res, qa, qb)
    return res

def random_quat():
    q = np.random.random(4)
    q = q/np.linalg.norm(q)
    return q

def quatIntegrate(q, v, dt=1.):
    res = q.copy()
    mujoco.mju_quatIntegrate(res,v,1.)
    return res

def quatAdd(q1, v):
    qv = quatIntegrate(identity_quat, v)
    res = mulQuat(qv, q1)
    return res

def rotVecQuat(v, q):
    res = np.zeros(3)
    mujoco.mju_rotVecQuat(res, v, q)
    return res

def quat2eul(q):
    mat = quat2Mat(q)
    q = R.from_matrix(mat.reshape(3,3)).as_euler('xyz')
    return q

    
