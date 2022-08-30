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

class PegInHoleRandomEvents(gym.Env):

    def __init__(self, sim_speed=16, headless=False, render_every_frame=True):
        
        xml_path = "/home/palinauskas/Documents/mujoco-eleanor/kuka/envs/assets/full_kuka_INRC3_mounted_camera_hole.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # init first position
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, headless=headless, render_every_frame=render_every_frame)
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


        self.current_pose = np.array([0.0, 0.63, 0.10, 3.14, 0, 0])
        # self.current_pose = np.array([0.0 ,0.6,0.10, 3.14, 0, 0])  # on top of the hole
        # self.current_pose = np.array([0.0 ,0.6,0.05, 3.14, 0, 0]) # inside the hole
        # self.goal_pose = np.array([0.0, 0.6, 0.02, 3.14, 0, 0])

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
        obs = position_ob + orientation_ob + contact_force_ob
        self.state_observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(obs,), dtype=np.float32)


        img_shape = 32, 32, 1
        self.image_observation_space = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)

        self.observation_space = gym.spaces.Tuple([self.state_observation_space, 
                                                self.image_observation_space])


        self.ac_position_scale = 0.1
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 0.1
        self.ob_orientation_scale = 0.001
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000

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

        # state
        current_pose = self.controller.fk()
        forces = self.controller.force_feedback()

        state_observation = current_pose[0], current_pose[1], current_pose[2], \
                    current_pose[3], current_pose[4], current_pose[5], \
                    forces[0], forces[1], forces[2], \
                    forces[3], forces[4], forces[5]

        
        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        img_observation = np.zeros((32, 32, 1))
        if out is not None:
            e_img, e = out
            img_observation = self.preprocessing(e_img)

        observation = [state_observation, img_observation]

        # ======== reward ==========

        err = np.linalg.norm(self.goal_pose[:3] - self.controller.fk()[:3])
        reward = 1/(2*err+0.001)-5*(err+0.001)-5

        # ======== done condition ==========

        condition = self.controller.fk()[0] < 0-0.1 or self.controller.fk()[0] > 0+0.1 or self.controller.fk()[1] < 0.6-0.1 or self.controller.fk()[1] > 0.6+0.1 or self.controller.fk()[2] > 0.20+0.1
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
        offset_pos = np.random.rand(3) * 0.03
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        # ======== observation ==========

        current_pose = self.controller.fk()
        forces = self.controller.force_feedback()

        observation = current_pose[0], current_pose[1], current_pose[2], \
                    current_pose[3], current_pose[4], current_pose[5], \
                    forces[0], forces[1], forces[2], \
                    forces[3], forces[4], forces[5]

        return observation

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

        res = 512
        half_size = int(res / 2)

        H, W, _ = img.shape
        mid_x = int(W / 2)
        img = img[H - res:H, mid_x-half_size:mid_x+half_size]

        img = np.where(img == 50, 255, img)

                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        max_value = np.max(img)
        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        # maxpool

        for i in range(4):
            img = skimage.measure.block_reduce(img, (2,2), np.max)


        # gray background

        img = np.where(img == 0, 127, img)
        img = np.where(img == 129, 0, img)


        # observe result. (debug camera view) 
        # cv2.imshow("cropped", img)
        # cv2.waitKey(0)

        return img
