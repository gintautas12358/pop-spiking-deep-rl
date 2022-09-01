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

class PegInHoleRandomEventsVisualServoingGuiding(gym.Env):

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

        self.data.qpos = self.init_pose

        # init hole position
        self.init_hole_pos = self.get_hole_pose()
        offset_pos = (np.random.rand(3) - 0.5) * 0.05
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        self.current_pose = np.array([0.0, 0.43, 0.10, 3.14, 0, 0])
        # self.current_pose = np.array([0.0 ,0.6,0.10, 3.14, 0, 0])  # on top of the hole
        # self.current_pose = np.array([0.0 ,0.6,0.05, 3.14, 0, 0]) # inside the hole
        # self.goal_pose = np.array([0.0, 0.6, 0.02, 3.14, 0, 0])

        self.img = np.ones((32, 32)) * 127

        goal_img_path = "/home/palinauskas/Documents/pop-spiking-deep-rl/gym-env/gym_env/envs/goal_image/00001654.png"
        self.goal_img = self.preprocessing(cv2.flip(cv2.imread(goal_img_path), 0))

        
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
        # self.state_observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(obs,), dtype=np.float32)


        img_shape = 32 , 32
        img_pixels = 32*32
        img_activity_coords = 2
        # self.image_observation_space = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(img_activity_coords,), dtype=np.float32)

        # self.observation_space = gym.spaces.Tuple([self.state_observation_space, 
        #                                         self.image_observation_space])


        self.ac_position_scale = 0.1
        self.ac_orientation_scale = 0.1

        self.ob_position_scale = 10
        self.ob_orientation_scale = 10
        self.ob_linear_force_scale = 1000
        self.ob_rotation_force_scale = 1000
        self.ob_image_scale = 255.0
        self.ob_img_dist_scale = 1000
        

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

        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        if out is not None:
            # print("events created")
            e_img, e = out
            self.img  = self.preprocessing(e_img)

        observation = self.observe(self.img)

        # ======== reward ==========

        dx = np.linalg.norm(action - self.old_a)
        self.old_a = action.copy()
        err = self.dist_metric(self.goal_img, self.img)
        reward = 1/(2*err+0.001)-5*(err+0.001)-5 - 5* dx
        # print(reward)

        # ======== done condition ==========

        boundary_offset = 0.4
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
        offset_pos = (np.random.rand(3) - 0.5) * 0.01
        offset_pos[2] = 0.0                     # no offset in z
        self.set_hole_pose(offset_pos)

        # ======== observation ==========

        observation = self.observe(self.img)

        return observation

    def observe(self, img):

        out = self.get_activity_coord(img)
        x, y = 16, 16
        if out is not None:
            x, y = out

        observation = (x / (32.0 / 2.0) - 1), (y / (32.0 / 2.0) - 1) 

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
        # cv2.imshow("resized image", img)
        # cv2.waitKey(0)

        return img

    def dist_metric(self, goal_img, img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)
        c = np.append(x[:, None], y[:,None], axis=1)
        
        gpx, gpy = np.where(goal_img == 0)
        gnx, gny = np.where(goal_img == 255)
        gx = np.append(gnx, gpx)
        gy = np.append(gny, gpy)
        gc = np.append(gx[:, None], gy[:,None], axis=1)
        
        dist_mat = distance_matrix(c, gc)
        
        self_dist_mat = distance_matrix(gc, gc)
        
        return dist_mat.flatten().sum() - self_dist_mat.flatten().sum()

    def get_activity_coord(self, img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)

        if x.size == 0:
            return None

        x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()

        return x_mean, y_mean
