import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from utils import *
import os

class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        def _get_obs_per_limb(b):
            if 'hip' in b:
                limb_type_vec = np.array((1, 0, 0, 0))
            elif 'knee' in b:
                limb_type_vec = np.array((0, 1, 0, 0))
            elif 'shoulder' in b:
                limb_type_vec = np.array((0, 0, 1, 0))
            elif 'elbow' in b:
                limb_type_vec = np.array((0, 0, 0, 1))
            else:
                limb_type_vec = np.array((0, 0, 0, 0))
            torso_x_pos = self.data.get_body_xpos('torso')[0]
            xpos = self.data.get_body_xpos(b)
            xpos[0] -= torso_x_pos
            q = self.data.get_body_xquat(b)
            expmap = quat2expmap(q)
            obs = np.concatenate([xpos, np.clip(self.data.get_body_xvelp(b), -10, 10), \
                self.data.get_body_xvelr(b), expmap, limb_type_vec])
            # include current joint angle and joint range as input
            if b == 'torso':
                angle = 0.
                joint_range = [0., 0.]
            else:

                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                qpos_adr = self.sim.model.jnt_qposadr[jnt_adr] # assuming each body has only one joint
                angle = np.degrees(self.data.qpos[qpos_adr]) # angle of current joint, scalar
                joint_range = np.degrees(self.sim.model.jnt_range[jnt_adr]) # range of current joint, (2,)
                # normalize
                angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                joint_range[0] = (180. + joint_range[0]) / 360.
                joint_range[1] = (180. + joint_range[1]) / 360.
            obs = np.concatenate([obs, [angle], joint_range])
            return obs

        full_obs = np.concatenate([_get_obs_per_limb(b) for b in self.model.body_names[1:]])
        return full_obs.ravel()

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        # Forward reward
        reward = ((posafter - posbefore) / self.dt)
        # Control penalty
        reward -= 1e-3 * np.square(a).sum()
        # Energy penalty
        reward -= 1e-3 * np.abs(a).sum()
        # Alive bonus
        reward += 1.0
        qpos = self.sim.data.qpos
        done = not (height > 0.4 and height < 2.1 and
                    ang > -1.0 and ang < 1.0)
        # If done, penalize
        if done:
            reward -= 10.0
        return self._get_obs(), reward, done, {}

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
