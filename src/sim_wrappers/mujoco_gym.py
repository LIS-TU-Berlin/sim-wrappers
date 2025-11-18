# initial version from e05-RobotGym.py (robot learning course)

from .mujoco_sim import *
import gymnasium as gym
import numpy as np
import time
import robotic as ry

###############################################################################

class MujocoGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode = 'human'

    def __init__(self, sim: MjSim, tau_step: float = 0.05, time_limit=1.,
                 feature_map = None, feature_target=None):
        self.sim = sim
        self.x0 = self.sim.getState()
        if self.x0.time > 0.:
            print('WARNING: initial sim has time>0 ... resetting this to 0')
            self.x0.time = 0.

        self.tau_step = tau_step
        self.feature_map = feature_map
        self.feature_target = feature_target
        
        self.time_limit = time_limit

        position_min_max = 1.

        # define the observation space (see also observation_fct())
        self.observation_space = gym.spaces.Box(-2., +2., shape=(self.x0.qpos.size + self.x0.qvel.size,), dtype=np.float32)

        # define the action space
        num_ctrl_pts = 2
        self.action_space = gym.spaces.Box(-position_min_max, +position_min_max, shape=(num_ctrl_pts*self.sim.ctrl_dim,), dtype=np.float32)

    def __del__(self):
        del self.sim

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=int(seed))
            # np.random.seed(seed)
            # ry.rnd_seed(seed)

        self.sim.setState(self.x0)
        self.sim.resetSplineRef(ctrl_time=0.)

        # if self.random_reset:
        #     # resetting the box position to a random initial position -- makes it MUCH harder
        #     self.box_pos0 = np.array([.0,-.1,.7]) + .7 * np.random.rand(3)
        #     self.box_pos0[2]=.7

        observation = self.observation_fct(self.x0)
        info = {"no": "additional info"}
        return observation, info

    def step(self, action):
        action = action.reshape(2, self.sim.ctrl_dim)
        self.sim.setSplineRef(action, np.array([self.tau_step, 2.*self.tau_step]), append=False)
        self.sim.step(tau_step=self.tau_step)
  
        x = self.sim.getState()
        assert x.time == self.sim.ctrl_time, "why not?"

        observation = self.observation_fct(x)
        reward = self.reward_fct(x)
        terminated = False
        truncated = (self.sim.ctrl_time >= self.time_limit) # terminated and truncated difference is super important
        info = {"no": "additional info"}
        return observation, reward, terminated, truncated, info
    
    def observation_fct(self, x: MjSimState):
        return np.concatenate((x.qpos, x.qvel))

    def reward_fct(self, x: MjSimState):
        assert self.feature_map is not None, "rewards w/o feature_map undefined"

        z = self.feature_map(x.qpos, x.qvel)
        phi = z - self.feature_target

        return -np.sum(np.square(phi))
        
    def rollout(self, pi, return_data=None, verbose=1):
        '''helper to play and view a policy'''
        obs, info = self.reset()

        if verbose>0:
            self.sim.C.view(True, 'START')
            self.sim.view_speed=1.
        else:
            self.sim.view_speed=-1.

        t = 0
        R = 0
        if return_data:
            data = {'obs': [], 'action': [], 'next_obs': [], 'reward': [], 'terminal': []}
        while True:
            action = pi(obs, t)
            next_obs, reward, terminated, truncated, info = self.step(action)
            if data is not None:
                data['obs'].append(obs)
                data['action'].append(action)
                data['next_obs'].append(next_obs)
                data['reward'].append(np.array([reward]))
                if terminated or truncated:
                    data['terminal'].append(np.array([1.]))
                else:
                    data['terminal'].append(np.array([0.]))
            obs = next_obs
            R += reward
            t += 1
            if verbose>0:
                print("reward: ", reward)
            if terminated or truncated:
                break

        if verbose>0:
            print('total return:', R)
            self.sim.C.view(True, 'END')
        
        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    def render(self):
        '''also part of the env.Gym'''
        self.C.view(False, f'RoboticGym time {self.time} / {self.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()
