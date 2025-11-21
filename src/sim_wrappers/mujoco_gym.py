# initial version from e05-RobotGym.py (robot learning course)

from .mujoco_sim import *
import gymnasium as gym
import numpy as np
import robotic as ry
import mujoco
import data_tools
from gymnasium.spaces import Box, Dict
from gymnasium_robotics.core import GoalEnv
from numpy._typing import NDArray
from typing import Any
import random

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
        action = action.reshape(1, self.sim.ctrl_dim)
        self.sim.setSplineRef(action, np.array([2.*self.tau_step]), append=False)
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
        
    def rollout(self, pi, return_data=False, verbose=1):
        '''helper to play and view a policy'''
        obs, info = self.reset()

        if verbose>0:
            self.sim.C.view(True, 'START')
            self.sim.view_speed=1.
        else:
            self.sim.view_speed=-1.

        if return_data:
            data = {'state': [], 'obs': [], 'action': [], 'next_obs': [], 'reward': [], 'terminal': []}

        t = 0
        R = 0
        while True:
            state = self.sim.getState().as_vector()
            action = pi(obs, t)
            next_obs, reward, terminated, truncated, info = self.step(action)
            if return_data:
                data['state'].append(state)
                data['obs'].append(obs)
                data['action'].append(action)
                data['next_obs'].append(next_obs)
                data['reward'].append(np.array([reward]))
                data['terminal'].append(np.array([(1. if terminated or truncated else 0.)]))
            obs = next_obs
            R += reward
            t += 1
            if verbose>1:
                print("reward: ", reward)
            if terminated or truncated:
                break

        if verbose>0:
            print('total (non-discounted) return:', R)
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

###############################################################################

# Not importing from diversemanip.Workspace - circular import?
class ConfigsFile:
    manifest: dict
    supports: list
    config: dict
    q: np.array
    contacts: np.array
    forces: np.array

    def __init__(self, h5_filename):
        h5 = data_tools.H5Reader(h5_filename)
        self.__dict__ = h5.read_all() #MAGIC!

        # TODO: ASK Why all config files do not have manifest
        if 'manifest' in self.__dict__.keys():
            self.supports = self.manifest['potential supports']

class MujocoGymGoal(GoalEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode = 'rgb_array'

    def __init__(
        self, 
        config_path:str, 
        xml_path:str,
        scene_path:str,
        engine:str='mujoco',
        start_id:int=-1,
        goal_id:int=-1,
        sparse_r_thr:float|None=None, # Non-sparse reward if this is None
        tau_sim:float = 0.001,
        tau_step:float = 0.05, 
        time_limit:float=1.,
        feature_map:Any=None, 
        feature_target:Any=None, # TODO remove
        ctrl_lim:float|None=1.0,
        num_ctrl_pts:int=1,
        img_h:int=240,
        img_w:int=320,
        ):

        super().__init__() 

        self.config_path = config_path
        self.xml_path = xml_path
        self.scene_path = scene_path
        self.engine = engine
        self.start_id = start_id
        self.goal_id = goal_id
        self.sparse_r_thr = sparse_r_thr
        self.tau_sim = tau_sim
        self.tau_step = tau_step
        self.time_limit = time_limit
        self.feature_map = feature_map
        self.feature_target = feature_target
        self.ctrl_lim = ctrl_lim
        self.num_ctrl_pts = num_ctrl_pts
        self.img_h = img_h
        self.img_w = img_w

        # Load config
        self.config = ConfigsFile(self.config_path)

        self.create_sim()

        self.x0 = self.sim.getState()
        if self.x0.time > 0.:
            print('WARNING: initial sim has time>0 ... resetting this to 0')
            self.x0.time = 0.

        # define the observation space (see also observation_fct())
        # TODO - obs contains qpos,qvel for x and goal
        obs_shape = (self.x0.qpos.size + self.x0.qvel.size + self.x0.qpos.size + self.x0.qvel.size,)
        goal_shape = (self.x0.qpos.size + self.x0.qvel.size,)
        self.observation_space = Dict(
            dict(
                observation=Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=obs_shape, 
                    dtype=np.float32
                    ),
                achieved_goal=Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=goal_shape, 
                    dtype=np.float32
                    ),
                desired_goal=Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=goal_shape, 
                    dtype=np.float32
                    ),
            )
        )

        ctrl_lim = np.inf if self.ctrl_lim is None else self.ctrl_lim

        # define the action space
        self.action_space = gym.spaces.Box(
            low=-ctrl_lim, 
            high=ctrl_lim, 
            shape=(self.num_ctrl_pts*self.sim.ctrl_dim,), 
            dtype=np.float32
            )

    def __del__(self):
        del self.sim

    def create_sim(self):

        C = ry.Config()
        C.addFile(self.scene_path)
        #C.getFrame('obj').unLink() #different between 'analytical' model and sim: the ball is free in sim

        if self.engine=='mujoco':
            self.sim = MjSim(
                self.xml_path, 
                C, 
                use_mj_viewer=False, 
                tau_sim=self.tau_sim
                )
        elif self.engine=='physx':
            self.sim = ry.Simulation(
                C, 
                engine=ry.SimulationEngine.physx, 
                verbose=0
                )
        else:
            raise Exception(f'engine "{self.engine}" not defined')
                

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=int(seed))
            # np.random.seed(seed)
            # ry.rnd_seed(seed)

        # Choose start and end configurations
        config_count = self.config.qpos.shape[0]
        _start_id, _goal_id = random.sample(range(config_count), 2)
        start_id = self.start_id if self.start_id != -1 else _start_id
        goal_id = self.goal_id if self.goal_id != -1 else _goal_id

        # Set the initial state from config
        self.x0 = self.sim.getState()
        self.x0.time = 0.0
        self.x0.qpos = self.config.qpos[start_id]
        self.x0.qvel = np.zeros_like(self.x0.qvel)
        self.x0.act = np.zeros_like(self.x0.act)

        self.sim.setState(self.x0)
        self.sim.resetSplineRef(ctrl_time=0.)

        # Set the goal qpos
        self.goal = self.sim.getState()
        self.goal.time = 0.0  # TODO Not sure if this is ok or used
        self.goal.qpos = self.config.qpos[goal_id]
        self.goal.qvel = np.zeros_like(self.goal.qvel)
        self.goal.act = np.zeros_like(self.goal.act)

        obs_dict = self.observation_fct(self.x0, self.goal)
        info = {"start_id": start_id, "goal_id": goal_id}
        
        return obs_dict, info

    def step(self, action):

        action = action.reshape(1, self.sim.ctrl_dim)
        self.sim.setSplineRef(action, np.array([self.tau_step]), append=False)
        self.sim.step(tau_step=self.tau_step)
  
        x = self.sim.getState()
        assert x.time == self.sim.ctrl_time, "why not?"

        info = {"no": "additional info"}

        obs_dict = self.observation_fct(x, self.goal)

        reward = self.compute_reward(
            achieved_goal=obs_dict['achieved_goal'],
            desired_goal=obs_dict['desired_goal'],
            info=info
        )

        terminated = False
        truncated = (self.sim.ctrl_time >= self.time_limit) # terminated and truncated difference is super important
        
        return obs_dict, reward, terminated, truncated, info
    
    def observation_fct(self, x: MjSimState, goal: MjSimState):

        obs = np.concatenate((x.qpos, x.qvel, goal.qpos, goal.qvel)).astype(np.float32)
        ach_goal = np.concatenate((x.qpos, x.qvel)).astype(np.float32)
        des_goal = np.concatenate((goal.qpos, goal.qvel)).astype(np.float32)

        obs_dict = {
            "observation": obs,
            "achieved_goal": ach_goal,
            "desired_goal": des_goal,
        }

        return obs_dict

    # Mandatory for HER, method signature is fixed
    # Must be vectorized
    def compute_reward(
        self,
        achieved_goal: NDArray[np.float32],
        desired_goal: NDArray[np.float32],
        info: dict[str, Any]
    ) -> NDArray[np.float32] | float:
        
        assert self.feature_map is not None, "rewards w/o feature_map undefined"

        # Map goals into feature space
        z_0 = self.feature_map(achieved_goal)
        z_goal = self.feature_map(desired_goal)

        # phi shape: (D,) for single, (N, D) for batch
        phi = z_0 - z_goal

        # squared error per example
        if phi.ndim == 1:
            # scalar
            error = float(np.sum(np.square(phi)))
            if self.sparse_r_thr is not None:
                # sparse: success=1, else 0
                return 1.0 if np.linalg.norm(phi) < float(self.sparse_r_thr) else 0.0
            else:
                return -error
        else:
            # batched -> shape (N,)
            # sum of squares along feature axis
            errors = np.sum(np.square(phi), axis=1)
            if self.sparse_r_thr is not None:
                norms = np.linalg.norm(phi, axis=1)
                rewards = np.where(norms < float(self.sparse_r_thr), 1.0, 0.0)
            else:
                rewards = -errors

            # Important
            return np.asarray(rewards, dtype=np.float32)

        
    def rollout(self, pi, return_data=False, verbose=1):
        '''helper to play and view a policy'''
        obs, info = self.reset()

        if verbose>0:
            self.sim.C.view(True, 'START')
            self.sim.view_speed=1.
        else:
            self.sim.view_speed=-1.

        if return_data:
            data = {'state': [], 'obs': [], 'action': [], 'next_obs': [], 'reward': [], 'terminal': []}

        t = 0
        R = 0
        while True:
            state = self.sim.getState().as_vector()
            action = pi(obs, t)
            next_obs, reward, terminated, truncated, info = self.step(action)
            if return_data:
                data['state'].append(state)
                data['obs'].append(obs)
                data['action'].append(action)
                data['next_obs'].append(next_obs)
                data['reward'].append(np.array([reward]))
                data['terminal'].append(np.array([(1. if terminated or truncated else 0.)]))
            obs = next_obs
            R += reward
            t += 1
            if verbose>1:
                print("reward: ", reward)
            if terminated or truncated:
                break

        if verbose>0:
            print('total (non-discounted) return:', R)
            self.sim.C.view(True, 'END')
        
        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    def render(self):
        '''also part of the env.Gym'''

        # TODO: Maybe make rendering more flexible (zoom, etc.) later
        if self.render_mode == 'rgb_array':
            #return self.sim.C.gl().getImage()
            with mujoco.Renderer(self.sim.model, height=self.img_h, width=self.img_w) as renderer:
                mujoco.mj_forward(self.sim.model, self.sim.data)
                renderer.update_scene(self.sim.data)
                return renderer.render()
        elif self.render_mode == 'human':
            self.sim.C.view(False, f'RoboticGym time {self.sim.data.time} / {self.time_limit}')
        else:
            raise NotImplementedError('Unknown render_mode: {render_mode}')
