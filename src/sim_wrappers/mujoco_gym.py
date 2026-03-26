# initial version from e05-RobotGym.py (robot learning course)

from .mujoco_sim import *
from .poly_ref import SecondOrderPolyRef
from gymnasium import Env, spaces
import numpy as np
from dataclasses import dataclass
import math
from enum import Enum

###############################################################################

@dataclass
class MujocoGymCfg:
    tau_step = 0.05
    time_limit = 1.
    goal_feat_eps = 1e-2   #WATCH
    cost_const = 0.0
    action_scale_sqrttau = 0.5   #WATCH
    obs_pos_scale = 1.
    obs_vel_scale = .05
    obs_referr_scale = 20.
    bounds_margin = .01
    eff_action = None
    observation_points = None

class MujocoGym(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode = 'human'
    verbose = 0

    def __init__(self, sim: MujocoSim, cfg: MujocoGymCfg, goal_feat_map = None, num_scenes=1, terminal_bounds=None):
        self.sim = sim

        self.cfg = cfg
        if self.cfg.eff_action=='none':
            self.cfg.eff_action=None
        if self.cfg.observation_points==[]:
            self.cfg.observation_points=None

        self.goal_feat_map = goal_feat_map
        self.terminal_bounds = terminal_bounds
        self.num_scenes = num_scenes
        self.num_worlds = 1
        if self.sim.warp_worlds>0.:
            self.num_worlds = self.sim.warp_worlds
            self.num_scenes *= self.sim.warp_worlds
            self.terminal_bounds = np.tile(self.terminal_bounds, (self.num_worlds, 1, 1))
            self.terminal_bounds = np.transpose(self.terminal_bounds, (1, 0, 2))
            self.terminal_bounds = np.reshape(self.terminal_bounds, (2, -1))
        self.scene_needs_reset = np.ones((self.num_scenes), dtype=bool)
        self.scene_time = np.zeros((self.num_scenes))
        self.ctrl_indices = self.sim.ctrl_indices.reshape(self.num_scenes//self.num_worlds, -1)[0]
        self.ctrl_current = self.qpos[:, self.ctrl_indices].copy()

        # to get the first observatin, we need to setup a ctrlRef, get a goal feature, then query an observation
        self.goal_feat = self.goal_feat_map(self.qpos, self.qvel)
        self.observation, self.feat = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat)
        self.observation_space = spaces.Box(-2., +2., shape=self.observation.shape, dtype=np.float32)
        self.single_observation_space = spaces.Box(-2., +2., shape=(self.observation.shape[1],), dtype=np.float32)

        # define the action space
        self.action_scale = self.cfg.action_scale_sqrttau*math.sqrt(self.cfg.tau_step)
        if self.cfg.eff_action is None:
            action_dim = self.sim.ctrl_dim // (self.num_scenes//self.num_worlds)
        else:
            action_dim = 6
            self.q_home = self.qpos[:, self.ctrl_indices]
        self.action_space = spaces.Box(-1., +1., shape=(self.num_scenes, action_dim), dtype=np.float32)
        self.single_action_space = spaces.Box(-1., +1., shape=(action_dim,), dtype=np.float32)

        # a counter to loop through provided starts/goals
        self.starts_goals_counter = 0

        # for logging
        self.total_steps = 0
        self.total_eps = 0
        self.total_reward = 0
        
        print(f"-- initialized MjGym with observation dim {self.observation.shape} (qdim:{sim.qpos_dim}-4+qvel:{sim.qvel_dim}+ctrl:{sim.ctrl_dim}+goal:{self.feat.size}), action dim {action_dim} (pose_action={self.cfg.eff_action}), tau step {self.cfg.tau_step}, and time limit {self.cfg.time_limit}")

    def set_starts_goals(self, starts_q: np.array, goals_q: np.array):
        assert starts_q.shape[0]==goals_q.shape[0]
        self.starts_goals_counter = 0
        self.starts_q = np.atleast_2d(starts_q)
        self.starts_v = np.zeros((starts_q.shape[0], self.sim.qvel_dim//(self.num_scenes//self.num_worlds)))
        self.goals_q = np.atleast_2d(goals_q)
        self.goals_v = np.zeros((goals_q.shape[0], self.sim.qvel_dim//(self.num_scenes//self.num_worlds)))

    def next_starts_goals_counter(self):
        i = self.starts_goals_counter % self.starts_q.shape[0]
        self.starts_goals_counter += 1
        return i

    def auto_reset(self):
        assert self.num_scenes>0
        qpos, qvel, act = self.qpos, self.qvel, self.act
        needs_set = False
        
        for s in range(self.num_scenes):
            if self.scene_needs_reset[s]:
                i = self.next_starts_goals_counter()
                # i = np.random.randint(0, self.starts_q.shape[0])

                self.goal_feat[s] = self.goal_feat_map(self.goals_q[i:i+1], self.goals_v[i:i+1])
                qpos[s] = self.starts_q[i]
                qvel[s] = self.starts_v[i]
                act[s] *= 0.
                
                cref = qpos[s:s+1, self.ctrl_indices]
                self.ctrl_current[s] = cref
                
                self.observation[s], self.feat[s] = self.observation_fct(qpos[s:s+1], qvel[s:s+1], cref, self.goal_feat[s:s+1], selected_scenes=[s])

                self.scene_needs_reset[s] = False
                self.scene_time[s] = 0.
                needs_set = True
                
        if needs_set:
            self.sim.set_state(qpos, qvel, act)

        return self.observation, { 'feature': self.feat }

    def reset(self, seed=None, options=None):
        """resets all scenes in the env"""
        if seed is not None:
            super().reset(seed=int(seed))

        self.scene_needs_reset[:] = True
        
        return self.auto_reset()    

    def step(self, action):
        if action.ndim==1:
            action = action.reshape(self.num_scenes, -1)
        assert action.shape[0]==self.num_scenes
        if self.cfg.eff_action is None:
            assert action.shape[1]==len(self.ctrl_indices)
        else:
            assert action.shape[1]==6

        # is a endeffector delta action?
        if self.cfg.eff_action is not None:
            action = self.convert_eff_action(action)

        # set action
        action_delta = self.action_scale * action
        current_vel = self.qvel[:, self.ctrl_indices]
        tmp = SecondOrderPolyRef(self.sim.ctrl_time, self.ctrl_current, current_vel, action_delta, 2.*self.cfg.tau_step)
        self.sim.ctrl_buffer = tmp.sample_buffer(self.sim.ctrl_time,
                                                    self.sim.ctrl_time+self.cfg.tau_step,
                                                    self.sim.tau_sim)
        self.sim.ctrl_bufferPtr = 0

        # step
        self.sim.step(tau_step=self.cfg.tau_step)
        self.scene_time += self.cfg.tau_step
        self.ctrl_current = self.sim.ctrl_buffer[self.sim.ctrl_bufferPtr]
  
        # get obs and truncation
        self.observation, self.feat = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat)
        reward = self.reward_fct(self.observation, self.feat)
        terminated = self.is_goal(self.observation, self.feat)
        truncated = (self.scene_time >= self.cfg.time_limit) # terminated and truncated difference is super important
        if self.terminal_bounds is not None:
            truncated |= self.is_out_of_bound(self.qpos, self.qvel)
        self.scene_needs_reset = np.logical_or(terminated, truncated)

        self.total_steps += self.num_scenes
        self.total_eps += np.count_nonzero(self.scene_needs_reset)
        self.total_reward += np.sum(reward)

        if self.num_scenes==1: # for stable_baselines to work..
            reward = reward.item()
        return self.observation, reward, terminated, truncated, { 'feature': self.feat }

    def reward_fct(self, obs, feat):
        # example method for a reward function -- this should be overloaded
        # this example returns a binary 1/0 indicating is_goal
        # BEWARE: the method needs to be vectorized, returning the vector of rewards for all scenes in the env
        return np.where(self.is_goal(obs, feat), 1., 0.)
        # if self.cfg.cost_const>0.:
        #     return -self.cfg.tau_step * self.cfg.cost_const
        
    def observation_fct(self, qpos, qvel, cref, goal_feat, selected_scenes=None):
        # example method for an observation function -- this should be overloaded
        # this example concatenates:
        # - joint position, velocity, and ctrl_err observations
        # - additional 3D observation points
        # - and the goal feature
        # BEWARE: the method needs to be vectorized, returning the matrix of observations for all scenes in the env
        o_pos = self.cfg.obs_pos_scale * qpos[:, :-4] # this excludes the quaternion of the object (assumed last joint!)
        o_vel = self.cfg.obs_vel_scale * qvel
        o_err = self.cfg.obs_referr_scale * (cref - qpos[:, self.ctrl_indices])
        obs = np.hstack((o_pos, o_vel, o_err))

        if self.cfg.observation_points is not None:
            o_eff = self.cfg.obs_pos_scale * self.get_effpos_observation(selected_scenes)
            obs = np.hstack((obs, o_eff))
        
        feat = self.goal_feat_map(qpos, qvel)
        o_goal = self.cfg.obs_pos_scale * (goal_feat - feat)
        obs = np.hstack((obs, o_goal))
        obs = np.clip(obs, -2., 2.)
        return obs, feat

    ### helpers to define the observation function (all of these are vectorized)
    
    def get_effpos_observation(self, selected_scenes=None):
        if selected_scenes is None:
            selected_scenes = range(self.num_scenes)
        o_eff = np.empty((len(selected_scenes), len(self.cfg.observation_points), 3))
        for i,s in enumerate(selected_scenes):
            for j,name in enumerate(self.cfg.observation_points):
                if self.num_scenes>1:
                    n = f'{s}_{name}'
                else:
                    n = name
                o_eff[i,j] = self.sim.get_position(n)
        return o_eff.reshape(len(selected_scenes), -1)

    def is_goal(self, obs, feat):
        err = np.linalg.norm(feat-self.goal_feat, axis=1)
        return (err <= self.cfg.goal_feat_eps)

    def is_out_of_bound(self, qpos, qvel):
        if self.terminal_bounds is None:
            return False
        assert self.terminal_bounds.shape[0]==2
        assert self.terminal_bounds.shape[1]==qpos.size
        assert not np.any(self.terminal_bounds[1]<=self.terminal_bounds[0]), f"bounds (joint) not proper: {self.terminal_bounds}"
        l = np.any(qpos < self.terminal_bounds[0].reshape(qpos.shape) - self.cfg.bounds_margin, axis=1)
        g = np.any(qpos > self.terminal_bounds[1].reshape(qpos.shape) + self.cfg.bounds_margin, axis=1)
        return np.logical_or(l,g) 

    ### helpers to convert actions

    def convert_eff_action(self, action):
        ns = self.num_scenes
        nc = len(self.ctrl_indices)
        pose_action = action
        action = np.zeros((ns, nc))
        for s in range(ns):
            Jpos, Jang = self.sim.get_Jacobian(f'{s}_{self.cfg.eff_action}')
            J = np.vstack((Jpos, Jang))
            J = J.reshape(6, ns, -1)
            J = J[:,s,:nc]
            Jinv = J.T @ np.linalg.pinv(J@J.T+1e-3*np.eye(J.shape[0]))
            action[s,:] = Jinv @ pose_action[s,:]
            if self.q_home is not None:
                action[s,:] += 0.1*(np.eye(nc) - Jinv@J) @ (self.q_home-self.qpos[s, :nc])
        return action
    
    ### minimal example to evaluate a policy

    def rollout(self, pi, return_data=False):
        '''helper to play and view a policy'''

        obs, info = self.reset()

        if return_data:
            data = {'state': [], 'obs': [], 'action': [], 'ctrl_cost': [], 'next_obs': [], 'reward': [], 'terminal': []}

        t = 0
        R = 0
        while True:
            state = self.sim.getState()
            action = pi(obs, t)
            self.sim.ctrl_costs=0.
            next_obs, reward, terminated, truncated, info = self.step(action)
            if return_data:
                data['state'].append(state.as_vector())
                data['obs'].append(obs)
                data['action'].append(action)
                data['ctrl_cost'].append(self.sim.ctrl_costs)
                data['next_obs'].append(next_obs)
                data['reward'].append(np.array([reward]))
                data['terminal'].append(np.array([(1 if terminated or truncated else 0)], dtype=np.int16))
            obs = next_obs
            R += reward
            t += 1
            if self.verbose>1:
                print("reward: ", reward)
            if np.any(np.logical_or(terminated, truncated)):
                break

        if self.verbose>0:
            print('total (non-discounted) return:', R)
        
        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    @property
    def qpos(self) -> np.array:
        return self.sim._qpos.reshape(self.num_scenes, -1)

    @property
    def qvel(self) -> np.array:
        return self.sim._qvel.reshape(self.num_scenes, -1)

    @property
    def act(self) -> np.array:
        return self.sim._act.reshape(self.num_scenes, -1)

    def render(self):
        '''also part of the env.Gym'''
        self.C.view(False, f'RoboticGym time {self.time} / {self.cfg.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()
