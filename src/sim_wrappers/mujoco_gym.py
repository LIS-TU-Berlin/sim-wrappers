# initial version from e05-RobotGym.py (robot learning course)

from .mujoco_sim import *
from gymnasium import Env, spaces
import numpy as np
from dataclasses import dataclass
import math
from enum import Enum

###############################################################################

@dataclass
class MujocoGymConfig:
    tau_step = 0.05
    time_limit = 1.
    goal_feat_eps = 1e-2   #WATCH
    cost_const = 0.0
    action_scale_sqrttau = 0.5   #WATCH
    obs_pos_scale = 1.
    obs_vel_scale = .05
    obs_referr_scale = 20.
    bounds_margin = .01

class MujocoGym(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode = 'human'
    verbose = 0
    qpos_offset = None

    def __init__(self, sim: MujocoSim, cfg: MujocoGymConfig, goal_feat_map = None, num_scenes=1, terminal_bounds=None):
        self.sim = sim
        x0 = self.sim.getState()
        if x0.time > 0.:
            print('WARNING: initial sim has time>0 ... resetting this to 0')
            x0.time = 0.

        self.cfg = cfg
        self.goal_feat_map = goal_feat_map
        self.terminal_bounds = terminal_bounds

        self.num_scenes = num_scenes
        self.scene_needs_reset = np.ones((num_scenes), dtype=bool)
        self.scene_time = np.zeros((num_scenes))
        self.ctrl_indices = self.sim.ctrl_indices.reshape(num_scenes, -1)[0]

        # to get the first observatin, we need to setup a ctrlRef, get a goal feature, then query an observation
        cref = self.qpos[:, self.ctrl_indices]
        self.sim.ctrlRef_spline = None
        self.sim.ctrlRef_poly = SecondOrderCtrlRef(self.sim.ctrl_time, cref, np.zeros(cref.shape), np.zeros(cref.shape), 2.*self.cfg.tau_step)
        self.goal_feat = self.goal_feat_map(self.qpos, self.qvel)
        self.observation, feat = self.observation_fct(self.qpos, self.qvel, cref, self.goal_feat)
        # observation_dim = self.observation.shape[1]
        self.observation_space = spaces.Box(-2., +2., shape=self.observation.shape, dtype=np.float32)

        # define the action space
        self.action_scale = self.cfg.action_scale_sqrttau*math.sqrt(self.cfg.tau_step)
        action_dim = self.sim.ctrl_dim // num_scenes
        self.action_space = spaces.Box(-1., +1., shape=(num_scenes, action_dim), dtype=np.float32)

        print(f"-- initialized MjGym with observation dim {self.observation.shape} (qdim:{x0.qpos.size}-4+qvel:{x0.qvel.size}+act:{x0.act.size}+goal:{feat.size}), action dim {action_dim}, tau step {self.cfg.tau_step}, and time limit {self.cfg.time_limit}")

    def __del__(self):
        del self.sim

    def set_starts_goals2(self, starts_q: np.array, goals_q: np.array):
        assert starts_q.shape[0]==goals_q.shape[0]
        self.starts_q = np.atleast_2d(starts_q)
        self.starts_v = np.zeros((starts_q.shape[0], self.sim.qvel_dim//self.num_scenes))
        self.goals_q = np.atleast_2d(goals_q)
        self.goals_v = np.zeros((goals_q.shape[0], self.sim.qvel_dim//self.num_scenes))

    def auto_reset(self):
        assert self.num_scenes>0
        # x = self.sim.getState()
        qpos, qvel, act = self.qpos, self.qvel, self.act
        needs_set = False
        
        for s in range(self.num_scenes):
            if self.scene_needs_reset[s]:
                i = np.random.randint(0, self.starts_q.shape[0])
                self.goal_feat[s] = self.goal_feat_map(self.goals_q[i:i+1], self.goals_v[i:i+1])

                qpos[s] = self.starts_q[i]
                qvel[s] = self.starts_v[i]
                act[s] *= 0.
                
                cref = qpos[s:s+1, self.ctrl_indices]
                if self.sim.ctrlRef_poly is not None:
                    self.sim.ctrlRef_poly.reset(cref, s)
                if self.sim.ctrlRef_spline is not None:
                    self.sim.resetSplineRef(0., cref)
                self.observation[s], _ = self.observation_fct(qpos[s:s+1], qvel[s:s+1], cref, self.goal_feat[s:s+1])
                self.scene_needs_reset[s] = False
                self.scene_time[s] = 0.
                needs_set = True
                
        if needs_set:
            # self.sim.setState(x)
            if self.qpos_offset is None:
                self.sim.data.qpos = qpos.reshape(-1)
            else:
                self.sim.data.qpos = qpos.reshape(-1) + self.qpos_offset
            self.sim.data.qvel = qvel.reshape(-1)
            self.sim.data.actuator_force = act.reshape(-1)
            mujoco.mj_forward(self.sim.model, self.sim.data)

        return self.observation, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=int(seed))

        self.scene_needs_reset[:] = True
        
        return self.auto_reset()    

        i = np.random.randint(0, self.starts_q.shape[0])
        self.goal_feat = self.goal_feat_map(self.goals_q[i:i+1], self.goals_v[i:i+1])

        # x0 = MjSimState(0., self.starts_q[i], self.starts_v[i], np.zeros((self.sim.ctrl_dim)))
        # self.sim.setState(x0)
        qpos, qvel, act = self.qpos, self.qvel, self.act

        qpos = self.starts_q[i]
        qvel = self.starts_v[i]
        act *= 0.
        self.sim.data.qpos = qpos
        self.sim.data.qvel = qvel
        self.sim.data.actuator_force = act
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # self.sim.resetSplineRef(ctrl_time=0.)
        cref = self.qpos[:, self.ctrl_indices]
        self.sim.ctrlRef.reset(cref, 0)
        self.scene_time[0] = 0.

        if self.verbose>2:
            self.sim.C.view(self.verbose>3, f'gym START - t:{self.sim.ctrl_time:6.3f}')

        observation, feat = self.observation_fct(self.starts_q[i:i+1], self.starts_v[i:i+1], self.sim.ctrlRef.eval(self.sim.ctrl_time))
        info = { 'start_goal_id': i, 'start_state_feat': feat}
        return observation, info

    def step(self, action):
        if action.ndim==1:
            action = action.reshape(self.num_scenes, -1)
        assert action.shape[0]==self.num_scenes
        assert action.shape[1]==len(self.ctrl_indices)

        # set action
        action_delta = self.action_scale * action
        if self.sim.ctrlRef_poly is not None:
            current_ref = self.sim.ctrlRef_poly.eval(self.sim.ctrl_time)
            current_vel = self.qvel[:, self.ctrl_indices]
            self.sim.ctrlRef_poly = SecondOrderCtrlRef(self.sim.ctrl_time, current_ref, current_vel, action_delta, 2.*self.cfg.tau_step)
        else:
            # current_pos = self.sim.spline_ref.eval3(self.sim.ctrl_time)[0] # relativ to current ref
            current_pos = self.qpos[:, self.ctrl_indices]
            target = action_delta + current_pos
            self.sim.updateSplineRef(target, np.array([2.*self.cfg.tau_step]), append=False)
 
        # step
        self.sim.step(tau_step=self.cfg.tau_step)
        self.scene_time += self.cfg.tau_step
  
        # get obs and truncation
        self.observation, feat = self.observation_fct(self.qpos, self.qvel, self.sim.get_ctrlRef(), self.goal_feat)
        reward = self.reward_fct(self.observation, feat)
        terminated = self.is_goal(self.observation, feat)
        truncated = (self.scene_time >= self.cfg.time_limit) # terminated and truncated difference is super important
        if self.terminal_bounds is not None:
            terminated |= self.is_out_of_bound(self.qpos, self.qvel)
        self.scene_needs_reset = np.logical_or(terminated, truncated)

        # if self.verbose>2:
        #     if terminated:
        #         self.sim.C.view(self.verbose>3, f'gym END - terminated, t:{self.sim.ctrl_time:6.3f}, reward: {reward}')
        #     elif truncated:
        #         self.sim.C.view(self.verbose>3, f'gym END - truncated, t:{self.sim.ctrl_time:6.3f}, reward: {reward}')
        #     # else:
        #     #     self.sim.C.view(False, f'GYM t:{self.sim.ctrl_time:6.3f} (reward: {reward})')

        if self.num_scenes==1: # for stable_baselines to work..
            reward = reward.item()
        return self.observation, reward, terminated, truncated, {}
    
    def observation_fct(self, qpos, qvel, cref, goal_feat):
        o_pos = self.cfg.obs_pos_scale * qpos[:, :-4]
        o_vel = self.cfg.obs_vel_scale * qvel
        o_err = self.cfg.obs_referr_scale * (cref - qpos[:, self.ctrl_indices]) #self.sim.ctrlRef.eval(self.sim.ctrl_time)
        obs = np.hstack((o_pos, o_vel, o_err))
        feat = self.goal_feat_map(qpos, qvel)
        o_goal = self.cfg.obs_pos_scale * (goal_feat - feat)
        obs = np.hstack((obs, o_goal))
        obs = np.clip(obs, -2., 2.)
        return obs, feat

    def is_goal(self, obs, feat):
        err = np.linalg.norm(feat-self.goal_feat, axis=1)
        # print('err', err)
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

    def reward_fct(self, obs, feat):
        return np.where(self.is_goal(obs, feat), 1., 0.)
        # if self.cfg.cost_const>0.:
        #     return -self.cfg.tau_step * self.cfg.cost_const
        
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
            if np.all(terminated or truncated):
                break

        if self.verbose>0:
            print('total (non-discounted) return:', R)
        
        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    @property
    def qpos(self) -> np.array:
        if self.qpos_offset is not None:
            return (self.sim.data.qpos-self.qpos_offset).reshape(self.num_scenes, -1)
        else:
            return self.sim.data.qpos.reshape(self.num_scenes, -1)

    @property
    def qvel(self) -> np.array:
        return self.sim.data.qvel.reshape(self.num_scenes, -1)

    @property
    def act(self) -> np.array:
        return self.sim.data.actuator_force.reshape(self.num_scenes, -1)

    def render(self):
        '''also part of the env.Gym'''
        self.C.view(False, f'RoboticGym time {self.time} / {self.cfg.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()
