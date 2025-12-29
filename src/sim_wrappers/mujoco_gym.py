# initial version from e05-RobotGym.py (robot learning course)

from .mujoco_sim import *
import gymnasium as gym
import numpy as np
import time, math
import robotic as ry

###############################################################################

class MujocoGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode = 'human'
    verbose = 0

    def __init__(self, sim: MjSim, tau_step: float = 0.05, time_limit=1.,
                 goal_feat_map = None):
        self.sim = sim
        x0 = self.sim.getState()
        if x0.time > 0.:
            print('WARNING: initial sim has time>0 ... resetting this to 0')
            x0.time = 0.

        self.tau_step = tau_step
        self.goal_feat_map = goal_feat_map
        self.goal_feat_eps = 1e-2  #WATCH
        self.cost_const = 0.0
        self.time_limit = time_limit
        self._max_episode_steps = time_limit/self.tau_step

        # define the observation space (see also observation_fct())
        observation_dim = self.observation_fct(x0).size
        self.goal_dim = 0
        if goal_feat_map is not None:
            self.goal_dim = self.goal_feat_map(self.observation_fct(x0)).size
            observation_dim += self.goal_dim
        self.observation_space = gym.spaces.Box(-2., +2., shape=(observation_dim,), dtype=np.float32)

        # define the action space
        self.action_scale = 0.5*math.sqrt(tau_step)  #WATCH
        num_ctrl_pts = 1
        action_dim = num_ctrl_pts*self.sim.ctrl_dim
        self.action_space = gym.spaces.Box(-1., +1., shape=(action_dim,), dtype=np.float32)

        print(f"-- initialized MjGym with observation dim {observation_dim} (qdim:{x0.qpos.size}-4+qvel:{x0.qvel.size}+act:{x0.act.size}+goal:{self.goal_dim}), action dim {action_dim}, tau step {self.tau_step}, and time limit {self.time_limit}")

    def __del__(self):
        del self.sim

    def set_start_goal(self, start: MjSimState, goal: MjSimState):
        self.starts = start.as_vector().reshape(1,-1)
        self.goals = goal.as_vector().reshape(1,-1)

    def set_starts_goals(self, starts: np.array, goals: np.array):
        assert starts.shape[0]==goals.shape[0]
        self.starts = starts
        self.goals = goals
        # np.empty((goals.shape[0], self.goal_dim))
        # for i in range(goals.shape[0]):
        #     x = self.sim.to_state(goals[i])
        #     self.goals[i] = self.goal_map(self.observation_fct(x))

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=int(seed))
            # np.random.seed(seed)
            # ry.rnd_seed(seed)

        i = np.random.randint(0, self.starts.shape[0])
        g0 = self.sim.to_state(self.goals[i])
        self.goal_feat = self.goal_feat_map(self.observation_fct(g0, without_goal=True))
        if self.verbose>2:
            self.sim.setState(g0)
            self.sim.C.view(self.verbose>3, f'gym GOAL - time limit:{self.time_limit:6.3f}, feature: {self.goal_feat}')
            self.goal_image = self.sim.C.get_viewer().getRgb().copy()

        x0 = self.sim.to_state(self.starts[i])
        x0.time = 0.
        self.sim.setState(x0)
        self.sim.resetSplineRef(ctrl_time=0.)
        self.sim.data.actuator_force[:] = 0.

        # if self.random_reset:
        #     # resetting the box position to a random initial position -- makes it MUCH harder
        #     self.box_pos0 = np.array([.0,-.1,.7]) + .7 * np.random.rand(3)
        #     self.box_pos0[2]=.7
        if self.verbose>2:
            self.sim.C.view(self.verbose>3, f'gym START - t:{self.sim.ctrl_time:6.3f}')

        observation = self.observation_fct(x0)
        info = {"no": "additional info"}
        return observation, info

    def step(self, action):
        # reward and termination depends on x_now, not x_next!!
        x_now = self.sim.getState()
        # obs_now = self.observation_fct(x_now)
        # reward = self.reward_fct(obs_now)
        # terminated = self.is_goal(obs_now)
        # if terminated: # and self.sim.view_speed==1.:
        #     self.sim.C.view(False, f'termination at time {x_now.time}')

        # set action
        ctrl_ref = self.action_scale * action.reshape(1, self.sim.ctrl_dim).copy()
        # ctrl_ref += self.sim.spline_ref.eval3(self.sim.ctrl_time)[0] # NEW! relativ        
        ctrl_ref += x_now.qpos[: self.sim.ctrl_dim] # WATCH - relative to current position
        self.sim.setSplineRef(ctrl_ref, np.array([2.*self.tau_step]), append=False)
        # self.sim.resetSplineRef(const_ref=ctrl_ref, ctrl_time=self.sim.ctrl_time) # this would be the command to set a constant ref -- but not good for larget tau_step

        # step
        self.sim.step(tau_step=self.tau_step)
  
        # get obs and truncation
        x_next = self.sim.getState()
        assert x_next.time == self.sim.ctrl_time, "why not?"
        obs_next = self.observation_fct(x_next)
        reward = self.reward_fct(obs_next)  ##WATCH reward based on next state?
        terminated = self.is_goal(obs_next)
        truncated = (self.sim.ctrl_time >= self.time_limit) # terminated and truncated difference is super important

        if self.verbose>2:
            if terminated:
                self.sim.C.view(self.verbose>3, f'gym END - terminated, t:{self.sim.ctrl_time:6.3f}, reward: {reward}')
            elif truncated:
                self.sim.C.view(self.verbose>3, f'gym END - truncated, t:{self.sim.ctrl_time:6.3f}, reward: {reward}')
            # else:
            #     self.sim.C.view(False, f'GYM t:{self.sim.ctrl_time:6.3f} (reward: {reward})')

        info = {"no": "additional info"}
        return obs_next, reward, terminated, truncated, info
    
    def observation_fct(self, x: MjSimState, without_goal=False):
        obs = np.concatenate((x.qpos[:-4], self.tau_step*x.qvel)) # WATCH!!  object pos only;  qvel rescaled to delta-position!
        obs = np.concatenate((obs, .01 * x.act))
        if self.has_wrapper_attr('goal_feat') and not without_goal:
            obs = np.concatenate((obs, self.goal_feat))
        return obs

    def is_goal(self, obs):
        err = np.linalg.norm(self.goal_feat_map(obs)-self.goal_feat)
        # print('err', err)
        if err <= self.goal_feat_eps:
            return True
        return False

    def reward_fct(self, obs):
        if self.is_goal(obs):
            return 1.
        if self.cost_const>0.:
            return -self.tau_step * self.cost_const
        return 0.
        # z = self.feature_map(obs)
        # phi = z - self.feature_target
        # return -np.sum(np.square(phi))
        
    def rollout(self, pi, return_data=False, absolute_actions=False):
        '''helper to play and view a policy'''
        obs, info = self.reset()

        if return_data:
            data = {'state': [], 'obs': [], 'action': [], 'ctrl_cost': [], 'next_obs': [], 'reward': [], 'terminal': []}

        t = 0
        R = 0
        while True:
            state = self.sim.getState()
            action = pi(obs, t)
            if absolute_actions: # pi returns absolute actions, need to convert back before executing
                action = action - state.qpos[:self.sim.ctrl_dim]
                action /= self.action_scale
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
            if truncated: #terminated or 
                break

        if self.verbose>0:
            print('total (non-discounted) return:', R)
        
        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    def render(self):
        '''also part of the env.Gym'''
        self.C.view(False, f'RoboticGym time {self.time} / {self.time_limit}')
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()
