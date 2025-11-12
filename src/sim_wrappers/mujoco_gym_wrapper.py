from typing import Any

import mujoco
import numpy as np
from chex import Array
from gymnasium.spaces import Box, Dict
from gymnasium_robotics.core import GoalEnv
from numpy._typing import NDArray
import robotic as ry

from sim_wrappers.gym_wrapper import GymWrapper
from sim_wrappers.mujoco_sim import MjSim, MjSimState


def sim_move_targets(
    q_target: Array,
    q_ref: Array,
    qvel_ref: Array,
    time_cost: float = 5.,
    overwrite: bool = True
) -> tuple[Array, Array]:
    """Simulates botop's moveTo with overwrite=True.
    Can be used with any sim.

    Params:
        q_target: target qpos.
        q_ref: current qpos.
        qvel_ref: current qvel.
        time_cost: penalty.
        overwrite: Currently only supports True.
    """
    assert overwrite, "moveTo currently only supports overwrite == True."

    dist = np.linalg.norm(q_ref - q_target) + 1e-4
    vel = np.dot(qvel_ref, (q_target - q_ref)) / dist
    t = (np.sqrt(6.0 * time_cost * dist + vel * vel) - vel) / time_cost
    t = np.where(dist < 1e-4 or t < 0.1, 0.1, t)
    path = q_target[np.newaxis, :]
    times = [t]

    return path, times


class MujocoGymWrapper(GymWrapper):
    def __init__(
        self,
        xml_path: str,
        ry_cfg: ry.Config,
        action_space: Box | None,
        observation_space: Box | None,
        tau_ctrl: float = 5e-2,
        tau_sim: float = 1e-3,
        render_mode: str | None = None,
    ):
        super().__init__(
            xml_path=xml_path,
            ry_cfg=ry_cfg,
            tau_ctrl=tau_ctrl,
            engine="mujoco",
            observation_space=observation_space,
            action_space=action_space,
        )
        self.render_mode = render_mode
        self.sim = MjSim(xml_path, ry_cfg, use_mj_viewer=True, tau_sim=tau_sim)
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.sim.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjSimState, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        mujoco.mj_forward(self.model, self.data)
        self.sim.resetSplineRef(0.0)
        self.sim.pullConfigFromSim()
        state = self.sim.getState()
        info = self.get_info(state)

        return state, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[MjSimState, np.float32, bool, bool, dict[str, Any]] :
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.sim.setSplineRef(action, times=self.spline_times)
        self.sim.step(self.sim.tau_step)
        obs = self.get_obs()
        info = self.get_info(obs)
        terminated = self.compute_terminated(obs, info)
        truncated = self.compute_truncated(obs, info)
        reward = self.compute_reward(obs, info)

        return obs, reward, terminated, truncated, info

    def get_obs(self) -> NDArray:
        raise NotImplementedError()

    def compute_terminated(self, obs: NDArray, info: dict, *args, **kwargs) -> bool:
        raise NotImplementedError()

    def compute_truncated(self, obs: NDArray, info: dict, *args, **kwargs) -> bool:
        raise NotImplementedError()

    def compute_reward(self, obs: NDArray, info: dict, *args, **kwargs) -> bool:
        raise NotImplementedError()

    def get_info(self, obs: NDArray) -> dict:
        return {}

    @property
    def model(self) -> mujoco.MjModel:
        return self.sim.model

    @property
    def data(self) -> mujoco.MjData:
        return self.sim.data


class MujocoGymWrapperConditionalEnv(MujocoGymWrapper, GoalEnv):

    def __init__(
            self, 
            xml_path: str, 
            ry_cfg: ry.Config, 
            action_space: Box | None, 
            observation_space: Dict | None, 
            tau_ctrl: float = 0.05, 
            tau_sim: float = 0.001, 
            render_mode: str | None = None,
            configs_pth: str | Path = "",
            config_start_id: int = -1,
            config_end_id: int = -1,
            mask: NDArray | None = None,
            use_vel: bool = True,
            max_steps: int = 1_000_000, 
            sparse: bool = False,
            verbose:int = 0,
            ):
        super().__init__(xml_path, ry_cfg, action_space, observation_space, tau_ctrl, tau_sim, render_mode)

        self.configs = h5py.File(configs_pth, 'r')
        self.config_count = self.configs["qpos"].shape[0]
        self.config_start_id = config_start_id
        self.config_end_id = config_end_id
        self.mask = mask
        self.use_vel = use_vel
        self.max_steps = max_steps
        self.sparse = sparse
        self.verbose = verbose

        # Ideally, should be in MjSim
        self.renderer = mujoco.Renderer(self.sim.model, 160, 120)
        self.camera = "fixed_cam"  # TODO Make argument

        if observation_space is None:
            # Observation comprises of qpos, qvel (if use_vel is True), pos goal
            obs_size = self.initial_qpos.shape[0] + self.initial_qvel.shape[0] if self.use_vel else 0 + self.initial_qpos.shape[0]
            goal_size = self.initial_qpos.shape[0]
            self.observation_space = Dict(
            dict(
                observation=Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32),
                achieved_goal=Box(low=-np.inf, high=np.inf, shape=(goal_size,), dtype=np.float32),
                desired_goal=Box(low=-np.inf, high=np.inf, shape=(goal_size,), dtype=np.float32),
            )
        )
            
        if action_space is None:
            # TODO verify control limits
            self.action_space = Box(low=-1.0, high=1.0, shape=(self.ctrl_dim,), dtype=np.float32)


    # TODO Test reset
    def reset(
            self, 
            *, 
            seed: int | None = None, 
            options: dict[str, Any] | None = None
            ) -> tuple[dict[str, NDArray], dict[str, Any]]:
        
        super().reset(seed=seed, options=options)
        np.random.seed(seed)

        # Choose start and end configurations
        def randint_excluding(low: int, high: int, exclude: int) -> int:
            """Return random integer in [low, high) excluding `exclude`."""
            x = np.random.randint(low, high - 1) if exclude >= low else np.random.randint(low, high)
            return x + 1 if x >= exclude >= low else x

        s_cfg_idx = self.config_start_id if self.config_start_id != -1 else np.random.randint(0, self.config_count)
        e_cfg_idx = self.config_end_id if self.config_end_id != -1 else randint_excluding(0, self.config_count, s_cfg_idx)

        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx}
        
        self.target_state = self.stable_configs["qpos"][e_cfg_idx]
        
        # Reset simulation state to start state from config
        # use self.sim.setState

        state = self.sim.getState()
        state.time = 0.0
        state.qpos = self.configs["qpos"][s_cfg_idx]
        state.qvel = np.zeros(self.sim.q_dim)
        state.act = self.configs["ctrl"][s_cfg_idx]
        self.sim.setState(state)
        
        self.iter = 0
                
        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        obs_dict = {
            "observation": self.getState(), # observation shape is (29,0)
            "achieved_goal": self.sim_state[1].copy(),
            "desired_goal": self.target_state.copy(),
        }

        return obs_dict, info

    
    def step(
            self, 
            action: NDArray
            ) -> tuple[dict[str, NDArray], np.float32, bool, bool, dict[str, Any]]:

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.sim.setSplineRef(action, times=self.spline_times)
        self.sim.step(self.sim.tau_step)
        self.iter += 1

        obs = self.get_obs()
        info = self.get_info(obs)
        terminated = self.compute_terminated(obs, info)
        truncated = self.compute_truncated(obs, info)

        achieved_goal = self.sim.getState().qpos.copy()
        desired_goal = self.target_state.copy()

        # Method signature is fixed for HER
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        obs_dict = {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

        return obs_dict, reward, terminated, truncated, info
    
    # TODO Test sparse reward
    def compute_reward(
            self, 
            achieved_goal: NDArray,
            desired_goal: NDArray,
            info: dict[str, Any]
            ) -> np.float32:
        
        # Vectorized code
        e_goal = achieved_goal*self.mask - desired_goal*self.mask  # shape: (n_env, ndim) or (ndim,)
        goal_reward = -np.sum(e_goal**2, axis=-1) # Can handle vectorized and single env
        
        if self.sparse:
            reward = 0.0 if goal_reward < -10.0 else 1.0
        else:
            reward = np.maximum(goal_reward, -5.0)

        return reward


    def get_obs(self) -> NDArray:

        state = self.sim.getState()

        qpos = state.qpos
        qvel = state.qvel
        goal = self.target_state

        obs = np.concatenate((qpos, qvel, goal)) if self.use_vel else np.concatenate((qpos, goal))
        
        return obs

    # TODO Decide about termination
    def compute_terminated(self, obs: NDArray, info: dict, *args, **kwargs) -> bool:
        # Termination: task success or failure
        return False

    def compute_truncated(self, obs: NDArray, info: dict, *args, **kwargs) -> bool:
        return self.iter >= self.max_steps


    # TODO Check
    def render(self, mode: str = "rgb_array") -> np.ndarray:
            
        if self.camera:
            self.renderer.update_scene(self.sim.data, self.camera)
        else:
            self.renderer.update_scene(self.sim.data)

        frame = self.renderer.render()
        return frame

