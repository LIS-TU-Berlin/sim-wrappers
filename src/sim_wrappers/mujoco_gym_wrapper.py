from typing import Any

import mujoco
import numpy as np
from gymnasium.spaces import Box
from numpy._typing import NDArray
import robotic as ry

from sim_wrappers.gym_wrapper import GymWrapper
from sim_wrappers.mujoco_sim import MjSim, MjSimState


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
            render_mode=render_mode,
        )
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
    ) -> tuple[MjSimState, np.float64, bool, bool, dict[str, np.float64]]:
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
