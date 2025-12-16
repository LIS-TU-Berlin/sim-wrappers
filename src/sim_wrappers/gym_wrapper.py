from typing import Any, SupportsFloat, Literal

import numpy as np
from gymnasium import Env as GymEnv, Space
from gymnasium.core import ObsType, ActType, RenderFrame
import robotic as ry
from sim_wrappers.mujoco_sim import MjSim, MjSimState


class GymWrapper(GymEnv):

    def __init__(
        self,
        xml_path: str,
        ry_cfg: ry.Config,
        action_space: Space | None,
        observation_space: Space | None,
        engine: Literal["mujoco", "physx"] = "mujoco",
        tau_step: float = 5e-2,
        tau_sim: float = 1e-3,
    ):
        super().__init__()
        if engine == 'physx':
            self.sim = ry.Simulation(ry_cfg, engine=ry.SimulationEngine.physx, verbose=0)
        elif engine == 'mujoco':
            self.sim = MjSim(xml_path, ry_cfg, use_mj_viewer=False, tau_sim=tau_sim)
        else:
            raise Exception(f'engine "{engine}" not defined')

        self._tau_step = tau_step
        self.observation_space = observation_space
        self.action_space = action_space
        self.init_state = self.sim.getState()
        self.spline_times = np.array([2. * self._tau_step])  # taken from mujoco_gym

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjSimState, dict[str, Any]]:
        super().reset(seed=seed)
        self.sim.resetSplineRef(ctrl_time=0.)
        self.sim.setState(self.init_state)
        return self.sim.getState(), {}

    def step(
        self, action: ActType
    ) -> tuple[MjSimState, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        self.__del__()

    def unwrapped(self) -> MjSim:
        return self.sim

    def __del__(self):
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    @property
    def tau_step(self):
        return self._tau_step
