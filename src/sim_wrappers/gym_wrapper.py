from abc import ABCMeta
from typing import Any, SupportsFloat, Literal

import numpy as np
from gymnasium import Env as GymEnv, Space
from gymnasium.core import ObsType, ActType, RenderFrame
import robotic as ry
from sim_wrappers.mujoco_sim import MjSim, MjSimState


class GymWrapper(GymEnv, ABCMeta):

    def __init__(
        self,
        xml_path: str,
        ry_cfg: ry.Config,
        action_space: Space | None,
        observation_space: Space | None,
        engine: Literal["mujoco", "physx"] = "mujoco",
        tau_ctrl: float = 5e-2,
        tau_sim: float = 1e-3,
    ):
        super().__init__()
        if engine == 'physx':
            self.sim = ry.Simulation(ry_cfg, engine=ry.SimulationEngine.physx, verbose=0)
        elif engine == 'mujoco':
            self.sim = MjSim(xml_path, ry_cfg, use_mj_viewer=True, tau_sim=tau_sim)
        else:
            raise Exception(f'engine "{engine}" not defined')

        self.tau_ctrl = tau_ctrl
        self.observation_space = observation_space
        self.action_space = action_space
        self.q0 = self.sim.getState()
        self.spline_times = np.linspace(0., tau_sim, int(tau_ctrl / tau_sim))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjSimState, dict[str, Any]]:
        self.sim.resetSplineRef()
        self.sim.setState(self.q0)
        return self.sim.getState(), {}

    def step(
        self, action: ActType
    ) -> tuple[MjSimState, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        self.sim.__del__()

    def unwrapped(self) -> MjSim:
        return self.sim
