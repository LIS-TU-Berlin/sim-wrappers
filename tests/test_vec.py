"""
Tests for vectorized environments.
"""

import gymnasium as gym
import numpy as np
import pytest
from typing import Union
import robotic as ry
from sim_wrappers.mujoco_sim import MujocoSim as MjSim

class SimpleGymEnv(gym.Env):
    """Minimal Gymnasium env on top of sim_wrappers.mujoco_sim.MujocoSim.

    This env is intentionally tiny:
    - observation: concatenated qpos and qvel
    - action: a vector of size sim.ctrl_dim (passed to setSplineRef)
    - reward: 0 always
    - termination/truncation: never (for this test)
    """
    metadata = {}

    def __init__(
            self, 
            xml_path: str, 
            ry_cfg: ry.Config, 
            tau_step: float = 0.05, # garbage 
            tau_sim: float = 0.001  # garbage
            ):
        
        super().__init__()
        self.sim = MjSim(xml_path, ry_cfg, use_mj_viewer=False, tau_sim=tau_sim)
        self.tau_step = tau_step
        self.spline_times = np.array([2.0 * self.tau_step], dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.sim.ctrl_dim,), dtype=np.float32)
        obs_dim = self.sim.data.qpos.size + self.sim.data.qvel.size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _obs(self):
        return np.concatenate([self.sim.data.qpos.copy(), self.sim.data.qvel.copy()]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetSplineRef(ctrl_time=0.0)
        return self._obs(), {}  # doesn't matter

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        spline_points = action.reshape(1, -1)

        # MjSim’s spline-based stepping logic
        self.sim.setSplineRef(spline_points, times=self.spline_times)
        self.sim.step(self.tau_step)

        obs = self._obs()
        reward = np.float32(0.0)
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        # subprocess cleanup
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim
            self.sim = None


def _make_ry_config_from_yaml(scene_yaml: str):
    # Import inside helper to reduce parent-process side effects
    import robotic as ry
    C = ry.Config()
    C.addFile(scene_yaml)
    return C


def test_async_vectorenv_succeeds_if_ry_config_created_in_worker():
    """Positive control"""
    
    scene_yaml = "sample/twoFingers.yml"
    xml_path = "sample/twoFingers.xml"

    def thunk():
        cfg = _make_ry_config_from_yaml(scene_yaml)  # created in worker
        return SimpleGymEnv(xml_path=xml_path, ry_cfg=cfg)

    vec = None
    try:
        vec = gym.vector.AsyncVectorEnv([thunk, thunk], context="spawn")
        obs, info = vec.reset(seed=0)
        assert obs.shape[0] == 2  # num envs

        actions = np.stack([vec.single_action_space.sample() for _ in range(vec.num_envs)], axis=0)
        obs, rew, term, trunc, info = vec.step(actions)

        assert obs.shape[0] == 2
        assert rew.shape == (2,)
    finally:
        vec.close()


def test_async_vectorenv_fails_if_ry_config_created_in_parent():
    """Negative control"""
    
    scene_yaml = "sample/twoFingers.yml"
    xml_path = "sample/twoFingers.xml"

    cfg = _make_ry_config_from_yaml(scene_yaml)  # created in parent https://vscode.dev/github/LIS-TU-Berlin/25-diverseKomoManip/blob/main/src/diversemanip/Workspace.py#L80

    def thunk():
        return SimpleGymEnv(xml_path=xml_path, ry_cfg=cfg)

    vec = None
    try:
        with pytest.raises(Exception, match="pickle|cannot|spawn"):
            vec = gym.vector.AsyncVectorEnv([thunk, thunk], context="spawn")
            obs, info = vec.reset(seed=0)
            assert obs.shape[0] == 2  # num envs

            actions = np.stack([vec.single_action_space.sample() for _ in range(vec.num_envs)], axis=0)
            obs, rew, term, trunc, info = vec.step(actions)

            assert obs.shape[0] == 2
            assert rew.shape == (2,)
    finally:
        if vec is not None:
            vec.close()

    
