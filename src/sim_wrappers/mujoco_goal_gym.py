from gymnasium_robotics.core import GoalEnv
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from sim_wrappers import MujocoGym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvIndices
from typing import Any

class MujocoGoalGym(GoalEnv):
    def __init__(self, org_env: MujocoGym):
        self.org_env = org_env
        self.action_space = org_env.action_space
        self.observation_space = spaces.Dict(
            spaces={
                "observation": org_env.observation_space,
                "achieved_goal": spaces.Box(-2., +2., shape=org_env.goal_feat.shape, dtype=np.float32),
                "desired_goal": spaces.Box(-2., +2., shape=org_env.goal_feat.shape, dtype=np.float32)
            }

        )
    
    def reset(self, seed=None, options=None):
        observation, info = self.org_env.reset(seed, options)
        achieved_goal = info['feature'].reshape(-1)
        desired_goal = self.org_env.goal_feat.reshape(-1)

        obs_dict = {
                    "observation": observation,
                    "achieved_goal": achieved_goal,
                    "desired_goal": desired_goal,
                }
        
        return obs_dict, info

    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.org_env.step(action)
        achieved_goal = info['feature'].reshape(-1)
        desired_goal = self.org_env.goal_feat.reshape(-1)

        obs_dict = {
                    "observation": obs_next,
                    "achieved_goal": achieved_goal,
                    "desired_goal": desired_goal,
                }
        
        alt_reward = self.compute_reward(achieved_goal, desired_goal, info)
        assert reward==alt_reward

        return obs_dict, reward, terminated, truncated, info

    def compute_reward(    
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict
        ) -> float:

        self.org_env.goal_feat = np.atleast_2d(desired_goal)
        reward = self.org_env.reward_fct(None, np.atleast_2d(achieved_goal))

        return reward


class SB_VecGym(VecEnv):
    def __init__(self, org_env: MujocoGym):
        self.org_env = org_env
        self.num_envs = org_env.num_scenes
        self.action_space = org_env.single_action_space
        self.observation_space = org_env.single_observation_space

    def reset(self) -> VecEnvObs:
        self.org_env.reset()
        return self.org_env.observation

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        observation, reward, terminated, truncated, _ = self.org_env.step(actions)

        done = np.logical_or(terminated, truncated)
        n = self.org_env.num_scenes
        infos = [{}]*n
        for i in range(n):
            infos[i]["TimeLimit.truncated"] = truncated[i] and not terminated[i]
            if done[i]:
                infos[i]["terminal_observation"] = observation[i]
        
        self.org_env.auto_reset()

        return self.org_env.observation, reward, done, infos

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()
    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()
    def close(self) -> None:
        raise NotImplementedError()
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        raise NotImplementedError()
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError()
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        raise NotImplementedError()
    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        raise NotImplementedError()


class VectorGym(gym.Env):
    def __init__(
        self,
        env_fns,
    ):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)

        self.single_action_space = self.envs[0].action_space
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.single_observation_space = self.envs[0].observation_space
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        assert len(self.single_observation_space.shape)==1


        self._observations = np.zeros((self.num_envs, self.single_observation_space.shape[0]), dtype=np.float64)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert (
            len(seed) == self.num_envs
        ), f"If seeds are passed as a list the length must match num_envs={self.num_envs} but got length={len(seed)}."

        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            self._observations[i], env_info = env.reset(seed=single_seed, options=options)
            if i==0:
                info = env_info
        return np.copy(self._observations), info

    def step(
        self, actions
    ):
        # actions = iterate(self.action_space, actions)

        for i, action in enumerate(actions):
            (   self._observations[i],
                self._rewards[i],
                self._terminations[i],
                self._truncations[i],
                env_info ) = self.envs[i].step(action)

            if i==0:
                info = env_info
        return (
            np.copy(self._observations),
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            info,
        )
