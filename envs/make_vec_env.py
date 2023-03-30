import numpy as np
import gym
from .subproc_vec_env import SubprocVecEnv
from .monitor_wrapper import Monitor

def make_vec_env(
    env_id: str,  
    num_envs: int,
    seed: int = 42, 
    clip_action: bool = False, 
    norm_obs: bool = False,
    clip_obs: bool = False,
    norm_rew: bool = False,
    clip_rew: bool = False, 
    monitor: bool = True,
    **env_kwargs,
):
    """
    Creates a vectorized environment with specified wrappers.

    :param env_id: The id (str) of the gym environment to be created.
    :param num_envs: The number of parallel environments.
    :param seed: Random seed.
    :param clip_action: Apply wrapper that clips actions to valid range (box action space only).
    :param norm_obs: Apply wrapper that normalizes observations.
    :param clip_obs: Apply wrapper that clips observations.
    :param norm_rew: Apply wrapper that normalizes rewards.
    :param clip_rew: Apply wrapper that clips rewards.
    :param monitor: Apply monitor wrapper used to log results (episode returns etc.).

    :return: The vectorized environment and corresponding environment specs in
        form of a dictionary containing action and observation space.
    """
    def thunk(i):
        env = gym.make(env_id, **env_kwargs)
        if clip_action:
            env = gym.wrappers.ClipAction(env)
        if norm_obs:
            env = gym.wrappers.NormalizeObservation(env)
        if clip_obs:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if norm_rew:
            env = gym.wrappers.NormalizeReward(env)
        if clip_rew:
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if monitor:
            env = Monitor(env)
        env.action_space.seed(seed + i)
        env.observation_space.seed(seed + i)
        return env
    
    sample_env = gym.make(env_id, **env_kwargs)
    if clip_action:
        sample_env = gym.wrappers.ClipAction(sample_env)
    if norm_obs:
        sample_env = gym.wrappers.NormalizeObservation(sample_env)
    if clip_obs:
        sample_env = gym.wrappers.TransformObservation(sample_env, lambda obs: np.clip(obs, -10, 10))
    if norm_rew:
        sample_env = gym.wrappers.NormalizeReward(sample_env)
    if clip_rew:
        sample_env = gym.wrappers.TransformReward(sample_env, lambda reward: np.clip(reward, -10, 10))
    if monitor:
        sample_env = Monitor(sample_env)
    environment_spec = {'observations': sample_env.observation_space, 
                    'actions': sample_env.action_space}

    return SubprocVecEnv([lambda: thunk(i) for i in range(num_envs)]), environment_spec