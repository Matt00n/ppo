# Proximal Policy Optimization (PPO)

This implementation of <a href="https://arxiv.org/abs/1707.06347">PPO</a> is a slightly condensed version of the <a href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html">implementation by Stable Baselines 3</a> and reuses most of its code. The main added features of this implementation are:

* Supports the new gym API (versions >= 0.26 and gymnasium), where `environment.step` returns 5 values instead of 4 (`terminated` and `truncated` flags instead of `done`).
* For increased flexibility, the environment loop is now longer contained within the PPO class but should be coded outside, see the example below.

---
## Usage example

```
import time
from collections import deque
import sys
import numpy as np
import torch as th
import gym
from ppo import PPO, obs_as_tensor, configure_logger, safe_mean, make_vec_env

num_envs = 4 # number of vectorized environments
env_id = "CartPole-v1"
seed = 42 

env, env_spec = make_vec_env(env_id=env_id,  
                             num_envs=num_envs,
                             seed=seed, 
                             )

model = PPO("MlpPolicy", env_spec, num_envs)

total_timesteps = 1e5
log_interval = 1
tb_log_name = "PPO"
ep_info_buffer = deque(maxlen=100)
n_rollout_steps = model.unroll_length
num_timesteps = 0
iteration = 0
start_time = time.time_ns()

_last_obs, _ = env.reset() 
_last_episode_starts = np.ones((num_envs,), dtype=bool)
if not model._custom_logger:
    model.set_logger(configure_logger(model.verbose, model.tensorboard_log, tb_log_name, True))

while num_timesteps < total_timesteps:
    model.policy.set_training_mode(False) # Switch to eval mode (this affects batch norm / dropout)
    n_steps = 0
    model.rollout_buffer.reset()
    if model.use_sde:
        model.policy.reset_noise(num_envs)

    while n_steps < n_rollout_steps:
        if model.use_sde and model.sde_sample_freq > 0 and n_steps % model.sde_sample_freq == 0:
            model.policy.reset_noise(num_envs)
        with th.no_grad():
            obs_tensor = obs_as_tensor(_last_obs, model.device)
            actions, values, log_probs = model.policy(obs_tensor)
        actions = actions.cpu().numpy()
        clipped_actions = actions
        if isinstance(env_spec['actions'], gym.spaces.Box):
            clipped_actions = np.clip(actions, env_spec['actions'].low, env_spec['actions'].high)

        new_obs, rewards, term, trunc, infos = env.step(clipped_actions)
        dones = np.logical_or(term, trunc)

        num_timesteps += num_envs
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            if maybe_ep_info is not None:
                ep_info_buffer.extend([maybe_ep_info])
        n_steps += 1

        if isinstance(env_spec['actions'], gym.spaces.Discrete):
            actions = actions.reshape(-1, 1) # Reshape in case of discrete action

        # bootstrap if episodes truncated
        for idx, truncated in enumerate(trunc):
            if (
                truncated
                and infos[idx].get("terminal_observation") is not None
            ):
                terminal_obs = model.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = model.policy.predict_values(terminal_obs)[0]
                rewards[idx] += model.gamma * terminal_value

        model.rollout_buffer.add(_last_obs, actions, rewards, _last_episode_starts, values, log_probs)
        _last_obs = new_obs
        _last_episode_starts = dones

    with th.no_grad():
        values = model.policy.predict_values(obs_as_tensor(new_obs, model.device)) # Compute value for the last timestep
    model.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    iteration += 1
    model.update_current_progress_remaining(num_timesteps, total_timesteps)

    if log_interval is not None and iteration % log_interval == 0:
        time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
        fps = int(num_timesteps / time_elapsed)
        model.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
            model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in ep_info_buffer]))
            model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in ep_info_buffer]))
        model.logger.record("time/fps", fps)
        model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        model.logger.record("time/total_timesteps", num_timesteps, exclude="tensorboard")
        model.logger.dump(step=num_timesteps)

    model.train()
env.close()
```
