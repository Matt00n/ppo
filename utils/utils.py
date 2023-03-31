import glob
import os
import warnings
import platform
import random
import re
from collections import deque
from itertools import zip_longest
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union, Callable, NamedTuple

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from gym import spaces

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .loggers import Logger, configure


TensorDict = Dict[Union[str, int], th.Tensor]
Schedule = Callable[[float], float]

class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: Pytorch optimizer
    :param learning_rate: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_schedule_fn(value_schedule: Union[Schedule, float]) -> Schedule:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return: Linear schedule function.
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_fn(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: constant value
    :return: Constant schedule function.
    """

    def func(_):
        return val

    return func


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: Path to the log folder containing several runs.
    :param log_name: Name of the experiment. Each run is stored
        in a folder named ``log_name_1``, ``log_name_2``, ...
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def configure_logger(
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """
    Configure the logger's outputs.

    :param verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param tb_log_name: tensorboard log
    :param reset_num_timesteps:  Whether the ``num_timesteps`` attribute is reset or not.
        It allows to continue a previous learning curve (``reset_num_timesteps=False``)
        or start from t=0 (``reset_num_timesteps=True``, the default).
    :return: The logger object
    """
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        if verbose >= 1:
            format_strings = ["stdout", "tensorboard"]
        else:
            format_strings = ["tensorboard"]
    elif verbose == 0:
        format_strings = [""]
    return configure(save_path, format_strings=format_strings)


# def check_for_correct_spaces(env: GymEnv, observation_space: spaces.Space, action_space: spaces.Space) -> None:
#     """
#     Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
#     spaces match after loading the model with given env.
#     Checked parameters:
#     - observation_space
#     - action_space

#     :param env: Environment to check for valid spaces
#     :param observation_space: Observation space to check against
#     :param action_space: Action space to check against
#     """
#     if observation_space != env.observation_space:
#         raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
#     if action_space != env.action_space:
#         raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")


# def check_shape_equal(space1: spaces.Space, space2: spaces.Space) -> None:
#     """
#     If the spaces are Box, check that they have the same shape.

#     If the spaces are Dict, it recursively checks the subspaces.

#     :param space1: Space
#     :param space2: Other space
#     """
#     if isinstance(space1, spaces.Dict):
#         assert isinstance(space2, spaces.Dict), "spaces must be of the same type"
#         assert space1.spaces.keys() == space2.spaces.keys(), "spaces must have the same keys"
#         for key in space1.spaces.keys():
#             check_shape_equal(space1.spaces[key], space2.spaces[key])
#     elif isinstance(space1, spaces.Box):
#         assert space1.shape == space2.shape, "spaces must have the same shape"


def is_vectorized_box_observation(observation: np.ndarray, observation_space: spaces.Box) -> bool:
    """
    For box observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + f"Box environment, please use {observation_space.shape} "
            + "or (n_env, {}) for the observation shape.".format(", ".join(map(str, observation_space.shape)))
        )


def is_vectorized_discrete_observation(observation: Union[int, np.ndarray], observation_space: spaces.Discrete) -> bool:
    """
    For discrete observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if isinstance(observation, int) or observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
        return False
    elif len(observation.shape) == 1:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            + "Discrete environment, please use () or (n_env,) for the observation shape."
        )


def is_vectorized_multidiscrete_observation(observation: np.ndarray, observation_space: spaces.MultiDiscrete) -> bool:
    """
    For multidiscrete observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == (len(observation_space.nvec),):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
            + f"environment, please use ({len(observation_space.nvec)},) or "
            + f"(n_env, {len(observation_space.nvec)}) for the observation shape."
        )


def is_vectorized_multibinary_observation(observation: np.ndarray, observation_space: spaces.MultiBinary) -> bool:
    """
    For multibinary observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif len(observation.shape) == len(observation_space.shape) + 1 and observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
            + f"environment, please use {observation_space.shape} or "
            + f"(n_env, {observation_space.n}) for the observation shape."
        )


def is_vectorized_dict_observation(observation: np.ndarray, observation_space: spaces.Dict) -> bool:
    """
    For dict observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    # We first assume that all observations are not vectorized
    all_non_vectorized = True
    for key, subspace in observation_space.spaces.items():
        # This fails when the observation is not vectorized
        # or when it has the wrong shape
        if observation[key].shape != subspace.shape:
            all_non_vectorized = False
            break

    if all_non_vectorized:
        return False

    all_vectorized = True
    # Now we check that all observation are vectorized and have the correct shape
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape[1:] != subspace.shape:
            all_vectorized = False
            break

    if all_vectorized:
        return True
    else:
        # Retrieve error message
        error_msg = ""
        try:
            is_vectorized_observation(observation[key], observation_space.spaces[key])
        except ValueError as e:
            error_msg = f"{e}"
        raise ValueError(
            f"There seems to be a mix of vectorized and non-vectorized observations. "
            f"Unexpected observation shape {observation[key].shape} for key {key} "
            f"of type {observation_space.spaces[key]}. {error_msg}"
        )


def is_vectorized_observation(observation: Union[int, np.ndarray], observation_space: spaces.Space) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """

    is_vec_obs_func_dict = {
        spaces.Box: is_vectorized_box_observation,
        spaces.Discrete: is_vectorized_discrete_observation,
        spaces.MultiDiscrete: is_vectorized_multidiscrete_observation,
        spaces.MultiBinary: is_vectorized_multibinary_observation,
        spaces.Dict: is_vectorized_dict_observation,
    }

    for space_type, is_vec_obs_func in is_vec_obs_func_dict.items():
        if isinstance(observation_space, space_type):
            return is_vec_obs_func(observation, observation_space)
    else:
        # for-else happens if no break is called
        raise ValueError(f"Error: Cannot determine if the observation is vectorized with the space type {observation_space}.")


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def get_parameters_by_name(model: th.nn.Module, included_names: Iterable[str]) -> List[th.Tensor]:
    """
    Extract parameters from the state dict of ``model``
    if the name contains one of the strings in ``included_names``.

    :param model: the model where the parameters come from.
    :param included_names: substrings of names to include.
    :return: List of parameters values (Pytorch tensors)
        that matches the queried names.
    """
    return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]


# def zip_strict(*iterables: Iterable) -> Iterable:
#     r"""
#     ``zip()`` function but enforces that iterables are of equal length.
#     Raises ``ValueError`` if iterables not of equal length.
#     Code inspired by Stackoverflow answer for question #32954486.

#     :param \*iterables: iterables to ``zip()``
#     """
#     # As in Stackoverflow #32954486, use
#     # new object for "empty" in case we have
#     # Nones in iterable.
#     sentinel = object()
#     for combo in zip_longest(*iterables, fillvalue=sentinel):
#         if sentinel in combo:
#             raise ValueError("Iterables have different lengths")
#         yield combo


# def polyak_update(
#     params: Iterable[th.Tensor],
#     target_params: Iterable[th.Tensor],
#     tau: float,
# ) -> None:
#     """
#     Perform a Polyak average update on ``target_params`` using ``params``:
#     target parameters are slowly updated towards the main parameters.
#     ``tau``, the soft update coefficient controls the interpolation:
#     ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
#     The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
#     or a computation graph, reducing memory cost and improving performance.  We scale the target params
#     by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
#     params (in place).
#     See https://github.com/DLR-RM/stable-baselines3/issues/93

#     :param params: parameters to use to update the target params
#     :param target_params: parameters to update
#     :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
#     """
#     with th.no_grad():
#         # zip does not raise an exception if length of parameters does not match.
#         for param, target_param in zip_strict(params, target_params):
#             target_param.data.mul_(1 - tau)
#             th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def obs_as_tensor(
    obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device
) -> Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def should_collect_more_steps(
    train_freq: TrainFreq,
    num_collected_steps: int,
    num_collected_episodes: int,
) -> bool:
    """
    Helper used in ``collect_rollouts()`` of off-policy algorithms
    to determine the termination condition.

    :param train_freq: How much experience should be collected before updating the policy.
    :param num_collected_steps: The number of already collected steps.
    :param num_collected_episodes: The number of already collected episodes.
    :return: Whether to continue or not collecting experience
        by doing rollouts of the current policy.
    """
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return num_collected_steps < train_freq.frequency

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return num_collected_episodes < train_freq.frequency

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )


def get_system_info(print_info: bool = True) -> Tuple[Dict[str, str], str]:
    """
    Retrieve system and python env info for the current system.

    :param print_info: Whether to print or not those infos
    :return: Dictionary summing up the version for each relevant package
        and a formatted string.
    """
    env_info = {
        # In OS, a regex is used to add a space between a "#" and a number to avoid
        # wrongly linking to another issue on GitHub. Example: turn "#42" to "# 42".
        "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
        "Python": platform.python_version(),
        "PyTorch": th.__version__,
        "GPU Enabled": str(th.cuda.is_available()),
        "Numpy": np.__version__,
        "Gym": gym.__version__,
    }
    env_info_str = ""
    for key, value in env_info.items():
        env_info_str += f"- {key}: {value}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
    

def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        if type(observation_space.n) in [tuple, list, np.ndarray]:
            return tuple(observation_space.n)
        else:
            return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
    

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.
    Used by the ``FlattenExtractor`` to compute the input shape.
    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)
    

def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).
    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).
    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.
    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    :return:
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if check_dtype and observation_space.dtype != np.uint8:
            return False

        # Check the value range
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
        if check_bounds and incorrect_bounds:
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # GrayScale, RGB, RGBD
        return n_channels in [1, 3, 4]
    return False


def maybe_transpose(observation: np.ndarray, observation_space: spaces.Space) -> np.ndarray:
    """
    Handle the different cases for images as PyTorch use channel first format.
    :param observation:
    :param observation_space:
    :return: channel first observation if observation is an image
    """
    # Avoid circular import
    from stable_baselines3.common.vec_env import VecTransposeImage

    if is_image_space(observation_space):
        if not (observation.shape == observation_space.shape or observation.shape[1:] == observation_space.shape):
            # Try to re-order the channels
            transpose_obs = VecTransposeImage.transpose_image(observation)
            if transpose_obs.shape == observation_space.shape or transpose_obs.shape[1:] == observation_space.shape:
                observation = transpose_obs
    return observation


def preprocess_obs(
    obs: th.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.
    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key], normalize_images=normalize_images)
        return preprocessed_obs

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")

def check_shape_equal(space1: spaces.Space, space2: spaces.Space) -> None:
    """
    If the spaces are Box, check that they have the same shape.
    If the spaces are Dict, it recursively checks the subspaces.
    :param space1: Space
    :param space2: Other space
    """
    if isinstance(space1, spaces.Dict):
        assert isinstance(space2, spaces.Dict), "spaces must be of the same type"
        assert space1.spaces.keys() == space2.spaces.keys(), "spaces must have the same keys"
        for key in space1.spaces.keys():
            check_shape_equal(space1.spaces[key], space2.spaces[key])
    elif isinstance(space1, spaces.Box):
        assert space1.shape == space2.shape, "spaces must have the same shape"

