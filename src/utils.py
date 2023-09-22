import torch
import torch.nn as nn
import minigrid
import gymnasium as gym

from enum import Enum
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    DummyVecEnv,
    VecTransposeImage,
)


class EnvType(Enum):
    MINIGRID = 0
    ATARI = 1
    MUJOCO = 2


def make_envs(config, run_name, env_type=EnvType.MINIGRID):
    create_env_fn = None
    match env_type:
        case EnvType.MINIGRID:
            create_env_fn = _create_minigrid_env
        case EnvType.ATARI:
            create_env_fn = _create_atari_env
        case EnvType.MUJOCO:
            create_env_fn = _create_mujoco_env
    envs = [create_env_fn(config, i, run_name) for i in range(config.train.n_envs)]
    envs = DummyVecEnv(envs)
    envs = VecTransposeImage(envs)
    envs = VecFrameStack(envs, config.env.frame_stack)
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    return envs


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _create_atari_env(config, idx, run_name):
    seed = config.meta.seed + 100 * idx

    def thunk():
        if config.env.capture_video:
            env = gym.make(config.env.id, render_mode="rgb_array")
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(config.env.id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (100, 100))
        if config.env.grayscale:
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def _create_minigrid_env(config, idx, run_name):
    seed = config.meta.seed + 100 * idx

    def thunk():
        if config.env.capture_video:
            env = gym.make(config.env.id, render_mode="rgb_array")
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(config.env.id)
        env = minigrid.wrappers.RGBImgObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (100, 100))
        if config.env.grayscale:
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def _create_mujoco_env(config, idx, run_name):
    seed = config.meta.seed + 100 * idx

    def thunk():
        if config.env.capture_video:
            env = gym.make(config.env.id, render_mode="rgb_array")
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(config.env.id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.env.grayscale:
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
