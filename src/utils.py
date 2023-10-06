import torch
import torch.nn as nn
import minigrid
import gymnasium as gym

from stable_baselines3.common.vec_env import (
    VecFrameStack,
    DummyVecEnv,
    SubprocVecEnv,
    VecTransposeImage,
)


def make_envs(config, run_name):
    envs = [_create_env(config, i, run_name) for i in range(config.train.n_envs)]
    if config.train.n_envs > 1:
        envs = SubprocVecEnv(envs)
    else:
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


def _create_env(config, idx, run_name):
    seed = config.meta.seed + 100 * idx

    def thunk():
        if config.env.capture_video:
            env = gym.make(config.env.id, render_mode="rgb_array")
        else:
            env = gym.make(config.env.id)
        env = minigrid.wrappers.RGBImgObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.env.capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ResizeObservation(env, (100, 100))
        if config.env.grayscale:
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
