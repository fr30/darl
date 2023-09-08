# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import random
import time
import gymnasium as gym
import numpy as np
import minigrid
import hydra
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from src.models import Critic, Actor, PixelEncoder
from src.sac import sac_train_loop

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = minigrid.wrappers.RGBImgObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        # env = gym.wrappers.TransformObservation(env, lambda x: x / 255.0)
        # env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

@hydra.main(version_base=None, config_path="cfg", config_name="default")
def train(config):
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    logger = SummaryWriter(f"runs/{run_name}")
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    # env setup
    envs = []
    for i in range(config.n_envs):
        if i == 0:
            envs.append(make_env(config.env_id, config.seed + 100 * i, 0, config.capture_video, run_name))
        else:
            envs.append(make_env(config.env_id, config.seed + 100 * i, 0, False, run_name))
    envs = gym.vector.AsyncVectorEnv(envs)
    envs = gym.wrappers.VectorListInfo(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n

    encoder = PixelEncoder(obs_shape).to(device)
    actor = Actor(encoder, num_actions).to(device)
    critic = Critic(encoder, num_actions).to(device)
    critic_target = Critic(encoder, num_actions).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    critic_optimizer = optim.Adam(list(critic.parameters()), lr=config.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config.policy_lr, eps=1e-4)

    summary(actor, obs_shape)
    summary(critic, obs_shape)

    # results = sac_train_loop(device, config, envs, actor, critic, critic_target, actor_optimizer, critic_optimizer, logger)

    # score = sum(results) / len(results)
    # print(f'SCORE: {score}')
    # return score

if __name__ == "__main__":
    train()
