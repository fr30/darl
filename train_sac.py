# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import random
import time
import gymnasium as gym
import numpy as np
import hydra
import torch
import torch.optim as optim

from src.models import Critic, Actor, PixelEncoder
from src.sac import sac_train_loop
from src.utils import make_envs
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="cfg", config_name="minigrid_grayscale")
def train(config):
    run_name = f"{config.env.id}__{config.meta.exp_name}__{config.meta.seed}__{int(time.time())}"
    if config.meta.track:
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config.meta.seed)
    np.random.seed(config.meta.seed)
    torch.manual_seed(config.meta.seed)
    torch.backends.cudnn.deterministic = config.meta.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.meta.cuda else "cpu"
    )

    envs = make_envs(config, run_name)
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    obs_shape = envs.observation_space.shape
    num_actions = envs.action_space.n

    encoder = PixelEncoder(
        channels=obs_shape[0],
        img_size=config.encoder.img_size,
        crop=config.encoder.crop,
        num_filters=config.encoder.num_filters,
        out_features=config.encoder.out_features,
    ).to(device)
    actor = Actor(
        encoder=encoder,
        num_actions=num_actions,
        num_features=config.actor.hidden_features,
    ).to(device)
    critic = Critic(
        encoder=encoder,
        num_actions=num_actions,
        num_features=config.critic.hidden_features,
    ).to(device)
    critic_target = Critic(
        encoder=encoder,
        num_actions=num_actions,
        num_features=config.critic.hidden_features,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    critic_optimizer = optim.Adam(
        list(critic.parameters()), lr=config.optim.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()), lr=config.optim.policy_lr, eps=1e-4
    )

    results = sac_train_loop(
        device,
        config,
        envs,
        actor,
        critic,
        critic_target,
        actor_optimizer,
        critic_optimizer,
        logger,
    )

    score = sum(results) / len(results)
    envs.close()
    logger.close()
    print(f"SCORE: {score}")
    return score


if __name__ == "__main__":
    train()
