import gymnasium as gym
import hydra
import numpy as np
import random
import time
import torch
import torch.optim as optim

from src.models import QNetwork
from src.dqn import dqn_train_loop
from src.utils import make_envs
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="cfg", config_name="dqn_minigrid_grayscale")
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

    qnetwork = QNetwork(
        channels=obs_shape[0], img_size=config.encoder.img_size, num_actions=num_actions
    ).to(device)
    qnetwork_optimizer = optim.Adam(qnetwork.parameters(), lr=config.optim.q_lr)
    target_qnetwork = QNetwork(
        channels=obs_shape[0], img_size=config.encoder.img_size, num_actions=num_actions
    ).to(device)
    target_qnetwork.load_state_dict(qnetwork.state_dict())
    results = dqn_train_loop(
        device, config, envs, qnetwork, qnetwork_optimizer, target_qnetwork, logger
    )
    score = sum(results) / len(results)
    envs.close()
    logger.close()
    print(f"SCORE: {score}")
    return score


if __name__ == "__main__":
    train()
