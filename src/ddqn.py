import numpy as np
import random
import time
import torch
import torch.nn.functional as F

from stable_baselines3.common.buffers import ReplayBuffer


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def dqn_train_loop(
    device, config, envs, qnetwork, qnetwork_optimizer, target_qnetwork, logger
):
    replay_buffer = ReplayBuffer(
        config.train.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        n_envs=config.train.n_envs,
    )
    recent_ep_rewards = []
    start_time = time.time()

    obs = envs.reset()
    for global_step in range(config.train.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            config.dqn.start_eps,
            config.dqn.end_eps,
            config.dqn.exploration_fraction * config.train.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = qnetwork(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, dones, infos = envs.step(actions)

        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                logger.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                logger.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                if len(recent_ep_rewards) < 100:
                    recent_ep_rewards += [info["episode"]["r"]]
                else:
                    recent_ep_rewards.pop(0)
                    recent_ep_rewards.append(info["episode"]["r"])
                break

        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        replay_buffer.add(obs, real_next_obs, actions, rewards, dones, infos)
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > config.dqn.learning_starts:
            if global_step % config.dqn.train_frequency == 0:
                data = replay_buffer.sample(config.train.batch_size)
                loss = dqn_loss(data, qnetwork, target_qnetwork, config.dqn.gamma)

                if global_step % 100 == 0:
                    logger.add_scalar("losses/td_loss", loss, global_step)
                    logger.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                qnetwork_optimizer.zero_grad()
                loss.backward()
                qnetwork_optimizer.step()

            # update target network
            if global_step % config.dqn.target_network_frequency == 0:
                for target_network_param, qnetwork_param in zip(
                    target_qnetwork.parameters(), qnetwork.parameters()
                ):
                    target_network_param.data.copy_(
                        config.dqn.tau * qnetwork_param.data
                        + (1.0 - config.dqn.tau) * target_network_param.data
                    )

    return recent_ep_rewards


def ddqn_loss(data, qnetwork, target_qnetwork, gamma):
    target_best_action = target_qnetwork(data.next_observations).argmax(dim=1).detach()
    new_qval = qnetwork(data.observations).gather(1, target_best_action).squeeze()
    td_target = data.rewards.flatten() + gamma * new_qval * (1 - data.dones.flatten())
    old_val = qnetwork(data.observations).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)
    return loss
