import time
import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3.common.buffers import ReplayBuffer


def sac_train_loop(
    device,
    config,
    envs,
    actor,
    critic,
    critic_target,
    actor_optimizer,
    critic_optimizer,
    logger,
):
    replay_buffer = ReplayBuffer(
        config.train.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        config.train.n_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    recent_ep_rewards = []
    start_time = time.time()

    if config.sac.autotune:
        target_entropy = -config.sac.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.action_space.n)
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = torch.optim.Adam([log_alpha], lr=config.optim.alpha_lr, eps=1e-4)
    else:
        alpha = config.sac.alpha

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(config.train.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < config.sac.learning_starts:
            actions = np.array(
                [envs.action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            obs_t = torch.from_numpy(obs).to(device)
            actions, _, _ = actor(obs_t)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        # dones = terminated
        # dones = [term | trunc for term, trunc in zip(terminated, truncated)]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                # info = info["final_info"]
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

        if global_step > config.sac.learning_starts:
            if global_step % config.sac.update_frequency == 0:
                data = replay_buffer.sample(config.train.batch_size)
                qf_loss, actor_loss, action_probs, log_pi = sac_loss(
                    data, actor, critic, critic_target, config.sac.gamma, alpha
                )

                critic_optimizer.zero_grad()
                qf_loss.backward()
                critic_optimizer.step()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if config.sac.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (
                        action_probs.detach()
                        * (-log_alpha * (log_pi + target_entropy).detach())
                    ).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % config.sac.target_network_frequency == 0:
                for param, target_param in zip(
                    critic.parameters(), critic_target.parameters()
                ):
                    target_param.data.copy_(
                        config.sac.tau * param.data
                        + (1 - config.sac.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                # actor_grad_norm = 0
                # for p in actor.parameters():
                #     gnorm = p.grad.data.cpu().norm()
                #     actor_grad_norm += gnorm

                # logger.add_scalar("grads/actor_grad_norm", np.mean(actor_grad_norm), global_step)
                logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                logger.add_scalar("losses/alpha", alpha, global_step)
                logger.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if config.sac.autotune:
                    logger.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
    envs.close()
    logger.close()

    return recent_ep_rewards


# @torch.compile
def sac_loss(data, actor, critic, critic_target, gamma, alpha):
    # CRITIC training
    with torch.no_grad():
        _, next_state_log_pi, next_state_action_probs = actor(data.next_observations)
        qf1_next_target, qf2_next_target = critic_target(data.next_observations)
        # we can use the action probabilities instead of MC sampling to estimate the expectation
        min_qf_next_target = next_state_action_probs * (
            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        )
        # adapt Q-target for discrete Q-function
        min_qf_next_target = min_qf_next_target.sum(dim=1)
        next_q_value = (
            data.rewards.flatten()
            + (1 - data.dones.flatten()) * gamma * min_qf_next_target
        )

    # use Q-values only for the taken actions
    qf1_values, qf2_values = critic(data.observations)
    qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
    qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # ACTOR training
    _, log_pi, action_probs = actor(data.observations)
    min_qf_values = torch.min(qf1_values, qf2_values).detach()
    # no need for reparameterization, the expectation can be calculated for discrete actions
    actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

    return qf_loss, actor_loss, action_probs, log_pi
