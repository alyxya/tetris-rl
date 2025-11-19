"""
PPO training script for Tetris CNN policy using PufferLib
Based on CleanRL's PPO implementation
"""

import os
import random
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

import pufferlib.vector
from pufferlib.ocean.tetris import tetris
from tetris_cnn_policy import TetrisCNNPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for Tetris")

    # Experiment settings
    parser.add_argument("--exp-name", type=str, default="tetris_cnn_ppo", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(int(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(int(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(int(x)), default=False, nargs="?", const=True, help="track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="tetris-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Algorithm parameters
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", type=lambda x: bool(int(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", type=lambda x: bool(int(x)), default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(int(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # CNN architecture
    parser.add_argument("--cnn-channels", type=str, default="32,64,64")
    parser.add_argument("--cnn-kernel-sizes", type=str, default="3,3,3")
    parser.add_argument("--cnn-strides", type=str, default="1,1,1")
    parser.add_argument("--mlp-hidden-dims", type=str, default="256,128")

    return parser.parse_args()


def make_env(buf=None, seed=None):
    """Create a Tetris environment"""
    env = tetris.Tetris()
    return env


if __name__ == "__main__":
    args = parse_args()
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = args.total_timesteps // batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # Setup logging
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create vectorized environments using PufferLib
    envs = pufferlib.vector.make(
        make_env,
        backend=pufferlib.vector.Serial,  # Use Serial for debugging, Multiprocessing for speed
        num_envs=args.num_envs,
    )

    # Parse architecture parameters
    cnn_channels = [int(x) for x in args.cnn_channels.split(",")]
    cnn_kernel_sizes = [int(x) for x in args.cnn_kernel_sizes.split(",")]
    cnn_strides = [int(x) for x in args.cnn_strides.split(",")]
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden_dims.split(",")]

    # Create agent
    agent = TetrisCNNPolicy(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        cnn_channels=cnn_channels,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
        mlp_hidden_dims=mlp_hidden_dims,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs_storage = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Episode tracking
    episode_rewards = []
    episode_lengths = []

    print(f"Starting training for {num_iterations} iterations ({args.total_timesteps} total timesteps)")
    print(f"Using device: {device}")
    print(f"Policy parameters: {sum(p.numel() for p in agent.parameters()):,}")

    for iteration in range(1, num_iterations + 1):
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_storage[step] = value.flatten()
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            # Execute action in environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Track episode statistics
            if "episode" in infos:
                for ep_info in infos["episode"]:
                    if ep_info is not None:
                        episode_rewards.append(ep_info["r"])
                        episode_lengths.append(ep_info["l"])

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_action_and_value(next_obs)[3]
            advantages = torch.zeros_like(rewards_storage).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    nextvalues = values_storage[t + 1]
                delta = rewards_storage[t] + args.gamma * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_storage

        # Flatten the batch
        b_obs = obs_storage.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            # Episode statistics
            if len(episode_rewards) > 0:
                writer.add_scalar("charts/episodic_return", np.mean(episode_rewards[-100:]), global_step)
                writer.add_scalar("charts/episodic_length", np.mean(episode_lengths[-100:]), global_step)

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
        else:
            sps = int(global_step / (time.time() - start_time))

        # Print progress
        if iteration % 10 == 0:
            avg_return = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) > 0 else 0
            print(f"Iter {iteration}/{num_iterations} | Steps: {global_step:,} | "
                  f"SPS: {sps} | Avg Return: {avg_return:.2f} | Avg Length: {avg_length:.0f} | "
                  f"Value Loss: {v_loss.item():.4f} | Policy Loss: {pg_loss.item():.4f}")

        # Save checkpoint
        if iteration % 100 == 0:
            checkpoint_path = f"checkpoints/{run_name}"
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, f"{checkpoint_path}/checkpoint_{iteration}.pt")
            print(f"Saved checkpoint to {checkpoint_path}/checkpoint_{iteration}.pt")

    # Save final model
    final_path = f"models/{run_name}"
    os.makedirs(final_path, exist_ok=True)
    torch.save(agent.state_dict(), f"{final_path}/final_model.pt")
    print(f"\nTraining complete! Final model saved to {final_path}/final_model.pt")

    envs.close()
    if writer is not None:
        writer.close()
