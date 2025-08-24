import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.Gen.Networks import ForwardDynamicsModel, GenerativeReplayBuffer
from Models.SB3 import PPO
from Models.Strategy import sampling_strategy


class GenPPO(PPO):
  """
    Proximal Policy Optimization with Generative Replay Buffer and entropy regularization.
    """

  def __init__(
    self,
    entropy_coef=0.01,
    sampling_coef=0.5,
    buffer_capacity=10000,
    batch_size=64,
    normalize_advantage=True,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.entropy_coef = entropy_coef
    self.sampling_coef = sampling_coef
    self.batch_size = batch_size
    self.normalize_advantage = normalize_advantage
    self.forward_dynamics_model = ForwardDynamicsModel(observation_space=self.observation_space, action_space=self.action_space).to(self.device)
    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=buffer_capacity,
      synthetic_capacity=buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      batch_size=batch_size,
      device=self.device,
    )

  def _compute_relevance(self, transition):
    obs, action, _, _, _ = transition
    obs = obs.to(self.device)
    action = action.to(self.device)
    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(obs.unsqueeze(0), action.unsqueeze(0))
    curiosity_score = 0.5 * th.norm(pred_h_next - h_s, p=2).item() ** 2
    return curiosity_score

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)
    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    # Collect rollout data
    rollout_data_list = list(self.rollout_buffer.get())
    on_policy_obs = th.cat([rd.observations for rd in rollout_data_list])
    on_policy_actions = th.cat([rd.actions for rd in rollout_data_list])
    on_policy_returns = th.cat([rd.returns for rd in rollout_data_list])
    on_policy_advantages = th.cat([rd.advantages for rd in rollout_data_list])
    on_policy_old_log_prob = th.cat([rd.old_log_prob for rd in rollout_data_list])

    # Add to replay buffer
    for i in range(on_policy_obs.shape[0]):
      transition = (
        on_policy_obs[i].cpu(),
        on_policy_actions[i].cpu(),
        on_policy_returns[i].cpu(),
        on_policy_advantages[i].cpu(),
        on_policy_old_log_prob[i].cpu(),
      )
      self.replay_buffer.add_real(transition)

    # Generate synthetic samples
    self.replay_buffer.generate_synthetic()

    # Determine replay sample size based on entropy
    distribution = self.policy.get_distribution(on_policy_obs)
    entropy_mean = distribution.entropy().mean().item()
    num_replay_samples = sampling_strategy(
      entropy=entropy_mean,
      sampling_coef=self.sampling_coef,
      min_samples=0,
      max_samples=self.batch_size,
    )

    if num_replay_samples > 0:
      replay_samples = self.replay_buffer.sample(num_replay_samples)
      if replay_samples:
        (
          replay_obs,
          replay_actions,
          replay_returns,
          replay_advantages,
          replay_old_log_prob,
        ) = zip(*replay_samples)
        replay_obs = th.stack(replay_obs).to(self.device)
        replay_actions = th.stack(replay_actions).to(self.device)
        replay_returns = th.stack(replay_returns).to(self.device)
        replay_advantages = th.stack(replay_advantages).to(self.device)
        replay_old_log_prob = th.stack(replay_old_log_prob).to(self.device)
        on_policy_obs = th.cat([on_policy_obs, replay_obs])
        on_policy_actions = th.cat([on_policy_actions, replay_actions])
        on_policy_returns = th.cat([on_policy_returns, replay_returns])
        on_policy_advantages = th.cat([on_policy_advantages, replay_advantages])
        on_policy_old_log_prob = th.cat([on_policy_old_log_prob, replay_old_log_prob])

    if isinstance(self.action_space, spaces.Discrete):
      on_policy_actions = on_policy_actions.long().flatten()

    # Normalize advantages if specified
    if self.normalize_advantage and len(on_policy_advantages) > 1:
      on_policy_advantages = (on_policy_advantages - on_policy_advantages.mean()) / (on_policy_advantages.std() + 1e-8)

    # Train for n_epochs epochs
    continue_training = True

    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      for start_idx in range(0, len(on_policy_obs), self.batch_size):
        end_idx = min(start_idx + self.batch_size, len(on_policy_obs))
        batch_indices = th.arange(start_idx, end_idx)

        obs_batch = on_policy_obs[batch_indices]
        actions_batch = on_policy_actions[batch_indices]
        returns_batch = on_policy_returns[batch_indices]
        advantages_batch = on_policy_advantages[batch_indices]
        old_log_prob_batch = on_policy_old_log_prob[batch_indices]

        values, log_prob, entropy = self.policy.evaluate_actions(obs_batch, actions_batch)
        values = values.flatten()

        # Ratio between old and new policy
        ratio = th.exp(log_prob - old_log_prob_batch)

        # Clipped surrogate loss
        policy_loss_1 = advantages_batch * ratio
        policy_loss_2 = advantages_batch * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
          # No clipping
          values_pred = values
        else:
          # Clip the difference between old and new value
          # NOTE: this depends on the reward scaling
          values_pred = returns_batch + th.clamp(values - returns_batch, -clip_range_vf, clip_range_vf)

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns_batch, values_pred)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
          # Approximate entropy when no analytical form
          entropy_loss = -th.mean(-log_prob)
        else:
          entropy_loss = -th.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
          log_ratio = log_prob - old_log_prob_batch
          approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
          approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          if self.verbose >= 1:
            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
          break

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1
      if not continue_training:
        break

    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Logs
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
      self.logger.record("train/clip_range_vf", clip_range_vf)


def sample_genppo_params(trial, n_actions, n_envs, additional_args):
  n_epochs = 10
  clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
  n_timesteps = trial.suggest_categorical("n_timesteps", [100000])

  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

  batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  # Entropy coefficient for regularization
  entropy_coef = trial.suggest_float("entropy_coef", -1, 1, step=0.01)
  sampling_coef = trial.suggest_float("sampling_coef", -1, 1, step=0.01)

  epsilon = trial.suggest_float("epsilon", 0.1, 0.9, step=0.05)

  orthogonal_init = trial.suggest_categorical("orthogonal_init", [True, False])

  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  normalized_advantage = trial.suggest_categorical("normalize_advantage", [True, False])

  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
  net_arch_dict = {
    "small": {"pi": [64, 64], "vf": [64, 64]},
    "medium": {"pi": [256, 256], "vf": [256, 256]},
    "large": {"pi": [400, 300], "vf": [400, 300]},
  }
  net_arch = net_arch_dict[net_arch_type]
  assert isinstance(net_arch["pi"], list) and all(isinstance(x, int) for x in net_arch["pi"]), "Invalid pi architecture"
  assert isinstance(net_arch["vf"], list) and all(isinstance(x, int) for x in net_arch["vf"]), "Invalid vf architecture"

  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "learning_rate": learning_rate,
    "n_epochs": n_epochs,
    "clip_range": clip_range,
    "gae_lambda": gae_lambda,
    "entropy_coef": entropy_coef,
    "sampling_coef": sampling_coef,
    "normalize_advantage": normalized_advantage,
    "n_envs": n_envs,
    "epsilon": epsilon,
    "target_kl": target_kl,
    "cg_max_steps": cg_max_steps,
    "n_critic_updates": n_critic_updates,
    "policy_kwargs": dict(
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=orthogonal_init,
    ),
    **additional_args,
  }
