# sb3/trpo.py
import copy
from functools import partial
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import optuna
import torch as th
import torch.nn as nn
from gymnasium import spaces
from rl_zoo3 import linear_schedule
from sb3_contrib import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from sb3.noise import MonitoredEntropyInjectionWrapper


class TRPO(TRPO):
  """
    Just a class to ingore extra arguments for compatibility.
    """

  def __init__(
    self,
    policy: Union[str, Type[ActorCriticPolicy]],
    env: Union[GymEnv, str],
    learning_rate: Union[float, Schedule] = 1e-3,
    n_steps: int = 2048,
    batch_size: int = 128,
    gamma: float = 0.99,
    cg_max_steps: int = 15,
    cg_damping: float = 0.1,
    line_search_shrinking_factor: float = 0.8,
    line_search_max_iter: int = 10,
    n_critic_updates: int = 10,
    gae_lambda: float = 0.95,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
    rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
    normalize_advantage: bool = True,
    target_kl: float = 0.01,
    sub_sampling_factor: int = 1,
    stats_window_size: int = 100,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[th.device, str] = "cuda",
    _init_setup_model: bool = True,
    noise_configs=None,
    **kwargs,
  ):

    super().__init__(
      policy=policy,
      env=env,
      learning_rate=learning_rate,
      n_steps=n_steps,
      batch_size=batch_size,
      gamma=gamma,
      cg_max_steps=cg_max_steps,
      cg_damping=cg_damping,
      line_search_shrinking_factor=line_search_shrinking_factor,
      line_search_max_iter=line_search_max_iter,
      n_critic_updates=n_critic_updates,
      gae_lambda=gae_lambda,
      use_sde=use_sde,
      sde_sample_freq=sde_sample_freq,
      rollout_buffer_class=rollout_buffer_class,
      rollout_buffer_kwargs=rollout_buffer_kwargs,
      normalize_advantage=normalize_advantage,
      target_kl=target_kl,
      sub_sampling_factor=sub_sampling_factor,
      stats_window_size=stats_window_size,
      tensorboard_log=tensorboard_log,
      policy_kwargs=policy_kwargs,
      verbose=verbose,
      seed=seed,
      device=device,
      _init_setup_model=_init_setup_model,
    )
    # Ignore kwargs for compatibility

    self.rollout_metrics = {}

    if noise_configs is not None:
      if self.env is None:
        raise ValueError("Environment must be provided when noise_configs is specified.")
      if hasattr(self.env, "envs"):
        for i in range(self.env.num_envs):
          self.env.envs[i] = MonitoredEntropyInjectionWrapper(self.env.envs[i], noise_configs=noise_configs)
      else:
        self.env = MonitoredEntropyInjectionWrapper(self.env, noise_configs=noise_configs)

    # Print device and verify it is being used
    if isinstance(device, str):
      device = th.device(device if th.cuda.is_available() else "cpu")
    elif isinstance(device, th.device):
      device = device if th.cuda.is_available() else th.device("cpu")

  def _compute_policy_objective(self, advantages, ratio, distribution):
    """Overridable method for computing policy objective."""
    return (advantages * ratio).mean()

  def _save_rollout_metrics(
    self,
    kl_divergences,
    explained_var,
    value_losses,
    policy_stds,
    line_search_results,
    grad_norm_policy,
    value_grad_norms,
    advantages,
    entropies,
    action_deltas,
    reward_deltas,
    rewards,
  ):
    """Overridable method for computing metrics."""

    def compute_stats(arr):
      arr = np.array(arr, dtype=float)
      if len(arr) == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
      return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
      }

    metrics = {
      "explained_variance": float(explained_var),
      "grad_norm_policy": float(grad_norm_policy),
    }

    for prefix, arr in [
      ("kl_div", kl_divergences),
      ("value_loss", value_losses),
      ("line_search_success", [float(r) for r in line_search_results]),
      ("grad_norm_value", value_grad_norms),
      ("adv", advantages),
      ("entropy", entropies),
      ("action_delta", action_deltas),
      ("reward_delta", reward_deltas),
      ("policy_std", policy_stds),
      ("reward", rewards),
    ]:
      stats = compute_stats(arr)
      for stat_name, value in stats.items():
        metrics[f"{prefix}_{stat_name}"] = value

    # in every pass ensure the self.rollout_metrics has all the keys initialized
    for key in metrics.keys():
      if key not in self.rollout_metrics:
        self.rollout_metrics[key] = []

    for key, value in metrics.items():
      self.rollout_metrics[key].append(value)

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []

    # This will only loop once (get all data in one go)
    for rollout_data in self.rollout_buffer.get(batch_size=None):

      # Optional: sub-sample data for faster computation
      if self.sub_sampling_factor > 1:
        rollout_data = RolloutBufferSamples(
          rollout_data.observations[:: self.sub_sampling_factor],
          rollout_data.actions[:: self.sub_sampling_factor],
          None,  # old values, not used here
          rollout_data.old_log_prob[:: self.sub_sampling_factor],
          rollout_data.advantages[:: self.sub_sampling_factor],
          None,  # returns, not used here
        )

      actions = rollout_data.actions
      if isinstance(self.action_space, spaces.Discrete):
        # Convert discrete action from float to long
        actions = rollout_data.actions.long().flatten()

      # Re-sample the noise matrix because the log_std has changed
      if self.use_sde:
        # batch_size is only used for the value function
        self.policy.reset_noise(actions.shape[0])

      with th.no_grad():
        # Note: is copy enough, no need for deepcopy?
        # If using gSDE and deepcopy, we need to use `old_distribution.distribution.distribution`
        # directly to avoid PyTorch errors.
        old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)

      advantages = rollout_data.advantages
      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

      # ratio between old and new policy, should be one at the first iteration
      ratio = th.exp(log_prob - rollout_data.old_log_prob)

      # surrogate policy objective
      policy_objective = self._compute_policy_objective(advantages, ratio, distribution)

      # KL divergence
      kl_div = kl_divergence(distribution, old_distribution).mean()

      # Surrogate & KL gradient
      self.policy.optimizer.zero_grad()

      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

      grad_norm_policy = th.norm(policy_objective_gradients).item()

      # Hessian-vector dot product function used in the conjugate gradient step
      hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

      # Computing search direction
      search_direction = conjugate_gradient_solver(
        hessian_vector_product_fn,
        policy_objective_gradients,
        max_iter=self.cg_max_steps,
      )

      # Maximal step length
      line_search_max_step_size = 2 * self.target_kl
      line_search_max_step_size /= th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
      line_search_max_step_size = th.sqrt(line_search_max_step_size)

      line_search_backtrack_coeff = 1.0
      original_actor_params = [param.detach().clone() for param in actor_params]

      is_line_search_success = False
      with th.no_grad():
        # Line-search (backtracking)
        for _ in range(self.line_search_max_iter):

          start_idx = 0
          # Applying the scaled step direction
          for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
            n_params = param.numel()
            param.data = original_param.data + line_search_backtrack_coeff * line_search_max_step_size * search_direction[
              start_idx : (start_idx + n_params)
            ].view(shape)
            start_idx += n_params

          # Recomputing the policy log-probabilities
          distribution = self.policy.get_distribution(rollout_data.observations)
          log_prob = distribution.log_prob(actions)

          # New policy objective
          ratio = th.exp(log_prob - rollout_data.old_log_prob)
          new_policy_objective = (advantages * ratio).mean()

          # New KL-divergence
          kl_div = kl_divergence(distribution, old_distribution).mean()

          # Constraint criteria:
          # we need to improve the surrogate policy objective
          # while being close enough (in term of kl div) to the old policy
          if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
            is_line_search_success = True
            break

          # Reducing step size if line-search wasn't successful
          line_search_backtrack_coeff *= self.line_search_shrinking_factor

        line_search_results.append(is_line_search_success)

        if not is_line_search_success:
          # If the line-search wasn't successful we revert to the original parameters
          for param, original_param in zip(actor_params, original_actor_params):
            param.data = original_param.data.clone()

          policy_objective_values.append(policy_objective.item())
          kl_divergences.append(0)
        else:
          policy_objective_values.append(new_policy_objective.item())
          kl_divergences.append(kl_div.item())

    # Critic update
    value_grad_norms = []
    for _ in range(self.n_critic_updates):
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        values_pred = self.policy.predict_values(rollout_data.observations)
        value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
        value_losses.append(value_loss.item())

        self.policy.optimizer.zero_grad()
        value_loss.backward()
        grad_norm_value = 0.0
        for p in self.policy.value_net.parameters():
          if p.grad is not None:
            grad_norm_value += p.grad.norm(2) ** 2
        grad_norm_value = grad_norm_value**0.5
        value_grad_norms.append(grad_norm_value)
        # Removing gradients of parameters shared with the actor
        # otherwise it defeats the purposes of the KL constraint
        for param in actor_params:
          param.grad = None
        self.policy.optimizer.step()

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Additional metrics
    if hasattr(self.policy, "log_std"):
      policy_stds = th.exp(self.policy.log_std).detach().cpu().numpy()
    else:
      policy_stds = np.array([0.0])

    # Entropy
    distribution = self.policy.get_distribution(rollout_data.observations)
    entropies = distribution.entropy().detach().cpu().numpy()

    # Noise stats
    action_deltas = []
    reward_deltas = []
    if hasattr(self.env, "envs"):
      for e in self.env.envs:
        if isinstance(e, MonitoredEntropyInjectionWrapper):
          a_deltas, r_deltas = e.get_noise_deltas()
          action_deltas.extend(a_deltas)
          reward_deltas.extend(r_deltas)
    elif isinstance(self.env, MonitoredEntropyInjectionWrapper):
      action_deltas, reward_deltas = self.env.get_noise_deltas()

    advantages_numpy = advantages.detach().cpu().numpy()

    rewards = self.rollout_buffer.rewards.flatten()

    self._save_rollout_metrics(
      kl_divergences,
      explained_var,
      value_losses,
      policy_stds,
      line_search_results,
      grad_norm_policy,
      value_grad_norms,
      advantages_numpy,
      entropies,
      action_deltas,
      reward_deltas,
      rewards,
    )

    # Logs
    self.logger.record("train/policy_objective", np.mean(policy_objective_values))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", np.mean(line_search_results))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def sample_trpo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> Dict[str, Any]:
  """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
  # Uncomment for gSDE (continuous actions)
  log_std_init = trial.suggest_float("log_std_init", -4, 1)
  # Uncomment for gSDE (continuous action)
  sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
  # Orthogonal initialization
  ortho_init = False
  ortho_init = trial.suggest_categorical("ortho_init", [False, True])
  # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  # lr_schedule = "constant"
  # Uncomment to enable learning rate schedule
  lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
  if lr_schedule == "linear":
    learning_rate = linear_schedule(learning_rate)

  # TODO: account when using multiple envs
  if batch_size > n_steps:
    batch_size = n_steps

  # Independent networks usually work best
  # when not working with images
  net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
  }[net_arch_type]

  activation_fn = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
  }[activation_fn_name]

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)

  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  return {
    "n_timesteps": n_timesteps,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "n_envs": n_envs,
    "cg_damping": cg_damping,
    "cg_max_steps": cg_max_steps,
    "line_search_shrinking_factor": line_search_shrinking_factor,
    "n_critic_updates": n_critic_updates,
    "target_kl": target_kl,
    "learning_rate": learning_rate,
    "gae_lambda": gae_lambda,
    "sde_sample_freq": sde_sample_freq,
    "policy_kwargs": dict(
      log_std_init=log_std_init,
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=ortho_init,
    ),
  }
