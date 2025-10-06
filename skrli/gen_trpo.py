import copy
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.agents.torch.trpo import TRPO_DEFAULT_CONFIG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import spaces
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from skrli.networks import ForwardDynamicsModel, GenerativeReplayBuffer
from skrli.strategy import sampling_strategy
from skrli.trpo import SKRL_TRPO_WITH_COLLECT


class GenTRPO(SKRL_TRPO_WITH_COLLECT):
  def __init__(
    self,
    models: Mapping[str, Model],
    memory: Optional[Union[Memory, Tuple[Memory]]] = None,
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    cfg: Optional[dict] = None,
    sampling_coef: float = 0.5,
    buffer_capacity: int = 10000,
    normalize_advantage: bool = False,
    entropy_coef=0.01,
    noise_type="uniform",
    noise_level=0.0,
    **kwargs: Any,
  ) -> None:
    _cfg = copy.deepcopy(TRPO_DEFAULT_CONFIG)
    _cfg.update(cfg if cfg is not None else {})
    super().__init__(
      models=models,
      memory=memory,
      cfg=_cfg,
      observation_space=observation_space,
      action_space=action_space,
      device=device,
      **kwargs,
    )
    self.sampling_coef = sampling_coef
    self.normalize_advantage = normalize_advantage
    self.buffer_capacity = buffer_capacity
    self.forward_dynamics_model = ForwardDynamicsModel(observation_space=self.observation_space, action_space=self.action_space).to(self.device)
    self.dynamics_optimizer = th.optim.Adam(self.forward_dynamics_model.parameters(), lr=1e-3)

    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=self.buffer_capacity,
      synthetic_capacity=self.buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      device=self.device,
      batch_size=self.cfg["batch_size"],
      action_dim=self.action_space.shape[0],
    )
    self._entropy_coef = entropy_coef
    self.noise_type = noise_type
    self.noise_level = noise_level
    self.base_std = 1.0
    self.base_range = 1.0
    self.base_scale = 1.0
    self.base_p = 0.5

  def _compute_relevance(self, transition):
    obs, action, _, _, _ = transition
    obs = obs.to(self.device)
    action = action.to(self.device)
    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(obs.unsqueeze(0), action.unsqueeze(0))
    curiosity_score = 0.5 * th.norm(pred_h_next - h_s, p=2).item() ** 2
    return curiosity_score

  def act(self, states, timestep=0, timesteps=1):
    actions, log_prob, infos = super().act(states, timestep, timesteps)
    if self.noise_level != 0:
      actions = self._add_noise(actions, "action")
      actions = th.clamp(actions, th.tensor(self.action_space.low, device=self.device), th.tensor(self.action_space.high, device=self.device))
    return actions, log_prob, infos

  def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
    if self.noise_level != 0:
      rewards = self._add_noise(rewards, "reward")
    super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

  def _add_noise(self, value, component):
    noise_level = abs(self.noise_level)
    noise_type = self.noise_type
    if component == "action":
      if noise_type == "bernoulli":
        return value
      if noise_type == "gaussian":
        std = noise_level * self.base_std
        return value + th.normal(0, std, size=value.shape, device=self.device)
      elif noise_type == "uniform":
        range_val = noise_level * self.base_range
        return value + th.empty(size=value.shape, device=self.device).uniform_(-range_val, range_val)
      elif noise_type == "laplace":
        scale = noise_level * self.base_scale
        return value + th.distributions.laplace.Laplace(0, scale).sample(value.shape).to(self.device)
    elif component == "reward":
      if noise_type == "gaussian":
        std = noise_level * self.base_std
        return value + th.normal(0, std, size=value.shape, device=self.device)
      elif noise_type == "uniform":
        range_val = noise_level * self.base_range
        return value + th.empty(size=value.shape, device=self.device).uniform_(-range_val, range_val)
      elif noise_type == "laplace":
        scale = noise_level * self.base_scale
        return value + th.distributions.laplace.Laplace(0, scale).sample(value.shape).to(self.device)
      elif noise_type == "bernoulli":
        p = noise_level * self.base_p
        mask = (th.rand(value.shape, device=self.device) < p).float()
        return value * (1 - mask)
    return value

  def _update(self, timestep: int, timesteps: int) -> None:
    def compute_gae(
      rewards: th.Tensor,
      dones: th.Tensor,
      values: th.Tensor,
      next_values: th.Tensor,
      discount_factor: float = 0.99,
      lambda_coefficient: float = 0.95,
    ) -> th.Tensor:
      advantage = 0
      advantages = th.zeros_like(rewards)
      not_dones = dones.logical_not()
      memory_size = rewards.shape[0]

      for i in reversed(range(memory_size)):
        next_value = values[i + 1] if i < memory_size - 1 else last_values
        advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_value + lambda_coefficient * advantage)
        advantages[i] = advantage
      returns = advantages + values
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      return returns, advantages

    def surrogate_loss(policy: Model, states: th.Tensor, actions: th.Tensor, log_prob: th.Tensor, advantages: th.Tensor) -> th.Tensor:
      _, new_log_prob, outputs = policy.act({"states": states, "taken_actions": actions}, role="policy")
      surrogate = (advantages * th.exp(new_log_prob - log_prob.detach())).mean()

      logstd = outputs["log_std"]
      action_dim = logstd.shape[-1]
      entropy = 0.5 * action_dim * (th.log(2 * th.pi * th.e)) + logstd.sum(dim=-1)
      entropy = entropy.mean()

      return surrogate + self._entropy_coef * entropy

    def conjugate_gradient(
      policy: Model,
      states: th.Tensor,
      b: th.Tensor,
      num_iterations: float = 10,
      residual_tolerance: float = 1e-10,
    ) -> th.Tensor:
      x = th.zeros_like(b)
      r = b.clone()
      p = b.clone()
      rr_old = th.dot(r, r)
      for _ in range(num_iterations):
        hv = fisher_vector_product(policy, states, p, damping=self._damping)
        alpha = rr_old / th.dot(p, hv)
        x += alpha * p
        r -= alpha * hv
        rr_new = th.dot(r, r)
        if rr_new < residual_tolerance:
          break
        p = r + rr_new / rr_old * p
        rr_old = rr_new
      return x

    def fisher_vector_product(policy: Model, states: th.Tensor, vector: th.Tensor, damping: float = 0.1) -> th.Tensor:
      kl = kl_divergence(policy, policy, states)
      kl_gradient = th.autograd.grad(kl, policy.parameters(), create_graph=True)
      flat_kl_gradient = th.cat([gradient.view(-1) for gradient in kl_gradient])
      hessian_vector_gradient = th.autograd.grad((flat_kl_gradient * vector).sum(), policy.parameters())
      flat_hessian_vector_gradient = th.cat([gradient.contiguous().view(-1) for gradient in hessian_vector_gradient])
      return flat_hessian_vector_gradient + damping * vector

    def kl_divergence(policy_1: Model, policy_2: Model, states: th.Tensor) -> th.Tensor:
      mu_1 = policy_1.act({"states": states}, role="policy")[2]["mean_actions"]
      logstd_1 = policy_1.get_log_std(role="policy")
      mu_1, logstd_1 = mu_1.detach(), logstd_1.detach()

      mu_2 = policy_2.act({"states": states}, role="policy")[2]["mean_actions"]
      logstd_2 = policy_2.get_log_std(role="policy")

      kl = logstd_1 - logstd_2 + 0.5 * (th.square(logstd_1.exp()) + th.square(mu_1 - mu_2)) / th.square(logstd_2.exp()) - 0.5
      return th.sum(kl, dim=-1).mean()

    # compute returns and advantages
    with th.no_grad():
      self.value.train(False)
      last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states.float())}, role="value")
      self.value.train(True)
    last_values = self._value_preprocessor(last_values, inverse=True)

    values = self.memory.get_tensor_by_name("values")
    returns, advantages = compute_gae(
      rewards=self.memory.get_tensor_by_name("rewards"),
      dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
      values=values,
      next_values=last_values,
      discount_factor=self._discount_factor,
      lambda_coefficient=self._lambda,
    )

    self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
    self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
    self.memory.set_tensor_by_name("advantages", advantages)

    # Train forward dynamics model on current rollout
    states = self.memory.get_tensor_by_name("states")
    actions = self.memory.get_tensor_by_name("actions")
    next_states = th.cat([states[1:], self._current_next_states.unsqueeze(0)], dim=0)
    for _ in range(5):
      self.forward_dynamics_model.train()
      self.dynamics_optimizer.zero_grad()
      h_s, pred_h_next = self.forward_dynamics_model(states, actions)
      h_next = self.forward_dynamics_model.encoder(next_states.reshape(-1, self.forward_dynamics_model.state_dim))
      loss = F.mse_loss(pred_h_next, h_next.detach())
      loss.backward()
      self.dynamics_optimizer.step()

    # Add to replay buffer
    states = self.memory.get_tensor_by_name("states")
    actions = self.memory.get_tensor_by_name("actions")
    returns = self.memory.get_tensor_by_name("returns")
    advantages = self.memory.get_tensor_by_name("advantages")
    log_prob = self.memory.get_tensor_by_name("log_prob")

    num_steps, num_envs = states.shape[:2]
    total_samples = num_steps * num_envs

    states_flat = states.reshape(total_samples, -1)
    actions_flat = actions.reshape(total_samples, -1)
    returns_flat = returns.reshape(total_samples)
    advantages_flat = advantages.reshape(total_samples)
    log_prob_flat = log_prob.reshape(total_samples)

    for i in range(total_samples):
      transition = (states_flat[i], actions_flat[i], returns_flat[i], advantages_flat[i], log_prob_flat[i])
      self.replay_buffer.add_real(transition)
    # Generate synthetic
    self.replay_buffer.generate_synthetic()

    # sample all from memory for policy
    sampled_states, sampled_actions, sampled_log_prob, sampled_advantages = self.memory.sample_all(names=self._tensors_names_policy, mini_batches=1)[0]
    sampled_returns = self.memory.get_tensor_by_name("returns")

    # Flatten them if not already
    sampled_states = sampled_states.reshape(-1, self.observation_space.shape[0])
    sampled_actions = sampled_actions.reshape(-1, self.action_space.shape[0])
    sampled_log_prob = sampled_log_prob.reshape(-1)
    sampled_advantages = sampled_advantages.reshape(-1)
    sampled_returns = sampled_returns.reshape(-1)

    # Compute entropy
    _, _, outputs = self.policy.act({"states": sampled_states}, role="policy")
    logstd = outputs["log_std"]
    action_dim = logstd.shape[-1]
    entropy = 0.5 * action_dim * th.log(th.tensor(2 * th.pi * th.e)) + logstd.sum(dim=-1)
    entropy_mean = entropy.mean().item()

    num_replay_samples = sampling_strategy(entropy_mean, self.sampling_coef, min_samples=0, max_samples=self.cfg["batch_size"])

    if num_replay_samples > 0:
      replay_sampled = self.replay_buffer.sample(num_replay_samples)
      replay_states = th.stack([t[0] for t in replay_sampled])
      replay_actions = th.stack([t[1] for t in replay_sampled])
      replay_returns = th.stack([t[2] for t in replay_sampled])
      replay_advantages = th.stack([t[3] for t in replay_sampled])
      replay_log_prob = th.stack([t[4] for t in replay_sampled])
      sampled_states = th.cat([sampled_states, replay_states])
      sampled_actions = th.cat([sampled_actions, replay_actions])
      sampled_returns = th.cat([sampled_returns, replay_returns])
      sampled_advantages = th.cat([sampled_advantages, replay_advantages])
      sampled_log_prob = th.cat([sampled_log_prob, replay_log_prob])

    sampled_states = self._state_preprocessor(sampled_states, train=True)

    if self.normalize_advantage:
      sampled_advantages = (sampled_advantages - sampled_advantages.mean()) / (sampled_advantages.std() + 1e-8)

    # compute policy loss gradient
    policy_loss = surrogate_loss(self.policy, sampled_states, sampled_actions, sampled_log_prob, sampled_advantages)
    policy_loss_gradient = th.autograd.grad(policy_loss, self.policy.parameters())
    flat_policy_loss_gradient = th.cat([gradient.view(-1) for gradient in policy_loss_gradient])

    # compute the search direction using the conjugate gradient algorithm
    search_direction = conjugate_gradient(self.policy, sampled_states, flat_policy_loss_gradient.data, num_iterations=self._conjugate_gradient_steps)

    # compute step size and full step
    xHx = (search_direction * fisher_vector_product(self.policy, sampled_states, search_direction, self._damping)).sum(0, keepdim=True)
    step_size = th.sqrt(2 * self._max_kl_divergence / xHx)[0]
    full_step = step_size * search_direction

    # backtracking line search
    restore_policy_flag = True
    self.backup_policy.update_parameters(self.policy)
    params = parameters_to_vector(self.policy.parameters())

    expected_improvement = (flat_policy_loss_gradient * full_step).sum(0, keepdim=True)

    for alpha in [self._step_fraction * 0.5**i for i in range(self._max_backtrack_steps)]:
      new_params = params + alpha * full_step
      vector_to_parameters(new_params, self.policy.parameters())

      expected_improvement *= alpha
      kl = kl_divergence(self.backup_policy, self.policy, sampled_states)
      loss = surrogate_loss(self.policy, sampled_states, sampled_actions, sampled_log_prob, sampled_advantages)

      if kl < self._max_kl_divergence and (loss - policy_loss) / expected_improvement > self._accept_ratio:
        restore_policy_flag = False
        break

    if restore_policy_flag:
      self.policy.update_parameters(self.backup_policy)

    if config.torch.is_distributed:
      self.policy.reduce_parameters()

    # value updates with augmented data
    cumulative_value_loss = 0

    # learning epochs
    for epoch in range(self._learning_epochs):
      s_states = self._state_preprocessor(sampled_states, train=not epoch)

      # compute value loss
      predicted_values, _, _ = self.value.act({"states": s_states}, role="value")

      value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

      # optimization step (value)
      self.value_optimizer.zero_grad()
      value_loss.backward()
      if config.torch.is_distributed:
        self.value.reduce_parameters()
      if self._grad_norm_clip > 0:
        nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
      self.value_optimizer.step()

      cumulative_value_loss += value_loss.item()

    if self._learning_rate_scheduler:
      self.value_scheduler.step()

    # record data
    self.track_data("Loss / Policy loss", policy_loss.item())
    self.track_data("Loss / Value loss", cumulative_value_loss / self._learning_epochs)

    self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

    if self._learning_rate_scheduler:
      self.track_data("Learning / Value learning rate", self.value_scheduler.get_last_lr()[0])
