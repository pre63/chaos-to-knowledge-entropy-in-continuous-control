import argparse
import copy
import os
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.pruners
import optuna.samplers
import optuna.storages
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.agents.torch.trpo import TRPO
from skrl.agents.torch.trpo import TRPO as SKRL_TRPO
from skrl.agents.torch.trpo import TRPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import Memory, RandomMemory
from skrl.models.torch import Model
from skrl.trainers.torch import SequentialTrainer
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from Environments.Noise import EntropyInjectionWrapper


class AutoPlotProgressWrapper(gym.Wrapper):
  def __init__(self, env, config, env_id, log_dir=".omega", plot_interval=100):
    super().__init__(env)
    self.config = config
    self.env_id = env_id
    self.log_dir = log_dir
    self.plot_interval = plot_interval
    self.timesteps = 0
    self.rewards = []
    os.makedirs(self.log_dir, exist_ok=True)

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.timesteps += 1
    self.rewards.append(float(reward))

    if self.timesteps % self.plot_interval == 0:
      self._save_rewards()
      self._plot_progress()

    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs):
    self.timesteps = 0
    return self.env.reset(**kwargs)

  def _save_rewards(self):
    config_name = self.config.upper().replace("_", " ")
    path = os.path.join(self.log_dir, f"rewards_{self.env_id}_{self.config}.yaml")
    with open(path, "w") as f:
      yaml.safe_dump({"rewards": self.rewards}, f, default_flow_style=False)

  def _plot_progress(self):
    plt.figure(figsize=(10, 6))
    plt.clf()

    configs = [
      "trpo_no_noise",
      "trpo_with_noise",
      "trpor_no_noise",
      "trpor_with_noise",
      "trponr_no_noise",
      "trponr_with_noise",
    ]
    for cfg in configs:
      path = os.path.join(self.log_dir, f"rewards_{self.env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        if rewards:
          timesteps = range(1, len(rewards) + 1)
          if len(rewards) >= 100:
            smoothed = np.convolve(rewards, np.ones(100) / 100, mode="valid")
            smoothed_timesteps = range(50, len(smoothed) + 50)
            plt.plot(
              smoothed_timesteps,
              smoothed,
              label=cfg.upper().replace("_", " "),
            )
          else:
            plt.plot(timesteps, rewards, label=cfg.upper().replace("_", " "))

    plt.title(f"Training Progress per Configuration - {self.env_id}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_dir, f"graph_{self.env_id}.png"))
    plt.close()


def flat_grad(y, x, retain_graph=False, create_graph=False):
  if create_graph:
    retain_graph = True
  g = th.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
  g = th.cat([t.view(-1) for t in g if t is not None])
  return g


def get_flat_params(model):
  return th.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
  n = 0
  for p in model.parameters():
    numel = p.numel()
    p.data.copy_(flat_params[n : n + numel].view(p.shape))
    n += numel


def conjugate_gradient(A, b, max_iterations=10, residual_tol=1e-10):
  x = th.zeros_like(b)
  r = b.clone()
  p = b.clone()
  rdotr_old = r.dot(r)
  if rdotr_old == 0:
    return x
  for i in range(max_iterations):
    Ap = A(p)
    alpha = rdotr_old / p.dot(Ap)
    x += alpha * p
    r -= alpha * Ap
    rdotr_new = r.dot(r)
    if rdotr_new < residual_tol:
      break
    beta = rdotr_new / rdotr_old
    p = r + beta * p
    rdotr_old = rdotr_new
  return x


class SKRL_TRPO_WITH_COLLECT(SKRL_TRPO):
  def __init__(
    self,
    models,
    memory=None,
    cfg=None,
    observation_space=None,
    action_space=None,
    device=None,
  ):
    super().__init__(
      models=models,
      memory=memory,
      cfg=cfg,
      observation_space=observation_space,
      action_space=action_space,
      device=device,
    )
    if str(self.device) != "cuda":
      raise ValueError("Device must be cuda for GPU acceleration")
    print(f"Using device: {self.device} for audit purposes")
    self.rewards = []

  def post_interaction(self, timestep, timesteps):
    super().post_interaction(timestep, timesteps)
    if self.memory is not None:
      rewards = self.memory.get_tensor_by_name("rewards")
      if rewards.numel() > 0:
        self.rewards.append(float(rewards.mean()))


class TRPOR(SKRL_TRPO_WITH_COLLECT):

  def __init__(
    self,
    models: Mapping[str, Model],
    memory: Optional[Union[Memory, Tuple[Memory]]] = None,
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    cfg: Optional[dict] = None,
  ) -> None:
    """TRPO with entropy regularization in the surrogate loss

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``)
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
    _cfg = copy.deepcopy(TRPO_DEFAULT_CONFIG)
    _cfg.update(cfg if cfg is not None else {})
    # Add entropy coefficient to config (default value can be tuned)
    _cfg["entropy_coef"] = _cfg.get("entropy_coef", 0.01)

    super().__init__(
      models=models,
      memory=memory,
      observation_space=observation_space,
      action_space=action_space,
      device=device,
      cfg=_cfg,
    )
    self._entropy_coef = self.cfg["entropy_coef"]

  def _update(self, timestep: int, timesteps: int) -> None:
    """Algorithm's main update step with entropy regularization

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

    def compute_gae(
      rewards: torch.Tensor,
      dones: torch.Tensor,
      values: torch.Tensor,
      next_values: torch.Tensor,
      discount_factor: float = 0.99,
      lambda_coefficient: float = 0.95,
    ) -> torch.Tensor:
      """Compute the Generalized Advantage Estimator (GAE)"""
      advantage = 0
      advantages = torch.zeros_like(rewards)
      not_dones = dones.logical_not()
      memory_size = rewards.shape[0]

      for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
        advantages[i] = advantage
      returns = advantages + values
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      return returns, advantages

    def surrogate_loss(policy: Model, states: torch.Tensor, actions: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
      """Compute the surrogate objective (policy loss) with entropy regularization"""
      _, new_log_prob, outputs = policy.act({"states": states, "taken_actions": actions}, role="policy")
      surrogate = (advantages * torch.exp(new_log_prob - log_prob.detach())).mean()

      # Compute Shannon entropy for Gaussian policy
      logstd = policy.get_log_std(role="policy")
      action_dim = logstd.shape[-1]
      entropy = 0.5 * action_dim * (torch.log(2 * torch.pi * torch.e)) + logstd.sum(dim=-1)
      entropy = entropy.mean()

      return surrogate + self._entropy_coef * entropy

    def conjugate_gradient(
      policy: Model,
      states: torch.Tensor,
      b: torch.Tensor,
      num_iterations: float = 10,
      residual_tolerance: float = 1e-10,
    ) -> torch.Tensor:
      """Conjugate gradient algorithm to solve Ax = b"""
      x = torch.zeros_like(b)
      r = b.clone()
      p = b.clone()
      rr_old = torch.dot(r, r)
      for _ in range(num_iterations):
        hv = fisher_vector_product(policy, states, p, damping=self._damping)
        alpha = rr_old / torch.dot(p, hv)
        x += alpha * p
        r -= alpha * hv
        rr_new = torch.dot(r, r)
        if rr_new < residual_tolerance:
          break
        p = r + rr_new / rr_old * p
        rr_old = rr_new
      return x

    def fisher_vector_product(policy: Model, states: torch.Tensor, vector: torch.Tensor, damping: float = 0.1) -> torch.Tensor:
      """Compute the Fisher vector product"""
      kl = kl_divergence(policy, policy, states)
      kl_gradient = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
      flat_kl_gradient = torch.cat([gradient.view(-1) for gradient in kl_gradient])
      hessian_vector_gradient = torch.autograd.grad((flat_kl_gradient * vector).sum(), policy.parameters())
      flat_hessian_vector_gradient = torch.cat([gradient.contiguous().view(-1) for gradient in hessian_vector_gradient])
      return flat_hessian_vector_gradient + damping * vector

    def kl_divergence(policy_1: Model, policy_2: Model, states: torch.Tensor) -> torch.Tensor:
      """Compute the KL divergence between two distributions"""
      mu_1 = policy_1.act({"states": states}, role="policy")[2]["mean_actions"]
      logstd_1 = policy_1.get_log_std(role="policy")
      mu_1, logstd_1 = mu_1.detach(), logstd_1.detach()

      mu_2 = policy_2.act({"states": states}, role="policy")[2]["mean_actions"]
      logstd_2 = policy_2.get_log_std(role="policy")

      kl = logstd_1 - logstd_2 + 0.5 * (torch.square(logstd_1.exp()) + torch.square(mu_1 - mu_2)) / torch.square(logstd_2.exp()) - 0.5
      return torch.sum(kl, dim=-1).mean()

    # compute returns and advantages
    with torch.no_grad():
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

    # sample all from memory
    sampled_states, sampled_actions, sampled_log_prob, sampled_advantages = self.memory.sample_all(names=self._tensors_names_policy, mini_batches=1)[0]

    sampled_states = self._state_preprocessor(sampled_states, train=True)

    # compute policy loss gradient
    policy_loss = surrogate_loss(self.policy, sampled_states, sampled_actions, sampled_log_prob, sampled_advantages)
    policy_loss_gradient = torch.autograd.grad(policy_loss, self.policy.parameters())
    flat_policy_loss_gradient = torch.cat([gradient.view(-1) for gradient in policy_loss_gradient])

    # compute the search direction using the conjugate gradient algorithm
    search_direction = conjugate_gradient(self.policy, sampled_states, flat_policy_loss_gradient.data, num_iterations=self._conjugate_gradient_steps)

    # compute step size and full step
    xHx = (search_direction * fisher_vector_product(self.policy, sampled_states, search_direction, self._damping)).sum(0, keepdim=True)
    step_size = torch.sqrt(2 * self._max_kl_divergence / xHx)[0]
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

    # sample mini-batches from memory
    sampled_batches = self.memory.sample_all(names=self._tensors_names_value, mini_batches=self._mini_batches)

    cumulative_value_loss = 0

    # learning epochs
    for epoch in range(self._learning_epochs):
      for sampled_states, sampled_returns in sampled_batches:
        sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

        # compute value loss
        predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

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
    self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))

    self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

    if self._learning_rate_scheduler:
      self.track_data("Learning / Value learning rate", self.value_scheduler.get_last_lr()[0])


class TRPONR(TRPOR):
  def __init__(
    self,
    models,
    memory=None,
    cfg=None,
    observation_space=None,
    action_space=None,
    device=None,
    ent_coef=0.0,
    noise_type="uniform",
    noise_level=0.0,
  ):
    _cfg = copy.deepcopy(TRPO_DEFAULT_CONFIG)
    _cfg.update(cfg if cfg is not None else {})
    _cfg["entropy_coef"] = ent_coef

    super().__init__(
      models=models,
      memory=memory,
      cfg=_cfg,
      observation_space=observation_space,
      action_space=action_space,
      device=device,
    )
    self.noise_type = noise_type
    self.noise_level = noise_level
    self.base_std = 1.0
    self.base_range = 1.0
    self.base_scale = 1.0
    self.base_p = 0.5

  def _update(self, timestep, timesteps):
    super()._update(timestep, timesteps)

  def act(self, states, timestep=0, timesteps=1):
    actions, log_prob, infos = super().act(states, timestep, timesteps)
    if self.noise_level > 0:
      actions = self._add_noise(actions, "action")
      actions = th.clamp(actions, th.tensor(self.action_space.low, device=self.device), th.tensor(self.action_space.high, device=self.device))
    return actions, log_prob, infos

  def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timesteps, role=""):
    if self.noise_level > 0:
      rewards = self._add_noise(rewards, "reward")
    super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timesteps, role)

  def _add_noise(self, value, component):
    entropy_level = abs(self.noise_level)
    noise_type = self.noise_type
    if component == "action":
      if noise_type == "bernoulli":
        return value
      if noise_type == "gaussian":
        std = entropy_level * self.base_std
        return value + th.normal(0, std, size=value.shape, device=self.device)
      elif noise_type == "uniform":
        range_val = entropy_level * self.base_range
        return value + th.empty(size=value.shape, device=self.device).uniform_(-range_val, range_val)
      elif noise_type == "laplace":
        scale = entropy_level * self.base_scale
        return value + th.distributions.laplace.Laplace(0, scale).sample(value.shape).to(self.device)
    elif component == "reward":
      if noise_type == "gaussian":
        std = entropy_level * self.base_std
        return value + th.normal(0, std, size=value.shape, device=self.device)
      elif noise_type == "uniform":
        range_val = entropy_level * self.base_range
        return value + th.empty(size=value.shape, device=self.device).uniform_(-range_val, range_val)
      elif noise_type == "laplace":
        scale = entropy_level * self.base_scale
        return value + th.distributions.laplace.Laplace(0, scale).sample(value.shape).to(self.device)
      elif noise_type == "bernoulli":
        p = entropy_level * self.base_p
        mask = (th.rand(value.shape, device=self.device) < p).float()
        return value * (1 - mask)
    return value


class TRPOWrapper:
  def __init__(
    self,
    env=None,
    learning_rate=1e-3,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    cg_max_steps=15,
    cg_damping=0.1,
    line_search_shrinking_factor=0.8,
    line_search_max_iter=10,
    n_critic_updates=10,
    gae_lambda=0.95,
    normalize_advantage=True,
    target_kl=0.01,
    policy_kwargs=None,
    seed=None,
    device="cuda",
    ent_coef=0.0,
    is_trpor=False,
    noise_type="uniform",
    noise_level=0.0,
    is_trponr=False,
    **kwargs,
  ):
    self.env = env
    self.learning_rate = learning_rate
    self.n_steps = n_steps
    self.batch_size = batch_size
    self.gamma = gamma
    self.cg_max_steps = cg_max_steps
    self.cg_damping = cg_damping
    self.line_search_shrinking_factor = line_search_shrinking_factor
    self.line_search_max_iter = line_search_max_iter
    self.n_critic_updates = n_critic_updates
    self.gae_lambda = gae_lambda
    self.normalize_advantage = normalize_advantage
    self.target_kl = target_kl
    self.policy_kwargs = policy_kwargs or {}
    self.seed = seed
    self.device = th.device("cuda")
    self.ent_coef = ent_coef
    self.is_trpor = is_trpor
    self.noise_type = noise_type
    self.noise_level = noise_level
    self.is_trponr = is_trponr

    self._setup_model()

  def _setup_model(self):
    if isinstance(self.env, str):
      self.env = gym.make(self.env)
    self.env = wrap_env(self.env, wrapper="gymnasium")

    if self.seed is not None:
      np.random.seed(self.seed)
      th.manual_seed(self.seed)

    net_arch = self.policy_kwargs.get("net_arch", dict(pi=[256, 256], vf=[256, 256]))
    pi_arch = net_arch.get("pi", [256, 256])
    vf_arch = net_arch.get("vf", [256, 256])

    shared_net = nn.Sequential().to(self.device)
    input_size = self.env.observation_space.shape[0]
    for size in pi_arch[:-1]:
      shared_net.append(nn.Linear(input_size, size).to(self.device))
      shared_net.append(nn.Tanh())
      input_size = size

    class Policy(Model):
      def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        self.shared_net = shared_net
        self.mean_layer = nn.Linear(input_size, action_space.shape[0]).to(self.device)
        self.log_std_parameter = nn.Parameter(th.zeros(action_space.shape[0]).to(self.device))

      def compute(self, inputs, role):
        mean = self.mean_layer(self.shared_net(inputs["states"]))
        log_std = th.clamp(self.log_std_parameter, -20, 2).expand_as(mean)
        return mean, log_std, {}

      def act(self, inputs, role, taken_action=None, inference=False):
        mean, log_std, _ = self.compute(inputs, role)
        std = log_std.exp()
        dist = th.distributions.Normal(mean, std)
        if taken_action is None:
          if inference:
            action = mean
          else:
            action = dist.rsample()
        else:
          action = taken_action
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, {"mean_actions": mean, "log_std": log_std}

      def get_log_std(self, role=""):
        return th.clamp(self.log_std_parameter, -20, 2)

      def distribution(self, role):
        log_std = th.clamp(self.log_std_parameter, -20, 2)
        std = log_std.exp()
        mean = th.zeros_like(std, device=self.device)
        return th.distributions.Normal(mean, std)

    class Value(Model):
      def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        self.shared_net = shared_net
        self.value_layer = nn.Linear(input_size, 1).to(self.device)

      def compute(self, inputs, role):
        return self.value_layer(self.shared_net(inputs["states"]))

      def act(self, inputs, role, taken_action=None, inference=False):
        value = self.compute(inputs, role)
        return value, {}, {}

    policy = Policy(self.env.observation_space, self.env.action_space, self.device)
    value = Value(self.env.observation_space, self.env.action_space, self.device)
    models = {"policy": policy, "value": value}

    memory = RandomMemory(
      memory_size=self.n_steps,
      num_envs=self.env.num_envs,
      device=self.device,
      replacement=False,
    )

    cfg = TRPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = self.n_steps
    cfg["batch_size"] = self.batch_size
    cfg["discount_factor"] = self.gamma
    cfg["conjugate_gradient_iterations"] = self.cg_max_steps
    cfg["damping"] = self.cg_damping
    cfg["backtrack_ratio"] = self.line_search_shrinking_factor
    cfg["line_search_iterations"] = self.line_search_max_iter
    cfg["value_solver_iterations"] = self.n_critic_updates
    cfg["gae_lambda"] = self.gae_lambda
    cfg["normalize_advantage"] = self.normalize_advantage
    cfg["kl_threshold"] = self.target_kl
    cfg["value_learning_rate"] = self.learning_rate
    cfg["value_clip"] = False
    cfg["clip_actions"] = False

    agent_class = TRPONR if self.is_trponr else TRPOR if self.is_trpor else SKRL_TRPO_WITH_COLLECT
    if self.is_trponr:
      self.agent = agent_class(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=self.env.observation_space,
        action_space=self.env.action_space,
        device=self.device,
        ent_coef=self.ent_coef,
        noise_type=self.noise_type,
        noise_level=self.noise_level,
      )
    elif self.is_trpor:
      self.agent = agent_class(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=self.env.observation_space,
        action_space=self.env.action_space,
        device=self.device,
        ent_coef=self.ent_coef,
      )
    else:
      self.agent = agent_class(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=self.env.observation_space,
        action_space=self.env.action_space,
        device=self.device,
      )

  def learn(self, total_timesteps: int, log_interval=1):
    trainer = SequentialTrainer(
      cfg={
        "timesteps": total_timesteps,
        "headless": True,
        "log_interval": log_interval,
      },
      env=self.env,
      agents=self.agent,
    )
    trainer.train()


def create_param_samplers(n_timesteps):
  defaults = {
    "n_critic_updates": 20,
    "cg_max_steps": 20,
    "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    "n_timesteps": n_timesteps,
    "n_envs": 4,
  }

  trpor_defaults = {"ent_coef": 0.0}
  trpo_defaults = {
    "cg_damping": 0.1,
    "line_search_shrinking_factor": 0.8,
  }

  def sample_trpor_params(trial):
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

    if batch_size > n_steps:
      batch_size = n_steps

    params = {
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "cg_max_steps": defaults["cg_max_steps"],
      "n_critic_updates": defaults["n_critic_updates"],
      "target_kl": target_kl,
      "learning_rate": learning_rate,
      "gae_lambda": gae_lambda,
      "ent_coef": ent_coef,
      "policy_kwargs": dict(
        net_arch=defaults["net_arch"],
      ),
    }
    return params

  def sample_trpo_params(trial):
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

    if batch_size > n_steps:
      batch_size = n_steps

    params = {
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "cg_damping": trpo_defaults["cg_damping"],
      "cg_max_steps": defaults["cg_max_steps"],
      "line_search_shrinking_factor": trpo_defaults["line_search_shrinking_factor"],
      "n_critic_updates": defaults["n_critic_updates"],
      "target_kl": target_kl,
      "learning_rate": learning_rate,
      "gae_lambda": gae_lambda,
      "policy_kwargs": dict(
        net_arch=defaults["net_arch"],
      ),
    }
    return params

  def sample_trponr_params(trial):
    params = sample_trpor_params(trial)
    params["noise_type"] = trial.suggest_categorical("noise_type", ["uniform", "gaussian", "laplace", "bernoulli"])
    params["noise_level"] = trial.suggest_float("noise_level", 0.0, 1.0)
    return params

  return sample_trpor_params, sample_trpo_params, sample_trponr_params


def objective(trial, config, env_id, n_timesteps, device, log_dir):
  entropy_level = -0.3 if "with_noise" in config and "trponr" not in config else 0.0

  env = gym.make(env_id)
  if entropy_level != 0:
    noise_configs = [
      {"component": "reward", "type": "uniform", "entropy_level": entropy_level},
      {"component": "action", "type": "uniform", "entropy_level": entropy_level},
    ]
    env = EntropyInjectionWrapper(env, noise_configs=noise_configs)
  env = AutoPlotProgressWrapper(env, config, env_id, log_dir=log_dir)

  sample_trpor_params, sample_trpo_params, sample_trponr_params = create_param_samplers(n_timesteps)

  if "trponr" in config:
    params = sample_trponr_params(trial)
    if "no_noise" in config:
      params["noise_level"] = 0
    model = TRPOWrapper(env=env, device=device, is_trpor=True, is_trponr=True, **params)
  elif "trpor" in config:
    params = sample_trpor_params(trial)
    model = TRPOWrapper(env=env, device=device, is_trpor=True, **params)
  else:
    params = sample_trpo_params(trial)
    model = TRPOWrapper(env=env, device=device, is_trpor=False, **params)

  model.learn(total_timesteps=n_timesteps)

  rewards = model.agent.rewards
  valid_rewards = [r for r in rewards if not np.isnan(r) and not np.isinf(r)]
  max_reward = max(valid_rewards) if valid_rewards else -1e6

  env.close()
  return max_reward


def compare_max_rewards(
  config,
  env_id="HumanoidStandup-v5",
  n_timesteps=100_000,
  n_trials=100,
  device="cuda",
):
  log_dir = ".omega/finetune_logs/"
  os.makedirs(log_dir, exist_ok=True)

  max_rewards = {}
  batch_size = 10  # Number of trials per batch

  # Create study for the specified config
  optuna_dir = ".omega/optuna_studies"
  os.makedirs(optuna_dir, exist_ok=True)
  sampler = optuna.samplers.TPESampler()
  pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

  storage_path = os.path.join(optuna_dir, f"{config}_storage")
  storage = JournalStorage(JournalFileBackend(storage_path))
  study_name = f"{config}_{env_id}_study"
  study = optuna.create_study(
    sampler=sampler,
    pruner=pruner,
    storage=storage,
    study_name=study_name,
    load_if_exists=True,
    direction="maximize",
  )

  # Run optimization in batches
  trials_remaining = n_trials
  batch_number = 1

  while trials_remaining > 0:
    current_batch_size = min(batch_size, trials_remaining)
    print(f"\nRunning batch {batch_number} for {config} ({current_batch_size} trials for {n_timesteps} timesteps)...")

    if len(study.get_trials(states=[TrialState.COMPLETE])) > 0:
      print(f"\nBest Trial Stats for {config.upper().replace('_', ' ')} (Batch {batch_number}):")
      bt = study.best_trial
      print(f"  Best Parameters: {bt.params}")
      print(f"  Best Max Reward: {bt.value:.2f}")
    else:
      print(f"{config.upper().replace('_', ' ')}: No trials yet")

    study.optimize(
      lambda trial: objective(trial, config, env_id, n_timesteps, device, log_dir),
      n_trials=current_batch_size,
    )
    print(f"Completed {current_batch_size} trials for {config}.")

    trials_remaining -= current_batch_size
    batch_number += 1

    # Save final results for this config
    if len(study.get_trials(states=[TrialState.COMPLETE])) > 0:
      results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
      }
    else:
      results = {
        "best_params": {},
        "best_value": None,
      }
    config_name = config.upper().replace("_", " ")
    path = os.path.join(log_dir, f"{config_name}.yaml")
    with open(path, "w") as f:
      yaml.safe_dump(results, f, default_flow_style=False)

  # Print final summary for this config
  print(f"\nFinal Best Trial Stats for {config.upper().replace('_', ' ')}:")
  if len(study.get_trials(states=[TrialState.COMPLETE])) > 0:
    print(f"  Best Parameters: {study.best_params}")
    print(f"  Best Value: {study.best_value:.2f}")
    max_rewards[config] = study.best_value
  else:
    print("  No completed trials.")
    max_rewards[config] = None

  reward = max_rewards[config]
  reward_str = f"{reward:.2f}" if reward is not None else "N/A"
  print(f"\nMax Reward for {config.upper().replace('_', ' ')}: {reward_str}")


def main():
  parser = argparse.ArgumentParser(description="Run training for specified model/config.")
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=[
      "trpo_no_noise",
      "trpo_with_noise",
      "trpor_no_noise",
      "trpor_with_noise",
      "trponr_no_noise",
      "trponr_with_noise",
    ],
    help="The model configuration to run.",
  )
  parser.add_argument("--env_id", type=str, default="HumanoidStandup-v5", help="Environment ID.")
  parser.add_argument(
    "--n_timesteps",
    type=int,
    default=1_000_000,
    help="Number of timesteps for training.",
  )
  parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for optimization.")
  parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu or cuda).")

  args = parser.parse_args()

  compare_max_rewards(
    config=args.config,
    env_id=args.env_id,
    n_timesteps=args.n_timesteps,
    n_trials=args.n_trials,
    device=args.device,
  )


if __name__ == "__main__":
  main()
