# sb3/noise.py
import numpy as np
from gymnasium import Wrapper, spaces


class MonitoredEntropyInjectionWrapper(Wrapper):
  def __init__(self, env, noise_configs=None):
    super().__init__(env)
    if not (isinstance(self.action_space, spaces.Box) and isinstance(self.observation_space, spaces.Box)):
      raise ValueError("This wrapper is designed for continuous action and observation spaces only.")
    self.noise_configs = noise_configs if noise_configs is not None else []
    self._validate_configs()
    self.base_std = 1.0
    self.base_range = 1.0
    self.base_scale = 1.0
    self.base_p = 0.5
    self.action_deltas = []
    self.reward_deltas = []

  def _validate_configs(self):
    if not self.noise_configs:
      return
    for config in self.noise_configs:
      required_keys = {"component", "noise_type", "noise_level"}
      if not all(key in config for key in required_keys):
        raise ValueError("Each noise_config must include 'component', 'type', and 'noise_level'.")
      component, noise_type, noise_level = (
        config["component"],
        config["noise_type"],
        config["noise_level"],
      )
      if component not in ["obs", "reward", "action"]:
        raise ValueError("Component must be 'obs', 'reward', or 'action'.")
      if noise_type not in ["gaussian", "uniform", "laplace", "bernoulli"]:
        raise ValueError("Noise type must be 'gaussian', 'uniform', 'laplace', or 'bernoulli'.")
      if noise_type == "bernoulli" and component != "reward":
        raise ValueError("Bernoulli noise is only supported for rewards.")
      if not -1 <= noise_level <= 1:
        raise ValueError("noise_level must be between -1 and 1.")

  def _add_obs_noise(self, obs):
    return obs

  def _add_reward_noise(self, reward):
    noisy = reward
    for config in self.noise_configs:
      if config["component"] == "reward":
        noise_type = config["noise_type"]
        noise_level = abs(config["noise_level"])
        if noise_type == "gaussian":
          noise = np.random.normal(0, noise_level * self.base_std)
        elif noise_type == "uniform":
          noise = np.random.uniform(-noise_level * self.base_range, noise_level * self.base_range)
        elif noise_type == "laplace":
          noise = np.random.laplace(0, noise_level * self.base_scale)
        elif noise_type == "bernoulli":
          p = noise_level * self.base_p
          noise = -reward if np.random.uniform() < p else 0
        noisy += noise
        self.reward_deltas.append(float(noise))
    return noisy

  def _add_action_noise(self, action):
    noisy = action.copy()
    for config in self.noise_configs:
      if config["component"] == "action":
        noise_type = config["noise_type"]
        noise_level = abs(config["noise_level"])
        if noise_type == "gaussian":
          noise = np.random.normal(0, noise_level * self.base_std, size=action.shape)
        elif noise_type == "uniform":
          noise = np.random.uniform(-noise_level * self.base_range, noise_level * self.base_range, size=action.shape)
        elif noise_type == "laplace":
          noise = np.random.laplace(0, noise_level * self.base_scale, size=action.shape)
        noisy += noise
        self.action_deltas.append(float(np.linalg.norm(noise)))
    return noisy

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    return obs, info

  def step(self, action):
    noisy_action = self._add_action_noise(action)
    action_to_use = np.clip(noisy_action, self.action_space.low, self.action_space.high)
    obs, reward, terminated, truncated, info = self.env.step(action_to_use)
    reward = self._add_reward_noise(reward)
    return obs, reward, terminated, truncated, info

  def get_noise_deltas(self, reset=True):
    action_deltas = self.action_deltas.copy()
    reward_deltas = self.reward_deltas.copy()
    if reset:
      self.action_deltas = []
      self.reward_deltas = []
    return action_deltas, reward_deltas
