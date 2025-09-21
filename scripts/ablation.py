# Script 1: Run the ablation study with hardcoded parameters for Humanoid-v5 and HumanoidStandup-v5
# This script assumes the necessary classes (e.g., TRPOWrapper, AutoPlotProgressWrapper, EntropyInjectionWrapper) from the provided code are available.
# Modifications to TRPOWrapper are implied: add support for activation_fn, ortho_init, log_std_init, buffer_capacity, normalize_advantage, etc.
# For example, in TRPOWrapper.__init__, add these as self attributes.
# In _setup_model, adjust shared_net to use the specified act_fn, apply ortho_init if True, set log_std_parameter.data.fill_(log_std_init),
# set memory = RandomMemory(memory_size=self.buffer_capacity or self.n_steps, ...),
# set cfg["normalize_advantage"] = self.normalize_advantage.
# Ignore unsupported params like sampling_coef, sde_sample_freq, lr_schedule (or implement if needed).

import os

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from models.gen_trpo import GEN_TRPO
from models.gen_trpo_ne import GEN_TRPO_NE
from models.trpo import SKRL_TRPO_WITH_COLLECT, AutoPlotProgressWrapper, TRPOWrapper


def run_ablation():
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "grentrpo-ne"]
  params_dict = {
    "Humanoid-v5": {
      "trpo": {
        "batch_size": 512,
        "n_steps": 1024,
        "gamma": 0.98,
        "learning_rate": 0.0001001422334677667,
        "line_search_shrinking_factor": 0.7,
        "n_critic_updates": 25,
        "cg_max_steps": 25,
        "cg_damping": 0.5,
        "target_kl": 0.1,
        "gae_lambda": 0.99,
        "net_arch": "small",
        "log_std_init": -1.1830649974937577,
        "ortho_init": False,
        "activation_fn": "relu",
        "lr_schedule": "linear",
        "n_timesteps": 700000,
        "n_envs": 2,
        "normalize_advantage": True,  # Assume default if not specified
      },
      "gentrpo": {
        "n_steps": 512,
        "gamma": 0.95,
        "learning_rate": 0.03648542602737527,
        "n_critic_updates": 20,
        "cg_max_steps": 5,
        "target_kl": 0.03,
        "gae_lambda": 0.95,
        "batch_size": 2048,
        "net_arch": "medium",
        "activation_fn": "tanh",
        "entropy_coef": -0.59,
        "buffer_capacity": 30000,
        "epsilon": 0.0,  # No noise for gentrpo
        "ortho_init": False,
        "n_timesteps": 600000,
        "n_envs": 6,
        "normalize_advantage": True,
      },
      "grentrpo-ne": {
        "n_steps": 512,
        "gamma": 0.95,
        "learning_rate": 0.03648542602737527,
        "n_critic_updates": 20,
        "cg_max_steps": 5,
        "target_kl": 0.03,
        "gae_lambda": 0.95,
        "batch_size": 2048,
        "net_arch": "medium",
        "activation_fn": "tanh",
        "entropy_coef": -0.59,
        "buffer_capacity": 30000,
        "epsilon": 0.15,
        "ortho_init": False,
        "n_timesteps": 600000,
        "n_envs": 6,
        "normalize_advantage": True,
      },
    },
    "HumanoidStandup-v5": {
      "trpo": {
        "batch_size": 64,
        "n_steps": 512,
        "gamma": 0.995,
        "learning_rate": 0.0002661447879672383,
        "line_search_shrinking_factor": 0.8,
        "n_critic_updates": 5,
        "cg_max_steps": 20,
        "cg_damping": 0.5,
        "target_kl": 0.005,
        "gae_lambda": 0.8,
        "net_arch": "medium",
        "log_std_init": -1.3944767094748196,
        "ortho_init": True,
        "activation_fn": "tanh",
        "lr_schedule": "constant",
        "n_timesteps": 1000000,
        "n_envs": 8,
        "normalize_advantage": True,  # Assume default
      },
      "gentrpo": {
        "n_steps": 8,
        "gamma": 0.8,
        "learning_rate": 0.11959944479404305,
        "n_critic_updates": 30,
        "cg_max_steps": 5,
        "target_kl": 0.1,
        "gae_lambda": 1,
        "batch_size": 2048,
        "net_arch": "large",
        "activation_fn": "tanh",
        "entropy_coef": 0.24,
        "buffer_capacity": 30000,
        "epsilon": 0.0,  # No noise for gentrpo
        "ortho_init": True,
        "n_timesteps": 100000,
        "n_envs": 4,
        "normalize_advantage": False,
      },
      "grentrpo-ne": {
        "n_steps": 8,
        "gamma": 0.8,
        "learning_rate": 0.11959944479404305,
        "n_critic_updates": 30,
        "cg_max_steps": 5,
        "target_kl": 0.1,
        "gae_lambda": 1,
        "batch_size": 2048,
        "net_arch": "large",
        "activation_fn": "tanh",
        "entropy_coef": 0.24,
        "buffer_capacity": 30000,
        "epsilon": 0.75,
        "ortho_init": True,
        "n_timesteps": 100000,
        "n_envs": 4,
        "normalize_advantage": False,
      },
    },
  }
  log_dir = ".logs/"
  os.makedirs(log_dir, exist_ok=True)
  for env_id in envs:
    for variant in variants:
      p = params_dict[env_id][variant]
      n_envs = p["n_envs"]

      def make_env():
        return gym.make(env_id)

      env = SyncVectorEnv([make_env for _ in range(n_envs)])
      entropy_level = abs(p["epsilon"]) if variant == "grentrpo-ne" else 0.0
      if entropy_level > 0:
        noise_configs = [
          {"component": "reward", "type": "uniform", "entropy_level": entropy_level},
          {"component": "action", "type": "uniform", "entropy_level": entropy_level},
        ]
        env = EntropyInjectionWrapper(env, noise_configs=noise_configs)
      # env = AutoPlotProgressWrapper(env, variant, env_id, log_dir=log_dir, plot_interval=1000)
      if variant == "trpo":
        agent_class = SKRL_TRPO_WITH_COLLECT
        is_entrpoy = False
      elif variant == "gentrpo":
        agent_class = GEN_TRPO
        is_entrpoy = False
      elif variant == "grentrpo-ne":
        agent_class = GEN_TRPO_NE
        is_entrpoy = True
      noise_type = "uniform" if is_entrpoy else ""
      noise_level = abs(p["epsilon"]) if is_entrpoy else 0.0
      ent_coef = p.get("entropy_coef", 0.0)
      policy_kwargs = {
        "net_arch": {
          "pi": [64, 64] if p["net_arch"] == "small" else [400, 300] if p["net_arch"] == "large" else [256, 256],
          "vf": [64, 64] if p["net_arch"] == "small" else [400, 300] if p["net_arch"] == "large" else [256, 256],
        },
      }
      model = TRPOWrapper(
        env=env,
        learning_rate=p["learning_rate"],
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        gamma=p["gamma"],
        cg_max_steps=p["cg_max_steps"],
        cg_damping=p.get("cg_damping", 0.1),
        line_search_shrinking_factor=p.get("line_search_shrinking_factor", 0.8),
        line_search_max_iter=10,
        n_critic_updates=p["n_critic_updates"],
        gae_lambda=p["gae_lambda"],
        normalize_advantage=p.get("normalize_advantage", True),
        target_kl=p["target_kl"],
        policy_kwargs=policy_kwargs,
        seed=None,
        device="cuda",
        ent_coef=ent_coef,
        noise_type=noise_type,
        noise_level=noise_level,
        # Additional params
        activation_fn=p["activation_fn"],
        ortho_init=p["ortho_init"],
        log_std_init=p.get("log_std_init", 0.0),
        lr_schedule=p.get("lr_schedule", "constant"),
        buffer_capacity=p.get("buffer_capacity", p["n_steps"]),
      )
      model.learn(total_timesteps=p["n_timesteps"])
      env.close()


if __name__ == "__main__":
  run_ablation()
