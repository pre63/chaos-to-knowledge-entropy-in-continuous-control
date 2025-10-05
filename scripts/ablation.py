# scripts/ablation.py
import argparse

import torch as th

from scripts.ablation_sb3 import run_ablation_sb3
from scripts.ablation_skrl import run_ablation_skrl

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", type=str, default=None, help="Environment ID")
  parser.add_argument("--model", type=str, default=None, help="Model variant")
  parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., cuda:0, cpu)")
  parser.add_argument("--framework", type=str, default="sb3", choices=["sb3", "skrl"], help="Framework to use")
  parser.add_argument("--noise", type=str, default="none", choices=["both", "action", "reward", "none"], help="Noise preset")
  args = parser.parse_args()
  if args.device == "auto":
    import torch as th

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
  else:
    device = th.device(args.device)
  if args.framework == "sb3":
    run_ablation_sb3(args.env, args.model, device, args.noise)
  elif args.framework == "skrl":
    run_ablation_skrl(args.env, args.model, device, args.noise)
