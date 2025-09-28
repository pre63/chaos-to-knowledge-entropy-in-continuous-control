SHELL := /bin/bash

OS := $(shell uname -s)

DEVICE=auto

n_jobs=15 # Default number of jobs to run in parallel
model=trpo # Default model to train

optimize=False # Default to not optimize hyperparameters
trials=1000 # Default number of trials for hyperparameter optimization
max_trials=1000 # Default maximum number of trials for hyperparameter optimization

n_timesteps=0 # Default number of timesteps to train for
n_eval_timesteps=1000000 # Default number of timesteps to evaluate for

env=Humanoid-v5 # Default environment to train on

default: install

board:
	@mkdir -p .logs
	@. .venv/bin/activate && PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006

ubuntu:
	@if [ "$(OS)" != "Linux" ]; then \
		YPATH=".noise/2025-03-10_07-15-27/GenPPO_100000_reward_action_5_runs.yml;.noise/2025-03-09_07-39-20/GenPPO_100000_reward_action_5_runs.yml;.noise/2025-03-08_10-51-44/GenPPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_18-34-02/GenTRPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/PPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/TRPO_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/TRPOER_100000_reward_action_5_runs.yml;.noise/2025-03-05_02-24-05/TRPOR_100000_reward_action_5_runs.yml"
	elif ! command -v lsb_release > /dev/null; then \
		echo "lsb_release not found, skipping Ubuntu setup."; \
	elif ! lsb_release -a 2>/dev/null | grep -q "Ubuntu"; then \
		echo "Not an Ubuntu system, skipping."; \
	else \
		echo "Running Ubuntu setup..."; \
		sudo apt-get update && \
		sudo apt-get -y install python3-dev swig build-essential cmake && \
		sudo apt-get -y install python3.12-venv python3.12-dev && \
		sudo apt-get -y install swig python-box2d; \
	fi

mac:
	@if [ "$(OS)" != "Darwin" ]; then \
		echo "Not a macOS system, skipping macOS setup."; \
	elif ! command -v sw_vers > /dev/null; then \
		echo "sw_vers not found, skipping macOS setup."; \
	elif ! sw_vers | grep -q "macOS"; then \
		echo "Not a macOS system, skipping."; \
	else \
		echo "Running macOS setup..."; \
		brew install python@3.12 box2d swig; \
	fi

venv:
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

install: ubuntu mac venv

fix:
	@echo "Will run black and isort on modified, added, untracked, or staged Python files"
	@changed_files=$$(git diff --name-only --diff-filter=AM | grep '\.py$$'); \
	untracked_files=$$(git ls-files --others --exclude-standard | grep '\.py$$'); \
	staged_files=$$(git diff --name-only --cached | grep '\.py$$'); \
	all_files=$$(echo "$$changed_files $$untracked_files $$staged_files" | tr ' ' '\n' | sort -u); \
	if [ ! -z "$$all_files" ]; then \
		. .venv/bin/activate && isort --multi-line=0 --line-length=100 $$all_files && black .; \
	else \
		echo "No modified, added, untracked, or staged Python files"; \
	fi

clean:
	@echo "Cleaning up"
	@rm -rf __pycache__/
	@rm -rf .venv

train_Humanoid-v5_trpo: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.ablation --env Humanoid-v5 --model trpo --device $(DEVICE)

train_Humanoid-v5_gentrpo: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.ablation --env Humanoid-v5 --model gentrpo --device $(DEVICE)

train_Humanoid-v5_grentrpo-ne: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.ablation --env Humanoid-v5 --model grentrpo-ne --device $(DEVICE)

train_HumanoidStandup-v5_trpo: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.ablation --env HumanoidStandup-v5 --model trpo --device $(DEVICE)

train_HumanoidStandup-v5_gentrpo: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.ablation --env HumanoidStandup-v5 --model gentrpo --device $(DEVICE)

train_HumanoidStandup-v5_grentrpo-ne: fix
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.ablation --env HumanoidStandup-v5 --model grentrpo-ne --device $(DEVICE)

ablation: train_Humanoid-v5_trpo train_HumanoidStandup-v5_trpo train_Humanoid-v5_gentrpo  train_HumanoidStandup-v5_gentrpo train_HumanoidStandup-v5_grentrpo-ne train_Humanoid-v5_grentrpo-ne


plots_no_fix: 
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.plots

plots: fix plots_no_fix

tables_no_fix: 
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.tables

tables: fix tables_no_fix

gen_report_no_fix: 
	@source .venv/bin/activate; \
	PYTHONPATH=. python -m scripts.report

gen_report: fix gen_report_no_fix

report: fix plots_no_fix tables_no_fix gen_report_no_fix