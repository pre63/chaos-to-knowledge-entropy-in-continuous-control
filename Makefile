SHELL := /bin/bash

OS := $(shell uname -s)

DEVICE=cpu

n_jobs=15 # Default number of jobs to run in parallel
model=trpo # Default model to train

optimize=False # Default to not optimize hyperparameters
trials=160 # Default number of trials for hyperparameter optimization
max_trials=160 # Default maximum number of trials for hyperparameter optimization

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

train:
	@echo "Will train model $(model) on environment $(env) and we will optimize hyperparameters: $(optimize), for $(trials) trials."
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@mkdir -p ".optuna-zoo/$(model)_$(env)"
	@. .venv/bin/activate; PYTHONPATH=. python -u Zoo/Train.py --model=$(model) --env=$(env) --optimize=$(optimize) --n_jobs=$(n_jobs) --trials=$(trials) --max_trials=$(max_trials) --device=$(DEVICE) 2>&1 | tee -a .logs/zoo-$(model)-$(env)-$(shell date +"%Y%m%d").log

eval:
	@echo "Will evaluate model $(model) on environment $(env)"
	@. .venv/bin/activate; PYTHONPATH=. python -u Zoo/Eval.py --n_timesteps=$(n_eval_timesteps) --model=$(model) --env=$(env) 2>&1 | tee -a .logs/eval-$(model)-$(env)-$(shell date +"%Y%m%d").log

report:
	@$(MAKE) fix
	@. .venv/bin/activate; PYTHONPATH=. python -u Zoo/Report.py

list:
	@# List all zoo models and environments combinations
	@echo "Listing all model and environment combinations:"
	@for env in $(zoologyenvs); do \
		for model in $(zoology); do \
			echo "$$env - $$model"; \
		done; \
	done
	@# Calculate and print the total number of combinations
	@zoologyenvs_count=$$(echo "$(zoologyenvs)" | wc -w); \
	zoology_count=$$(echo "$(zoology)" | wc -w); \
	trials=$(trials); \
	total_combinations=$$(echo "$$zoologyenvs_count * $$zoology_count * $$trials" | bc); \
	echo "Total number of combinations: $$total_combinations"

noise:
	@. .venv/bin/activate; CUDA_VISIBLE_DEVICES="" python -m Environments.Noise

noise-plot:
	@. .venv/bin/activate; python -m Environments.NoisePlot

tune:
	#PPO for HalfCheetah-v5 Hopper-v5 Humanoid-v5 HumanoidStandup-v5 Reacher-v5 Swimmer-v5
	$(MAKE) train model=ppo env=HalfCheetah-v5 n_jobs=15 optimize=True
	$(MAKE) train model=ppo env=Hopper-v5 n_jobs=15 optimize=True
	$(MAKE) train model=ppo env=Humanoid-v5 n_jobs=15 optimize=True
	$(MAKE) train model=ppo env=HumanoidStandup-v5 n_jobs=15 optimize=True
	$(MAKE) train model=ppo env=Reacher-v5 n_jobs=15 optimize=True
	$(MAKE) train model=ppo env=Swimmer-v5 n_jobs=15 optimize=True

	#TRPO for HalfCheetah-v5 Hopper-v5 Humanoid-v5 HumanoidStandup-v5 Reacher-v5 Swimmer-v5
	$(MAKE) train model=trpo env=HalfCheetah-v5 n_jobs=15 optimize=True
	$(MAKE) train model=trpo env=Hopper-v5 n_jobs=15 optimize=True
	$(MAKE) train model=trpo env=Humanoid-v5 n_jobs=15 optimize=True
	$(MAKE) train model=trpo env=HumanoidStandup-v5 n_jobs=15 optimize=True
	$(MAKE) train model=trpo env=Reacher-v5 n_jobs=15 optimize=True
	$(MAKE) train model=trpo env=Swimmer-v5 n_jobs=15 optimize=True

plot:
	@rm -rf .plots
	@python Reports/Plot.py --data_dir .noise/final --smooth_window 30 --markers_per_line 4


.PHONY: run
run:
	@while true; do \
		. .venv/bin/activate; CUDA_VISIBLE_DEVICES="" python -m Environments.Noise; \
		if [ $$? -eq 0 ]; then \
			echo "All variants completed. Exiting."; \
			break; \
		else \
			echo "Some variants remain. Restarting..."; \
			sleep 1; \
		fi \
	done