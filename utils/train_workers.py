"""
Worker functions for parallel/seeded GPU training.

Must live in a .py file (not the notebook) so that ProcessPoolExecutor with
spawn start method can import them in child processes.
(CUDA cannot be re-initialized in forked subprocesses — spawn is required.)
"""
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories

from utils.common import make_env_kwargs


class AdvantageStatsCallback(BaseCallback):
    """
    Logs advantage mean, std, and variance after every rollout.

    Works for on-policy algos (A2C, PPO): on_rollout_end() is called by
    OnPolicyAlgorithm immediately after compute_returns_and_advantage(),
    so rollout_buffer.advantages is always fully populated at that point.
    Off-policy algos (e.g. SAC) have no rollout_buffer -- safely skipped.

    Logged keys (visible in progress.csv and TensorBoard):
        train/advantage_mean
        train/advantage_std
        train/advantage_var
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> None:
        if not hasattr(self.model, "rollout_buffer"):
            return  # off-policy algos (e.g. SAC) have no rollout_buffer; nothing to log here
        adv = self.model.rollout_buffer.advantages.flatten()
        self.model.logger.record("train/advantage_mean", float(np.mean(adv)))
        self.model.logger.record("train/advantage_std",  float(np.std(adv)))
        self.model.logger.record("train/advantage_var",  float(np.var(adv)))

    def _on_step(self) -> bool:
        return True


def _get_dirs():
    base = os.environ["FINRL_PATH"]
    trained_dir = Path(base) / "models/pilot_models/test"
    results_dir = Path(base) / "logs/test"
    check_and_make_directories([str(trained_dir), str(results_dir)])
    return trained_dir, results_dir


def _maybe_limit_gpu(model_params, gpu_fraction):
    """Set GPU memory fraction only when actually training on CUDA."""
    if model_params.get("device", "cuda") != "cpu":
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=0)


def train_model_with_gpu_limit(df, indicators, algo_name, model_params, timesteps,
                               model_name, gpu_fraction=0.2, policy_kwargs=None,
                               save_dir=None, log_dir=None):
    """
    Train a single model with limited GPU memory.
    policy_kwargs is passed directly to DRLAgent.get_model() (SB3 requirement —
    it controls net_arch and cannot go inside model_params).
    save_dir / log_dir override the default hardcoded paths (backward-compatible).
    """
    _maybe_limit_gpu(model_params, gpu_fraction)

    _default_model_dir, _default_log_dir = _get_dirs()
    TRAINED_MODEL_DIR = Path(save_dir) if save_dir is not None else _default_model_dir
    RESULTS_DIR       = Path(log_dir)  if log_dir  is not None else _default_log_dir
    check_and_make_directories([str(TRAINED_MODEL_DIR), str(RESULTS_DIR)])

    env_kwargs = make_env_kwargs(df, indicators)
    env, _ = StockTradingEnv(df=df, **env_kwargs).get_sb_env()
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    agent = DRLAgent(env=env)
    model = agent.get_model(algo_name, model_kwargs=model_params, policy_kwargs=policy_kwargs)

    tmp_path = str(RESULTS_DIR / model_name)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"[{model_name}] Training started with {gpu_fraction*100:.0f}% GPU memory...")
    trained_model = agent.train_model(
        model=model,
        tb_log_name=model_name,
        total_timesteps=timesteps,
        callbacks=[AdvantageStatsCallback()],
    )

    trained_model.save(str(TRAINED_MODEL_DIR / model_name))
    env.save(str(TRAINED_MODEL_DIR / f"{model_name}_vecnorm.pkl"))
    return f"Completed: {model_name}"


def train_model_seeded(df, indicators, algo_name, model_params, timesteps, model_name,
                       gpu_fraction=0.2, seed=None, policy_kwargs=None,
                       save_dir=None, log_dir=None):
    """
    Seeded variant of train_model_with_gpu_limit.
    seed is forwarded to DRLAgent.get_model() → SB3 set_random_seed()
    (seeds python / numpy / torch / gym / action_space simultaneously).
    Saved as TRAINED_MODEL_DIR/{model_name}_s{seed}.
    save_dir / log_dir override the default hardcoded paths (backward-compatible).
    """
    _maybe_limit_gpu(model_params, gpu_fraction)

    _default_model_dir, _default_log_dir = _get_dirs()
    TRAINED_MODEL_DIR = Path(save_dir) if save_dir is not None else _default_model_dir
    RESULTS_DIR       = Path(log_dir)  if log_dir  is not None else _default_log_dir
    check_and_make_directories([str(TRAINED_MODEL_DIR), str(RESULTS_DIR)])

    env_kwargs = make_env_kwargs(df, indicators)
    env, _ = StockTradingEnv(df=df, **env_kwargs).get_sb_env()
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    agent = DRLAgent(env=env)
    model = agent.get_model(algo_name, model_kwargs=model_params,
                            policy_kwargs=policy_kwargs, seed=seed)

    run_name  = f"{model_name}_s{seed}" if seed is not None else model_name
    tmp_path  = str(RESULTS_DIR / run_name)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"[{run_name}] training  gpu={gpu_fraction*100:.0f}%  seed={seed}")
    trained_model = agent.train_model(
        model=model,
        tb_log_name=run_name,
        total_timesteps=timesteps,
        callbacks=[AdvantageStatsCallback()],
    )

    trained_model.save(str(TRAINED_MODEL_DIR / run_name))
    env.save(str(TRAINED_MODEL_DIR / f"{run_name}_vecnorm.pkl"))
    return f"Completed: {run_name}"
