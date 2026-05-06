# train_franka_stage2.py

import os

# 반드시 mujoco / numpy / torch / env import보다 먼저 설정
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from franka_stage2_env import FrankaStage2PrecisionEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel PPO training for Franka Stage2 Precision Env"
    )

    parser.add_argument(
        "--stage1-model",
        type=str,
        default="./models/stage1/ppo_franka_stage1_reach_pose_final.zip",
        help="Path to the trained Stage1 PPO model.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Additional training timesteps for Stage2.",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments.",
    )

    parser.add_argument(
        "--orientation-level",
        type=str,
        default="small",
        choices=["none", "small", "medium"],
        help="Target orientation randomization level.",
    )

    parser.add_argument(
        "--randomize-target-pos",
        action="store_true",
        help="Randomize target position.",
    )

    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Gymnasium environment checker before training.",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default="ppo_franka_stage2_precision_final",
        help="Final model save name without .zip",
    )

    return parser.parse_args()


def make_env(rank: int, args):
    """
    SubprocVecEnv용 환경 생성 함수.
    rank는 병렬 환경 번호입니다.
    """

    def _init():
        env = FrankaStage2PrecisionEnv(
            render_mode=None,
            print_step_info=False,
            max_steps=120,
            randomize_target_pos=args.randomize_target_pos,
            orientation_level=args.orientation_level,
            use_adaptive_action_scale=True,
        )

        env = Monitor(env)
        return env

    return _init


class RewardTermsTensorboardCallback(BaseCallback):
    """
    info["reward_terms"]에 들어있는 각 reward 항을 TensorBoard에 기록합니다.

    기록되는 항목 예:
        reward_terms_step_mean/dist
        reward_terms_step_mean/ori
        reward_terms_step_mean/progress
        reward_terms_step_mean/action_smooth
        metrics_step_mean/dist_to_goal
        metrics_step_mean/orientation_error
        terminal/success_rate_batch
        terminal/final_dist_mean
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # ----------------------------------------
        # 1. 매 step마다 reward term 평균 기록
        # ----------------------------------------
        reward_term_values = {}

        metric_keys = [
            "dist_to_goal",
            "orientation_error",
            "linear_vel_norm",
            "angular_vel_norm",
            "action_norm",
            "action_smooth",
            "subgoal_jump_pos",
            "subgoal_jump_rot",
            "workspace_violation",
            "success_counter",
            "max_step_size",
            "max_rot_step",
        ]

        metric_values = {key: [] for key in metric_keys}

        for info in infos:
            reward_terms = info.get("reward_terms", {})

            for key, value in reward_terms.items():
                reward_term_values.setdefault(key, []).append(float(value))

            for key in metric_keys:
                if key in info:
                    metric_values[key].append(float(info[key]))

        for key, values in reward_term_values.items():
            if len(values) > 0:
                self.logger.record(
                    f"reward_terms_step_mean/{key}",
                    float(np.mean(values)),
                )

        for key, values in metric_values.items():
            if len(values) > 0:
                self.logger.record(
                    f"metrics_step_mean/{key}",
                    float(np.mean(values)),
                )

        # ----------------------------------------
        # 2. episode 종료 시 terminal metric 기록
        # ----------------------------------------
        terminal_success = []
        terminal_failure = []
        terminal_dist = []
        terminal_ori = []
        terminal_lin_vel = []
        terminal_ang_vel = []
        terminal_reward = []

        for done, info in zip(dones, infos):
            if not done:
                continue

            terminal_success.append(float(info.get("success", False)))
            terminal_failure.append(float(info.get("failure", False)))

            if "dist_to_goal" in info:
                terminal_dist.append(float(info["dist_to_goal"]))

            if "orientation_error" in info:
                terminal_ori.append(float(info["orientation_error"]))

            if "linear_vel_norm" in info:
                terminal_lin_vel.append(float(info["linear_vel_norm"]))

            if "angular_vel_norm" in info:
                terminal_ang_vel.append(float(info["angular_vel_norm"]))

            if "reward_total" in info:
                terminal_reward.append(float(info["reward_total"]))

        if len(terminal_success) > 0:
            self.logger.record(
                "terminal/success_rate_batch",
                float(np.mean(terminal_success)),
            )

        if len(terminal_failure) > 0:
            self.logger.record(
                "terminal/failure_rate_batch",
                float(np.mean(terminal_failure)),
            )

        if len(terminal_dist) > 0:
            self.logger.record(
                "terminal/final_dist_mean",
                float(np.mean(terminal_dist)),
            )

        if len(terminal_ori) > 0:
            self.logger.record(
                "terminal/final_ori_mean",
                float(np.mean(terminal_ori)),
            )

        if len(terminal_lin_vel) > 0:
            self.logger.record(
                "terminal/final_linear_vel_mean",
                float(np.mean(terminal_lin_vel)),
            )

        if len(terminal_ang_vel) > 0:
            self.logger.record(
                "terminal/final_angular_vel_mean",
                float(np.mean(terminal_ang_vel)),
            )

        if len(terminal_reward) > 0:
            self.logger.record(
                "terminal/final_reward_mean",
                float(np.mean(terminal_reward)),
            )

        return True


def main():
    args = parse_args()

    torch.set_num_threads(1)

    print("==============================================")
    print("Franka Stage2 PPO + TSID Parallel Training")
    print("==============================================")
    print(f"Stage1 model          : {args.stage1_model}")
    print(f"Total timesteps       : {args.timesteps}")
    print(f"Number of envs        : {args.n_envs}")
    print(f"Orientation level     : {args.orientation_level}")
    print(f"Random target pos     : {args.randomize_target_pos}")
    print("==============================================")

    if not os.path.exists(args.stage1_model):
        raise FileNotFoundError(f"Stage1 model not found: {args.stage1_model}")

    # ----------------------------------------
    # check_env는 SubprocVecEnv가 아니라 단일 env로 검사
    # ----------------------------------------
    if args.check_env:
        print("Checking single Stage2 environment...")

        test_env = FrankaStage2PrecisionEnv(
            render_mode=None,
            print_step_info=False,
            max_steps=120,
            randomize_target_pos=args.randomize_target_pos,
            orientation_level=args.orientation_level,
            use_adaptive_action_scale=True,
        )

        check_env(test_env)
        test_env.close()

        print("Environment check passed.")

    # ----------------------------------------
    # 저장 폴더
    # ----------------------------------------
    os.makedirs("./models/stage2", exist_ok=True)
    os.makedirs("./models/stage2/checkpoints", exist_ok=True)
    os.makedirs("./ppo_tensorboard/stage2", exist_ok=True)

    # ----------------------------------------
    # 병렬 환경 생성
    # ----------------------------------------
    print(f"Creating {args.n_envs} parallel environments...")

    env = SubprocVecEnv(
        [make_env(i, args) for i in range(args.n_envs)],
        start_method="fork",
    )

    # ----------------------------------------
    # Stage1 모델 불러와서 Stage2 환경에 연결
    # observation/action space가 같아야 load 가능
    # ----------------------------------------
    print("Loading Stage1 PPO model...")

    model = PPO.load(
        args.stage1_model,
        env=env,
        tensorboard_log="./ppo_tensorboard/stage2/",
        print_system_info=True,
    )

    # 병렬환경에서는 callback save_freq를 n_envs로 나눠주는 게 일반적
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path="./models/stage2/checkpoints/",
        name_prefix="ppo_franka_stage2_precision",
    )

    reward_terms_callback = RewardTermsTensorboardCallback()

    callback = CallbackList(
        [
            reward_terms_callback,
            checkpoint_callback,
        ]
    )

    print("Start Stage2 training...")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    save_path = f"./models/stage2/{args.save_name}"
    model.save(save_path)

    print("==============================================")
    print(f"Stage2 model saved: {save_path}.zip")
    print("==============================================")

    env.close()


if __name__ == "__main__":
    main()