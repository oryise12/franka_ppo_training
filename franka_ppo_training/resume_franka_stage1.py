import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from franka_stage1_env import FrankaStage1ReachPoseEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume PPO training for Franka Stage 1 PPO + TSID environment."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/stage1/ppo_franka_stage1_reach_pose_final.zip",
        help="Path to the pretrained PPO model .zip file.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Additional timesteps for resumed training.",
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
        help="Randomize target position during resumed training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("1. Stage 1 Franka PPO + TSID 환경 생성 중...")

    env = FrankaStage1ReachPoseEnv(
        render_mode=None,
        print_step_info=False,
        max_steps=100,
        randomize_target_pos=args.randomize_target_pos,
        orientation_level=args.orientation_level,
    )

    os.makedirs("./models/stage1", exist_ok=True)
    os.makedirs("./models/stage1/checkpoints", exist_ok=True)
    os.makedirs("./ppo_tensorboard/stage1_resume", exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"불러올 모델을 찾을 수 없습니다: {args.model_path}"
        )

    print(f"2. 기존 PPO 모델 불러오는 중: {args.model_path}")

    model = PPO.load(
        args.model_path,
        env=env,
        tensorboard_log="./ppo_tensorboard/stage1_resume/",
        print_system_info=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/stage1/checkpoints/",
        name_prefix="ppo_franka_stage1_reach_pose_resume",
    )

    print(f"3. 추가 학습 시작: {args.timesteps} timesteps")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    save_path = "./models/stage1/ppo_franka_stage1_reach_pose_resumed"
    model.save(save_path)
    print(f"4. 재학습 모델 저장 완료: {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()