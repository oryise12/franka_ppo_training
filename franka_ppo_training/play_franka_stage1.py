import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np

from stable_baselines3 import PPO
from franka_stage1_env import FrankaStage1ReachPoseEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test trained PPO policy for Franka Stage 1 PPO + TSID environment."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/stage1/ppo_franka_stage1_reach_pose_final.zip",
        help="Path to trained PPO model .zip file.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of test episodes.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy action.",
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
        help="Randomize target position during test.",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without MuJoCo viewer.",
    )

    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"

    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    render_mode = None if args.no_render else "human"

    env = FrankaStage1ReachPoseEnv(
        render_mode=render_mode,
        print_step_info=True,
        max_steps=100,
        randomize_target_pos=args.randomize_target_pos,
        orientation_level=args.orientation_level,
    )

    print(f"1. 학습된 PPO 모델 불러오는 중: {model_path}")
    model = PPO.load(model_path, env=env)

    episode_rewards = []
    final_dist_errors = []
    final_ori_errors = []
    success_list = []

    print("2. 정책 테스트 시작")

    for ep in range(args.episodes):
        obs, info = env.reset()

        done = False
        episode_reward = 0.0
        step_count = 0

        final_dist = None
        final_ori = None
        success = False

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=args.deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            final_dist = info.get("dist_to_goal", None)
            final_ori = info.get("orientation_error", None)
            success = bool(info.get("success", False))

        episode_rewards.append(episode_reward)

        if final_dist is not None:
            final_dist_errors.append(final_dist)

        if final_ori is not None:
            final_ori_errors.append(final_ori)

        success_list.append(success)

        print(
            f"[Episode {ep + 1:02d}] "
            f"reward={episode_reward:.3f} | "
            f"steps={step_count:03d} | "
            f"success={success} | "
            f"final_dist={final_dist:.4f} m | "
            f"final_ori={final_ori:.4f} rad"
        )

    try:
        env.close()
    except Exception as e:
        print(f"viewer close 중 경고 발생: {e}")

    print("\n================ Test Summary ================")
    print(f"Episodes           : {args.episodes}")
    print(f"Deterministic      : {args.deterministic}")
    print(f"Orientation level  : {args.orientation_level}")
    print(f"Random target pos  : {args.randomize_target_pos}")
    print(f"Mean reward        : {np.mean(episode_rewards):.3f}")

    if final_dist_errors:
        print(f"Mean final dist    : {np.mean(final_dist_errors):.4f} m")
        print(f"Std final dist     : {np.std(final_dist_errors):.4f} m")

    if final_ori_errors:
        print(f"Mean final ori     : {np.mean(final_ori_errors):.4f} rad")
        print(f"Std final ori      : {np.std(final_ori_errors):.4f} rad")

    print(f"Success rate       : {100.0 * np.mean(success_list):.1f} %")
    print("==============================================")


if __name__ == "__main__":
    main()