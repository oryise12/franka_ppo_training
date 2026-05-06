# play_franka_stage2.py

import os

# franka_stage2_env import 전에 설정해야 함
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np

from stable_baselines3 import PPO

from franka_stage2_env import FrankaStage2PrecisionEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play / evaluate trained PPO policy on Franka Stage2 environment."
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default="./models/stage2/ppo_franka_stage2_precision_final.zip",
        help="Path to trained PPO model .zip file.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--orientation-level",
        type=str,
        default="small",
        choices=["none", "small", "medium"],
    )

    parser.add_argument(
        "--randomize-target-pos",
        action="store_true",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
    )

    parser.add_argument(
        "--print-step-info",
        action="store_true",
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

    env = FrankaStage2PrecisionEnv(
        render_mode=render_mode,
        print_step_info=args.print_step_info,
        max_steps=120,
        randomize_target_pos=args.randomize_target_pos,
        orientation_level=args.orientation_level,
        use_adaptive_action_scale=True,
    )

    print(f"1. Stage2 PPO 모델 불러오는 중: {model_path}")
    model = PPO.load(model_path, env=env)

    episode_rewards = []
    final_dist_errors = []
    final_ori_errors = []
    final_linear_vels = []
    final_angular_vels = []
    success_list = []
    step_counts = []

    print("2. Stage2 정책 테스트 시작")

    for ep in range(args.episodes):
        obs, info = env.reset()

        done = False
        episode_reward = 0.0
        step_count = 0

        final_dist = np.nan
        final_ori = np.nan
        final_linear_vel = np.nan
        final_angular_vel = np.nan
        success = False

        reward_term_sums = {}

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=args.deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += float(reward)
            step_count += 1
            done = terminated or truncated

            final_dist = info.get("dist_to_goal", np.nan)
            final_ori = info.get("orientation_error", np.nan)
            final_linear_vel = info.get("linear_vel_norm", np.nan)
            final_angular_vel = info.get("angular_vel_norm", np.nan)
            success = bool(info.get("success", False))

            reward_terms = info.get("reward_terms", {})
            for key, value in reward_terms.items():
                reward_term_sums[key] = reward_term_sums.get(key, 0.0) + float(value)

        episode_rewards.append(episode_reward)
        final_dist_errors.append(final_dist)
        final_ori_errors.append(final_ori)
        final_linear_vels.append(final_linear_vel)
        final_angular_vels.append(final_angular_vel)
        success_list.append(success)
        step_counts.append(step_count)

        print(
            f"[Episode {ep + 1:02d}] "
            f"reward={episode_reward:.3f} | "
            f"steps={step_count:03d} | "
            f"success={success} | "
            f"final_dist={final_dist:.4f} m | "
            f"final_ori={final_ori:.4f} rad | "
            f"final_v={final_linear_vel:.4f} m/s | "
            f"final_w={final_angular_vel:.4f} rad/s"
        )

        # episode별 reward term 합이 보고 싶으면 주석 해제
        # print("  reward term sums:")
        # for key, value in sorted(reward_term_sums.items()):
        #     print(f"    {key}: {value:.4f}")

    env.close()

    print("\n================ Stage2 Test Summary ================")
    print(f"Episodes              : {args.episodes}")
    print(f"Deterministic         : {args.deterministic}")
    print(f"Orientation level     : {args.orientation_level}")
    print(f"Random target pos     : {args.randomize_target_pos}")
    print(f"Mean reward           : {np.mean(episode_rewards):.3f}")
    print(f"Mean steps            : {np.mean(step_counts):.1f}")
    print(f"Mean final dist       : {np.mean(final_dist_errors):.4f} m")
    print(f"Std final dist        : {np.std(final_dist_errors):.4f} m")
    print(f"Mean final ori        : {np.mean(final_ori_errors):.4f} rad")
    print(f"Std final ori         : {np.std(final_ori_errors):.4f} rad")
    print(f"Mean final lin vel    : {np.mean(final_linear_vels):.4f} m/s")
    print(f"Mean final ang vel    : {np.mean(final_angular_vels):.4f} rad/s")
    print(f"Success rate          : {100.0 * np.mean(success_list):.1f} %")
    print("======================================================")


if __name__ == "__main__":
    main()