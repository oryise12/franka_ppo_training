import argparse
import time
import warnings

import numpy as np
from stable_baselines3 import PPO

from franka_env import FrankaTsidEnv

warnings.filterwarnings("ignore", message="Converted matrix .* to scipy.sparse.csc_matrix.*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a trained PPO pose-obstacle policy in MuJoCo viewer."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ppo_franka_pose_obstacle_v1_seed0_final.zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--viewer-sync-interval", type=int, default=10)
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--continue-after-collision", action="store_true")
    return parser.parse_args()


def is_success(info):
    return (
        info["dist_to_goal"] < 0.05
        and info["orientation_error"] < 0.20
        and not info["obstacle_collision"]
    )


def main():
    args = parse_args()

    print("1. pose-obstacle 관전 환경을 생성합니다.")
    env = FrankaTsidEnv(
        render_mode="human",
        print_step_info=False,
        viewer_sync_interval=args.viewer_sync_interval,
    )

    print(f"2. PPO 모델을 불러옵니다: {args.model}")
    model = PPO.load(args.model, env=env, device=args.device)

    max_steps = max(1, int(args.duration / env.rl_dt))
    print_interval_steps = max(1, int(args.print_interval / env.rl_dt))
    deterministic = not args.stochastic

    episode_results = []

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            min_clearance = np.inf
            last_info = None
            done_reason = "timeout"

            print(
                f"\nEp {ep}/{args.episodes} | "
                f"target={env.target_bottle_pos.round(3)} | "
                f"obs={env.obstacle_pos.round(3)} r={env.obstacle_radius:.3f}"
            )

            for step in range(1, max_steps + 1):
                step_start = time.time()
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                last_info = info
                min_clearance = min(min_clearance, info["obstacle_clearance"])

                if step == 1 or step % print_interval_steps == 0:
                    sim_t = step * env.rl_dt
                    print(
                        f"t={sim_t:3.1f}s "
                        f"R={reward:+.2f} "
                        f"pos={info['dist_to_goal']:.3f}m "
                        f"ori={info['orientation_error']:.3f}rad "
                        f"clr={info['obstacle_clearance']:.3f}m "
                        f"col={int(info['obstacle_collision'])}",
                        flush=True,
                    )

                if terminated or truncated:
                    if info["obstacle_collision"]:
                        done_reason = "collision"
                        if args.continue_after_collision and not truncated:
                            continue
                    elif is_success(info):
                        done_reason = "success"
                    elif truncated:
                        done_reason = "truncated"
                    else:
                        done_reason = "terminated"
                    break

                if not args.no_real_time:
                    elapsed = time.time() - step_start
                    time.sleep(max(0.0, env.rl_dt - elapsed))

            if last_info is None:
                continue

            success = is_success(last_info)
            episode_results.append(
                {
                    "success": success,
                    "collision": last_info["obstacle_collision"],
                    "pos_err": last_info["dist_to_goal"],
                    "ori_err": last_info["orientation_error"],
                    "min_clearance": min_clearance,
                    "steps": step,
                    "done_reason": done_reason,
                }
            )

            print(
                f"Ep {ep} done | reason={done_reason} | "
                f"ok={int(success)} | "
                f"pos={last_info['dist_to_goal']:.3f}m | "
                f"ori={last_info['orientation_error']:.3f}rad | "
                f"min_clr={min_clearance:.3f}m | "
                f"steps={step}"
            )

    except KeyboardInterrupt:
        print("\n사용자 중단으로 테스트를 종료합니다.")
    finally:
        env.close()

    if not episode_results:
        return

    success_rate = np.mean([r["success"] for r in episode_results]) * 100.0
    collision_rate = np.mean([r["collision"] for r in episode_results]) * 100.0
    mean_pos_err = np.mean([r["pos_err"] for r in episode_results])
    mean_ori_err = np.mean([r["ori_err"] for r in episode_results])
    mean_min_clearance = np.mean([r["min_clearance"] for r in episode_results])
    mean_steps = np.mean([r["steps"] for r in episode_results])

    print(
        f"\nSummary | eps={len(episode_results)} | "
        f"success={success_rate:.1f}% | collision={collision_rate:.1f}% | "
        f"pos={mean_pos_err:.3f}m | ori={mean_ori_err:.3f}rad | "
        f"clr={mean_min_clearance:.3f}m | steps={mean_steps:.1f}"
    )


if __name__ == "__main__":
    main()
