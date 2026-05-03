# --------- 논문용 PPO + TSID error plot 생성 코드 ----------
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from franka_env import FrankaTsidEnv


def get_random_target():
    """
    여기 수정: interactive_play.py와 같은 target 범위 사용
    """
    x = np.random.uniform(0.35, 0.55)
    y = np.random.uniform(-0.2, 0.2)
    z = np.random.uniform(0.15, 0.5)
    return np.array([x, y, z])


def run_single_trial(model, env, trial_duration=3.0, use_parking_brake=True):
    """
    여기 수정: 3초 동안 하나의 고정 target에 대해 PPO + TSID를 실행하고
    sub-goal tracking error와 goal tracking error를 기록합니다.
    """

    obs, info = env.reset()

    # trial마다 새 최종 목표 지정
    target = get_random_target()
    env.target_bottle_pos = target.copy()

    # PPO가 생성하는 sub-goal은 현재 EE 위치에서 시작
    env.target_ee_pos = env._get_ee_pos().copy()

    # env.rl_dt = 0.1 이므로 기본적으로 3초 = 30 step
    steps = int(trial_duration / env.rl_dt)

    t_log = []
    sub_err_log = []
    goal_err_log = []

    ee_pos_log = []
    subgoal_pos_log = []
    target_pos_log = []

    for k in range(steps):
        t = k * env.rl_dt

        action, _states = model.predict(obs, deterministic=True)

        # interactive_play.py에 있던 주차 브레이크 로직
        if use_parking_brake:
            curr_ee_pos = env._get_ee_pos()
            dist = np.linalg.norm(target - curr_ee_pos)
            if dist < 0.03:
                action = action * 0.1

        obs, reward, terminated, truncated, info = env.step(action)

        # 여기 수정: franka_env.py에서 반환한 info를 저장
        t_log.append(t)
        sub_err_log.append(info["subgoal_tracking_error"])
        goal_err_log.append(info["goal_tracking_error"])

        ee_pos_log.append(info["ee_pos"])
        subgoal_pos_log.append(info["subgoal_pos"])
        target_pos_log.append(info["target_pos"])

        # 논문용 고정 길이 trial이므로 truncated는 무시하고 current_step만 초기화
        if truncated:
            env.current_step = 0

        # 치명적 실패 시 남은 구간은 마지막 값으로 채우고 종료
        if terminated:
            print(f"⚠️ Trial terminated at t={t:.2f}s")
            break

    # 혹시 terminated로 조기 종료되면 길이를 맞춤
    target_len = steps

    def pad_to_len(arr, target_len):
        arr = np.array(arr)
        if len(arr) == 0:
            return np.zeros(target_len)
        if len(arr) < target_len:
            pad_len = target_len - len(arr)
            if arr.ndim == 1:
                arr = np.pad(arr, (0, pad_len), mode="edge")
            else:
                arr = np.vstack([arr, np.repeat(arr[-1][None, :], pad_len, axis=0)])
        return arr

    t_arr = np.arange(target_len) * env.rl_dt
    sub_err_arr = pad_to_len(sub_err_log, target_len)
    goal_err_arr = pad_to_len(goal_err_log, target_len)

    ee_pos_arr = pad_to_len(ee_pos_log, target_len)
    subgoal_pos_arr = pad_to_len(subgoal_pos_log, target_len)
    target_pos_arr = pad_to_len(target_pos_log, target_len)

    return t_arr, sub_err_arr, goal_err_arr, ee_pos_arr, subgoal_pos_arr, target_pos_arr


def run_experiment(model_path, n_trials=10, trial_duration=3.0):
    """
    여기 수정: 10회 trial 수행
    """
    env = FrankaTsidEnv(render_mode=None)
    model = PPO.load(model_path, env=env)

    sub_err_all = []
    goal_err_all = []

    ee_pos_all = []
    subgoal_pos_all = []
    target_pos_all = []

    for i in range(n_trials):
        print(f"\n========== PPO Trial {i+1}/{n_trials} ==========")
        t_arr, sub_err, goal_err, ee_pos, subgoal_pos, target_pos = run_single_trial(
            model,
            env,
            trial_duration=trial_duration,
            use_parking_brake=True
        )

        print(
            f"Final sub-goal error: {sub_err[-1]:.4f} m | "
            f"Final goal error: {goal_err[-1]:.4f} m | "
            f"Min goal error: {np.min(goal_err):.4f} m"
        )

        sub_err_all.append(sub_err)
        goal_err_all.append(goal_err)

        ee_pos_all.append(ee_pos)
        subgoal_pos_all.append(subgoal_pos)
        target_pos_all.append(target_pos)

    env.close()

    sub_err_all = np.vstack(sub_err_all)
    goal_err_all = np.vstack(goal_err_all)

    ee_pos_all = np.array(ee_pos_all)
    subgoal_pos_all = np.array(subgoal_pos_all)
    target_pos_all = np.array(target_pos_all)

    sub_err_mean = np.mean(sub_err_all, axis=0)
    goal_err_mean = np.mean(goal_err_all, axis=0)

    return (
        t_arr,
        sub_err_all,
        goal_err_all,
        sub_err_mean,
        goal_err_mean,
        ee_pos_all,
        subgoal_pos_all,
        target_pos_all
    )


def plot_single_error(
    t_arr,
    err_all,
    err_mean,
    ylabel,
    title,
    output_path,
    mean_color="tab:blue",
    trial_linestyle="--",
    mean_linestyle="-",
    show_trials=3
):
    """
    여기 수정: 그래프 하나를 저장하는 함수
    - 개별 trial 1~3개: 검정색 연한 점선
    - 평균값: 진한 선
    """
    plt.figure(figsize=(8, 4.8))

    n_show = min(show_trials, err_all.shape[0])

    for i in range(n_show):
        plt.plot(
            t_arr,
            err_all[i],
            linestyle=trial_linestyle,
            color="black",
            alpha=0.25,
            linewidth=1.5,
            label="Individual trials" if i == 0 else None
        )

    plt.plot(
        t_arr,
        err_mean,
        linestyle=mean_linestyle,
        color=mean_color,
        linewidth=3.2,
        label="Mean error"
    )
    

    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./models/checkpoints/ppo_franka_delta_v4_150000_steps.zip"
    )
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.expanduser("~/ros2_ws_py/ppo_eval_results")
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    (
        t_arr,
        sub_err_all,
        goal_err_all,
        sub_err_mean,
        goal_err_mean,
        ee_pos_all,
        subgoal_pos_all,
        target_pos_all
    ) = run_experiment(
        model_path=args.model,
        n_trials=args.n_trials,
        trial_duration=args.duration
    )

    # 데이터 저장
    data_path = os.path.join(args.out_dir, "ppo_tsid_error_data.npz")
    np.savez(
        data_path,
        time=t_arr,
        sub_err_trials=sub_err_all,
        goal_err_trials=goal_err_all,
        sub_err_mean=sub_err_mean,
        goal_err_mean=goal_err_mean,
        ee_pos_all=ee_pos_all,
        subgoal_pos_all=subgoal_pos_all,
        target_pos_all=target_pos_all
    )
    print(f"✅ Saved: {data_path}")

    # 그래프 1: Sub-goal tracking error
    plot_single_error(
        t_arr=t_arr,
        err_all=sub_err_all,
        err_mean=sub_err_mean,
        ylabel="Sub-goal tracking error [m]",
        title="PPO Sub-goal Tracking Error with TSID",
        output_path=os.path.join(args.out_dir, "ppo_subgoal_tracking_error.png"),
        mean_color="tab:blue",
        trial_linestyle="--",
        mean_linestyle="--"
    )

    # 그래프 2: Goal tracking error
    plot_single_error(
        t_arr=t_arr,
        err_all=goal_err_all,
        err_mean=goal_err_mean,
        ylabel="Goal tracking error [m]",
        title="PPO Goal Tracking Error with TSID",
        output_path=os.path.join(args.out_dir, "ppo_goal_tracking_error.png"),
        mean_color="tab:red",
        trial_linestyle="--",
        mean_linestyle="-"
    )

    # 표에 넣을 수 있는 간단 지표 출력
    success_rate = np.mean(np.min(goal_err_all, axis=1) < 0.05) * 100.0

    print("\n========== Summary ==========")
    print(f"Mean sub-goal tracking error: {np.mean(sub_err_all):.5f} m")
    print(f"Mean goal tracking error:     {np.mean(goal_err_all):.5f} m")
    print(f"Final goal tracking error:    {np.mean(goal_err_all[:, -1]):.5f} m")
    print(f"Best goal tracking error:     {np.mean(np.min(goal_err_all, axis=1)):.5f} m")
    print(f"Success rate within 5 cm:     {success_rate:.1f} %")


if __name__ == "__main__":
    main()