import argparse
import os
import time
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from franka_env import FrankaTsidEnv

warnings.filterwarnings("ignore", message="Converted matrix .* to scipy.sparse.csc_matrix.*")


class StepPrinterCallback(BaseCallback):
    def __init__(self, print_freq, total_timesteps):
        super().__init__()
        self.print_freq = print_freq
        self.total_timesteps = total_timesteps
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.print_freq <= 0:
            return True

        if self.num_timesteps == 1 or self.num_timesteps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0.0 else 0.0
            progress = 100.0 * self.num_timesteps / self.total_timesteps
            print(
                f"[progress] {self.num_timesteps:,}/{self.total_timesteps:,} "
                f"({progress:.1f}%) | elapsed {elapsed:.1f}s | {fps:.1f} steps/s",
                flush=True,
            )

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO for Franka pose reaching with obstacle avoidance."
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-prefix", type=str, default="ppo_franka_pose_obstacle_v1")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--log-dir", type=str, default="./ppo_tensorboard_pose_obstacle")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.models_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_random_seed(args.seed)

    print("1. Franka pose-obstacle PPO 환경을 생성합니다.")
    env = FrankaTsidEnv(render_mode=None)
    env = Monitor(env)

    if args.check_env:
        print("2. Gymnasium 환경 규격을 검사합니다.")
        check_env(env.unwrapped)
        print("   환경 검사 통과.")
    else:
        print("2. 환경 검사는 건너뜁니다. 필요하면 --check-env 옵션을 사용하세요.")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_dir,
        name_prefix=args.model_prefix,
    )
    progress_callback = StepPrinterCallback(args.print_freq, args.timesteps)
    callbacks = CallbackList([checkpoint_callback, progress_callback])

    if args.resume:
        print(f"3. 기존 모델을 불러와 이어서 학습합니다: {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=args.log_dir, device=args.device)
        reset_num_timesteps = False
    else:
        print("3. 새 PPO 모델을 생성합니다.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            device=args.device,
        )
        reset_num_timesteps = True

    print(f"4. 학습 시작: {args.timesteps:,} timesteps")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=args.progress_bar,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_path = os.path.join(args.models_dir, f"{args.model_prefix}_seed{args.seed}_final")
    model.save(final_path)
    print(f"5. 최종 모델 저장 완료: {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
