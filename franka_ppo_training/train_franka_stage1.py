import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# 멀티프로세싱을 위해 추가된 모듈
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from franka_stage1_env import FrankaStage1ReachPoseEnv

def main():
    print("1. 환경 규격 검사 중...")
    # check_env는 단일 환경에서만 작동하므로 임시로 하나 만들어서 검사하고 닫습니다.
    temp_env = FrankaStage1ReachPoseEnv(
        render_mode=None,
        print_step_info=False,
        max_steps=100,
        randomize_target_pos=False,
        orientation_level="small",
    )
    check_env(temp_env)
    temp_env.close()
    print("환경 검사 통과.")

    os.makedirs("./models/stage1", exist_ok=True)
    os.makedirs("./models/stage1/checkpoints", exist_ok=True)
    os.makedirs("./ppo_tensorboard/stage1", exist_ok=True)

    print("2. Stage 1 병렬 환경(VecEnv) 생성 중...")
    
    # 노트북 환경에 맞게 사용할 스레드(환경) 개수 설정 (3 또는 4 추천)
    num_envs = 4 

    # 환경 생성 시 넘겨줄 인자들
    env_kwargs = dict(
        render_mode=None,
        print_step_info=False,
        max_steps=100,
        randomize_target_pos=False,
        orientation_level="small",
    )

    # 핵심: make_vec_env와 SubprocVecEnv를 이용해 환경을 4개 복사해서 띄웁니다.
    vec_env = make_vec_env(
        env_id=FrankaStage1ReachPoseEnv,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    print(f"3. PPO 모델 생성 중... (환경 개수: {num_envs})")

    # 기존 단일 env 대신 vec_env를 넣어줍니다.
    model = PPO(
        policy="MlpPolicy",
        env=vec_env, 
        verbose=1,
        tensorboard_log="./ppo_tensorboard/stage1/",
        n_steps=2048,
        batch_size=64,    # 환경이 4배가 되어 데이터가 늘어났으므로 256정도로 키워보셔도 좋습니다.
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # 🚨 주의: 병렬 환경에서는 각 환경이 병렬로 스텝을 진행합니다.
    # 총 스텝 5만 번마다 저장하고 싶다면, 환경 개수로 나눠줘야 합니다.
    save_freq_adjusted = max(50_000 // num_envs, 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path="./models/stage1/checkpoints/",
        name_prefix="ppo_franka_stage1_reach_pose",
    )

    print(f"4. Stage 1 학습 시작! ({num_envs}개 환경 동시 진행)")
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = "./models/stage1/ppo_franka_stage1_reach_pose_final"
    model.save(final_path)
    print(f"5. 최종 모델 저장 완료: {final_path}.zip")

    # 학습 완료 후 모든 병렬 환경을 닫아줍니다.
    vec_env.close()

if __name__ == "__main__":
    main()