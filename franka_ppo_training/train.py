import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback 

from franka_env import FrankaTsidEnv

def main():
    print("1. Franka TSID 가변 임피던스(VIC) 환경을 생성합니다...")
    env = FrankaTsidEnv(render_mode=None)

    print("2. 환경 표준 규격 검사 중...")
    check_env(env)
    print("환경 검사 통과! 완벽합니다.")

    os.makedirs("./models", exist_ok=True)

    print("3. PPO 인공지능 에이전트를 생성합니다...")
    # 🌟 관측 공간/행동 공간이 바뀌었으므로 이전 뇌(load)는 쓰면 에러가 납니다. 새로 백지부터 시작합니다.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

    # 🌟 [수정] 자동 세이브 이름 변경 (reaching_delta -> vic_contact)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, 
        save_path='./models/checkpoints/',
        name_prefix='ppo_franka_vic_contact' # 🌟 논문 주제에 맞게 변경!
    )

    print("4. 학습 시작! (접촉 힘과 가변 강성을 학습합니다)")
    # 일단 100만 번으로 테스트 (접촉 태스크는 더 오래 걸릴 수 있습니다)
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback, progress_bar=True)

    print("5. 학습 최종 완료! 모델을 저장합니다.")
    # 🌟 [수정] 최종 저장 이름 변경
    model.save("./models/ppo_franka_vic_contact_final") 
    print("저장 완료: ./models/ppo_franka_vic_contact_final.zip")

    env.close()

if __name__ == "__main__":
    main()