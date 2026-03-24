import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback # 🌟 자동 세이브 기능 추가!

from franka_env import FrankaTsidEnv

def main():
    print("🌍 1. Franka TSID 환경을 생성합니다...")
    env = FrankaTsidEnv(render_mode=None)

    print("🔍 2. 환경 표준 규격 검사 중...")
    check_env(env)
    print("✅ 환경 검사 통과! 완벽합니다.")

    os.makedirs("./models", exist_ok=True)

    print("🧠 3. PPO 인공지능 에이전트를 생성합니다...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    #model = PPO.load("./models/checkpoints/ppo_franka_delta_v3_650000_steps", env=env)

    # 🌟 [NEW 이어하기 코드] 30만 번 학습된 뇌 파일(v3)을 불러옵니다!
    #print("🧠 30만 번 학습된 뇌를 불러와서 이어서 학습합니다...")
    #model = PPO.load("./models/checkpoints/ppo_franka_delta_v3_300000_steps", env=env)
    # -------------------------------------------------------------------
    # 🌟 [NEW] 자동 세이브 (Checkpoint Callback) 설정
    # 50만 번(500,000) 스텝을 돌 때마다 그 시점의 뇌를 자동으로 백업합니다.
    # 체크포인트 이름도 변경 (선택)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, 
        save_path='./models/checkpoints/',
        name_prefix='ppo_franka_delta_v4' # 🌟 여기!
    )

    print("🚀 4. 500만 번의 뼈 깎는 학습을 시작합니다! (1분 관전 모드 켜짐)")
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback, progress_bar=True)

# 🌟 [수정된 학습 코드] reset_num_timesteps=False 가 핵심입니다!
    # 이 옵션을 넣어야 내부 스텝 카운터가 0으로 안 돌아가고 30만 번부터 숫자가 올라갑니다.
    #model.learn(
    #    total_timesteps=4_700_000,  # 남은 목표 횟수 (500만 - 30만 = 470만 번)
    #    callback=checkpoint_callback, 
    #    progress_bar=True,
    #    reset_num_timesteps=False   # 👈 핵심! "지금까지 한 스텝 수 초기화하지 마!"
    #)
    print("💾 5. 500만 번 학습 최종 완료! 천재의 뇌를 파일로 저장합니다.")
    model.save("./models/ppo_franka_reaching_delta") # 🌟 여기! (기존 파일 덮어쓰기 방지)
    print("🎉 저장 완료: ./models/ppo_franka_reaching_delta.zip")

    env.close()

if __name__ == "__main__":
    main()