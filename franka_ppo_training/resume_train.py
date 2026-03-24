from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from franka_env import FrankaTsidEnv

def main():
    print("🌍 1. 호버링(Hovering) 심화 환경 생성!")
    env = FrankaTsidEnv()

    # 🚨 여기에 60만 번 학습된 최고의 모델 파일 경로를 적어주세요!
    model_path = "./models/checkpoints/ppo_franka_delta_v3_600000_steps"
    print(f"🧠 2. 천재 로봇의 뇌를 불러옵니다... ({model_path})")
    model = PPO.load(model_path, env=env, tensorboard_log="./ppo_tensorboard/")

    # 10만 번마다 저장
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./models/checkpoints/',
        name_prefix='ppo_franka_hovering' # 이름은 hovering으로!
    )

    print("🚀 3. 호버링 심화 학습(과외) 시작! (약 50만 번 진행)")
    # 🌟 핵심: reset_num_timesteps=False 로 해야 텐서보드 그래프가 60만 번부터 예쁘게 이어집니다!
    model.learn(
        total_timesteps=500_000, 
        callback=checkpoint_callback, 
        progress_bar=True,
        reset_num_timesteps=False
    )

    print("💾 4. 심화 학습 완료! 모델을 저장합니다.")
    model.save("./models/ppo_franka_hovering_final")

if __name__ == "__main__":
    main()