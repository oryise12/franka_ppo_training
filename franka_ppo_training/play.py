import time
import argparse  # 🌟 [NEW] 터미널 명령어를 읽어오는 라이브러리 추가
from stable_baselines3 import PPO
from franka_env import FrankaTsidEnv

def main():
    # 🌟 [NEW] 터미널에서 --model 옵션으로 경로를 받을 수 있게 세팅합니다.
    parser = argparse.ArgumentParser(description="학습된 PPO 뇌를 테스트합니다.")
    # 기본값은 최종 완료 파일로 두되, 명령어로 덮어쓸 수 있습니다. (확장자 .zip은 빼도 됩니다)
    parser.add_argument('--model', type=str, default="./models/checkpoints/ppo_franka_delta_v4_150000_steps", help="./model/ppo_franka_delta_v4_150000_steps")
    args = parser.parse_args()

    print("🌍 1. 관전용 환경을 생성합니다. (화면 켜기!)")
    env = FrankaTsidEnv(render_mode="human")

    print(f"🧠 2. 저장된 뇌를 불러옵니다... 경로: {args.model}")
    # 🌟 [NEW] 하드코딩된 경로 대신, 입력받은 args.model을 사용합니다.
    model = PPO.load(args.model, env=env)

    print("🍿 3. AI의 플레이를 감상합니다! (종료하려면 터미널에서 Ctrl+C)")
    
    while True:
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # time.sleep(0.01) 

if __name__ == "__main__":
    main()