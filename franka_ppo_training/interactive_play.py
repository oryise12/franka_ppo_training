import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from franka_env import FrankaTsidEnv

def get_random_target():
    """
    로봇이 닿을 수 있는 안전한 작업 공간(Workspace) 내에서 랜덤 좌표 생성
    🚨 질문자님의 실제 환경 크기에 맞춰 x, y, z 범위를 조절해주세요!
    """
    x = np.random.uniform(0.35, 0.55)   # 앞뒤
    y = np.random.uniform(-0.2, 0.2)  # 좌우
    z = np.random.uniform(0.15, 0.5)  # 위아래 (바닥에 안 박게 최소 높이 0.15)
    return np.array([x, y, z])

def main():
    parser = argparse.ArgumentParser(description="학습된 PPO 뇌를 3초 랜덤 타겟 환경에서 테스트합니다.")
    # 🌟 15만 번 학습된 최고의 모델 경로를 기본값으로 넣어주세요!
    parser.add_argument('--model', type=str, default="./models/checkpoints/ppo_franka_delta_v4_150000_steps.zip", help="불러올 모델 파일의 경로")
    args = parser.parse_args()

    print("🌍 1. 관전용 환경을 생성합니다.")
    env = FrankaTsidEnv(render_mode="human")

    print(f"🧠 2. 에이스(15만 번) 모델을 불러옵니다... 경로: {args.model}")
    model = PPO.load(args.model, env=env)

    obs, info = env.reset()
    
    # 첫 랜덤 타겟 소환!
    current_target = get_random_target()
    env.target_bottle_pos = current_target.copy()
    
    # 타이머 시작
    last_target_time = time.time()

    print("\n" + "="*60)
    print("🚀 [스트레스 테스트 모드 시작]")
    print("3초마다 목표 지점이 랜덤으로 바뀝니다! 팝콘 챙기세요 🍿")
    print("="*60 + "\n")
    
    while True:
        # ⏱️ 1. 3초가 지났는지 확인
        current_time = time.time()
        if current_time - last_target_time > 3.0:
            # 타겟 위치 랜덤 갱신
            current_target = get_random_target()
            env.target_bottle_pos = current_target.copy()
            last_target_time = current_time
            print(f"🎯 [타겟 순간이동!] 새 좌표: {current_target.round(3)}")
            
        # 2. AI 행동 예측 (수전증 방지 deterministic=True)
        action, _states = model.predict(obs, deterministic=True)
        
        # 🌟 3. 주차 브레이크 (도착하면 얌전해지게)
        # obs의 구조가 어떻게 되어있는지에 따라 인덱스는 달라질 수 있습니다. 
        # (예: env._get_ee_pos() 같은 함수가 있다면 그걸 쓰셔도 됩니다)
        curr_ee_pos = env._get_ee_pos() 
        dist = np.linalg.norm(current_target - curr_ee_pos)
        
        if dist < 0.03:
            action = action * 0.1  # 속도를 확 줄여서 안착
            
        # 4. 환경 스텝 진행
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 시간 초과(500스텝 등) 시 리셋 막고 타이머만 초기화
        if truncated:
            env.current_step = 0
            
        # 💥 치명적 사고(바닥 충돌) 시에만 초기화
        if curr_ee_pos[2] < 0.0:
            print("\n💥 [사고 발생] 바닥 충돌! 자세를 리셋합니다.")
            obs, info = env.reset()
            # 리셋 후에도 방금 생성했던 타겟 위치를 그대로 유지
            env.target_bottle_pos = current_target.copy()
            # 충돌해서 리셋됐으니, 타겟 변경 타이머도 다시 3초 리셋!
            last_target_time = time.time()

if __name__ == "__main__":
    main()