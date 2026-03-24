import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer
import pinocchio as pin
from qpsolvers import solve_qp

# 질문자님의 로봇 URDF 경로 (환경에 맞게 수정하세요)
from robot_descriptions.panda_description import URDF_PATH 

class FrankaTsidEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.is_play_mode = (render_mode == "human") # 🌟 [NEW] play.py인지 기억해두기!
        self.viewer = None
        self.target_ee_pos = np.zeros(3)
        self.max_steps = 500 
        self.current_step = 0
        # --- (기존 Action/Observation Space 세팅 그대로) ---
        # 🌟 [수정됨] 절대 좌표 대신, -1.0 ~ 1.0 사이의 정규화된 상댓값(비율) 사용
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        # --- (기존 MuJoCo / Pinocchio 초기화 세팅 그대로) ---
        xml_path = os.path.expanduser("~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda/scene.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()
        self.n_pin, self.n_arm = self.pin_model.nq, 7
        
        # 게인 및 변수들 (기존과 동일)
        self.Kp_pos, self.Kd_pos = 200.0, 2.0 * np.sqrt(200.0)
        self.Kp_rot, self.Kd_rot = 200.0, 2.0 * np.sqrt(200.0)
        self.Kp_post, self.Kd_post = 100.0, 2.0 * np.sqrt(100.0)
        self.w_ee, self.w_post = 1.0, 0.009
        self.q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        self.target_body_name = "panda_hand_tcp"
        self.pin_ee_id = self.pin_model.getFrameId(self.target_body_name)
        self.q_ref = pin.Quaternion(0, 1, 0, 0) 
        
        self.sim_dt = self.mj_model.opt.timestep
        self.rl_dt = 0.1
        self.n_substeps = int(self.rl_dt / self.sim_dt)

        # ----------------------------------------------------
        # 🌟 [NEW] 시각화 및 타이머용 변수 (Mocap 방식으로 변경)
        # ----------------------------------------------------
        target_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_bottle")
        action_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ai_action")
        
        # Mocap 데이터는 별도의 인덱스를 가집니다!
        self.target_mocap_idx = self.mj_model.body_mocapid[target_body_id] if target_body_id != -1 else -1
        self.action_mocap_idx = self.mj_model.body_mocapid[action_body_id] if action_body_id != -1 else -1
        
        self.start_time = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 로봇 관절 초기화
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.n_arm] = self.q_nominal.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 이번 에피소드의 물병(최종 목표) 위치 랜덤 생성
        self.target_bottle_pos = np.array([
            np.random.uniform(0.35, 0.55),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.25, 0.45)
        ])
        curr_ee_pos = self._get_ee_pos()
        self.target_ee_pos = np.copy(curr_ee_pos)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.start_time is None:
            self.start_time = time.time()
            
        # 1. 현재 손끝 위치 가져오기
        curr_ee_pos = self._get_ee_pos()
        
        # 2. 🌟 [핵심] 상댓값(Delta) 계산
        # AI가 -1.0 ~ 1.0 사이의 값을 주면, 최대 0.07m(7cm)씩만 움직이도록 스케일링!
        max_step_size = 0.07 
        delta_pos = action * max_step_size
        
        # 다음 목표 위치 = 현재 위치 + 이동할 상댓값

        intended_pos = self.target_ee_pos + delta_pos
        desired_pos = np.copy(intended_pos)
        # 3. 안전 장치 (클리핑): 파란 공이 로봇의 한계 범위를 벗어나지 않게 가둬둡니다.
        desired_pos[0] = np.clip(desired_pos[0], 0.3, 0.6)
        desired_pos[1] = np.clip(desired_pos[1], -0.3, 0.3)
        desired_pos[2] = np.clip(desired_pos[2], 0.2, 0.6)

        # ⚡ [NEW] 전기 충격 페널티 계산!
        # AI가 의도한 위치(intended_pos)와 잘려나간 위치(desired_pos)가 다르다? -> 벽을 넘었다는 뜻!
        # 선을 넘은 거리(차이)를 계산해서 10배로 뻥튀기한 다음 감점(전기충격)을 먹입니다.
        wall_violation_dist = np.linalg.norm(intended_pos - desired_pos)
        wall_penalty = 10.0 * wall_violation_dist

        self.target_ee_pos = np.copy(desired_pos)

        # 🌟 [NEW] 시각화 공 위치 업데이트
        if self.target_mocap_idx != -1:
            self.mj_data.mocap_pos[self.target_mocap_idx] = self.target_bottle_pos
        if self.action_mocap_idx != -1:
            self.mj_data.mocap_pos[self.action_mocap_idx] = desired_pos

        # TSID + 시뮬레이션 50번 반복
        for _ in range(self.n_substeps):
            self._run_tsid_control(desired_pos)
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            if self.viewer is not None:
                self.viewer.sync()

       # 4. 상태 및 보상 계산
        obs = self._get_obs()
        new_ee_pos = self._get_ee_pos() # 이동 후의 진짜 위치
        dist_to_bottle = np.linalg.norm(self.target_bottle_pos - new_ee_pos)
        
        # ----------------------------------------------------
        # 🌟 보상 함수 계산
        # ----------------------------------------------------
        reward = -dist_to_bottle 
        reward -= 0.01 * np.linalg.norm(action) # Jerk 방지
        reward -= wall_penalty                   # 벽 충돌 전기충격
        
        # 🌟 스텝 카운트 증가
        self.current_step += 1 

        terminated = False
        truncated = False
        
        # 성공 조건 (조기 퇴근)
        if dist_to_bottle < 0.05:
            reward += 1.0  
            #terminated = True
            
        # 실패 조건 (해고)
        elif new_ee_pos[2] < 0.0 or dist_to_bottle > 1.0:
            reward -= 10.0
            terminated = True
            
        # 시간 초과 (정시 퇴근)
        elif self.current_step >= self.max_steps:
            truncated = True 

        # ----------------------------------------------------
        # 🌟 뷰어 (관전 모드) 로직
        # ----------------------------------------------------
        elapsed_time = time.time() - self.start_time
        
        if self.is_play_mode or elapsed_time < 60.0:
            if self.viewer is None:
                self._render_viewer()
            
            act_str = f"[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
            print(f"👀 남은거리: {dist_to_bottle:.2f}m | 🎯 AI명령: {act_str} | 💰 보상: {reward:.3f} | ⏰ 스텝: {self.current_step}/{self.max_steps}")
            
        else:
            if self.viewer is not None:
                print("\n⏰ 1분 관전 종료! 초고속 터보 모드로 전환합니다.\n")
                self.close()
                self.viewer = None

        return obs, reward, terminated, truncated, {}
 
    def _run_tsid_control(self, desired_pos):
        """질문자님이 짠 완벽한 TSID + QP + CBF 코드가 들어가는 곳"""
        q_arm = self.mj_data.qpos[:self.n_arm].copy()
        v_arm = self.mj_data.qvel[:self.n_arm].copy()
        
        q_pin = np.zeros(self.n_pin); q_pin[:self.n_arm] = q_arm
        v_pin = np.zeros(self.n_pin); v_pin[:self.n_arm] = v_arm

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        curr_pos = self.pin_data.oMf[self.pin_ee_id].translation
        curr_quat = pin.Quaternion(self.pin_data.oMf[self.pin_ee_id].rotation)
        
        # Jacobian 계산 등등...
        pin.computeAllTerms(self.pin_model, self.pin_data, q_pin, v_pin)
        J = pin.getFrameJacobian(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, :self.n_arm]
        
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q_pin, v_pin)
        Jdot_full = pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJV = Jdot_full[:, :self.n_arm] @ v_arm
        
        curr_v_pin = pin.getFrameVelocity(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        curr_w_pin = pin.getFrameVelocity(self.pin_model, self.pin_data, self.pin_ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).angular

        # 오차 계산 (v_ref, a_ref, w_ref 등은 이제 0입니다!)
        e_pos = desired_pos - curr_pos
        e_vel = np.zeros(3) - curr_v_pin
        e_rot_vec = pin.log3((self.q_ref * curr_quat.inverse()).matrix())
        
        # PD 가속도 지령
        a_pos = (self.Kp_pos * e_pos) + (self.Kd_pos * e_vel)
        a_rot = (self.Kp_rot * e_rot_vec) - (self.Kd_rot * curr_w_pin)
        
        b_acc = np.concatenate([a_pos, a_rot]) - dJV
        
        # Postural Task (기본 자세 유지)
        e_post = self.q_nominal - q_arm
        a_post = self.Kp_post * e_post - self.Kd_post * v_arm
        
        # QP Formulation
        P = (self.w_ee * J.T @ J) + (self.w_post * np.eye(self.n_arm)) + (1e-4 * np.eye(self.n_arm))
        q_qp = -(self.w_ee * J.T @ b_acc) - (self.w_post * a_post)
        M = self.pin_data.M[:self.n_arm, :self.n_arm]
        h = self.pin_data.nle[:self.n_arm]
        
        G_ineq = np.vstack([M, -M])
        h_ineq = np.concatenate([self.tau_max - h, self.tau_max + h])

        # 질문자님의 CBF (조인트 리밋 회피)
        q_lower = self.mj_model.jnt_range[:self.n_arm, 0]
        q_upper = self.mj_model.jnt_range[:self.n_arm, 1]
        K_lim, D_lim, margin = 50.0, 2.0 * np.sqrt(50.0), 0.05
        
        ddq_ub = np.clip(K_lim * (q_upper - margin - q_arm) - D_lim * v_arm, -50.0, 50.0)
        ddq_lb = np.clip(K_lim * (q_lower + margin - q_arm) - D_lim * v_arm, -50.0, 50.0)
        ddq_lb = np.minimum(ddq_lb, ddq_ub - 0.1)

        ddq = solve_qp(P, q_qp, G_ineq, h_ineq, lb=ddq_lb, ub=ddq_ub, solver="osqp")
        
        if ddq is not None: 
            current_tau = M @ ddq + h
            self.mj_data.ctrl[:self.n_arm] = current_tau

    def _get_ee_pos(self):
        # 1. 피노키오가 원하는 9자유도(팔 7 + 그리퍼 2) 빈 배열 생성
        q_pin = np.zeros(self.n_pin)
        
        # 2. 앞의 7개만 현재 관절 각도로 채워넣기 (그리퍼는 0으로 고정)
        q_pin[:self.n_arm] = self.mj_data.qpos[:self.n_arm].copy()
        
        # 3. 계산!
        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        return self.pin_data.oMf[self.pin_ee_id].translation.copy()
    def _get_obs(self):
        q = self.mj_data.qpos[:self.n_arm].copy()
        dq = self.mj_data.qvel[:self.n_arm].copy()
        curr_ee_pos = self._get_ee_pos()
        
        return np.concatenate([q, dq, curr_ee_pos, self.target_bottle_pos]).astype(np.float32)

    def _render_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()