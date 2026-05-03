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
    def __init__(self, render_mode=None, print_step_info=True, viewer_sync_interval=10):
        super().__init__()
        self.render_mode = render_mode
        self.is_play_mode = (render_mode == "human") # 🌟 [NEW] play.py인지 기억해두기!
        self.viewer = None
        self.print_step_info = print_step_info
        self.viewer_sync_interval = max(1, int(viewer_sync_interval))
        self.target_ee_pos = np.zeros(3)
        self.target_arrow_mocap_idx = -1
        self.action_arrow_mocap_idx = -1
        self.ee_arrow_mocap_idx = -1
        self.obstacle_pos = np.array([0.45, 0.0, 0.35], dtype=float)
        self.obstacle_size = np.array([0.018, 0.035, 0.035], dtype=float)
        self.obstacle_radius = np.linalg.norm(self.obstacle_size)
        self.obstacle_safety_margin = 0.0
        self.obstacle_safe_dist = 0.08
        self.obstacle_mocap_idx = -1
        self.obstacle_geom_id = -1
        self.contact_marker_mocap_idx = -1
        self.last_obstacle_contact_pos = np.array([0.0, 0.0, -1.0], dtype=float)
        self.prev_dist_to_bottle = 0.0
        self.progress_reward_weight = 2.0
        self.orientation_reward_weight = 0.15
        self.action_reward_weight = 0.01
        self.obstacle_near_weight = 2.0
        self.collision_penalty = 10.0
        self.success_bonus = 5.0
        self.max_steps = 500 
        self.current_step = 0
        # --- (기존 Action/Observation Space 세팅 그대로) ---
        # action = [delta position xyz, delta orientation rotvec xyz]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
        
        # --- (기존 MuJoCo / Pinocchio 초기화 세팅 그대로) ---
        menagerie_dir = os.path.expanduser("~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda")
        pose_obstacle_xml = os.path.join(menagerie_dir, "scene_pose_obstacle.xml")
        fallback_xml = os.path.join(menagerie_dir, "scene.xml")
        xml_path = pose_obstacle_xml if os.path.exists(pose_obstacle_xml) else fallback_xml
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()
        self.n_pin, self.n_arm = self.pin_model.nq, 7
        
        # 게인 및 변수들 (기존과 동일)
        self.Kp_pos, self.Kd_pos = 260.0, 2.0 * np.sqrt(260.0)
        self.Kp_rot, self.Kd_rot = 260.0, 2.0 * np.sqrt(260.0)
        self.Kp_post, self.Kd_post = 130.0, 2.0 * np.sqrt(130.0)
        self.Ki_pos, self.Ki_rot = 120.0, 30.0
        self.integral_error_pos = np.zeros(3)
        self.integral_error_rot = np.zeros(3)
        self.w_ee, self.w_post = 1.0, 0.009
        self.q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        self.target_body_name = "panda_hand_tcp"
        self.pin_ee_id = self.pin_model.getFrameId(self.target_body_name)
        self.q_ref = pin.Quaternion(0, 1, 0, 0) 
        self.target_ee_quat = pin.Quaternion(self.q_ref.matrix())
        self.subgoal_ee_quat = pin.Quaternion(self.q_ref.matrix())
        
        self.sim_dt = self.mj_model.opt.timestep
        self.rl_dt = 0.1
        self.n_substeps = int(self.rl_dt / self.sim_dt)

        # ----------------------------------------------------
        # 🌟 [NEW] 시각화 및 타이머용 변수 (Mocap 방식으로 변경)
        # ----------------------------------------------------
        target_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_bottle")
        action_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ai_action")
        target_arrow_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_arrow")
        action_arrow_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ai_action_arrow")
        ee_arrow_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ee_arrow")
        obstacle_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        contact_marker_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "contact_marker")
        self.obstacle_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
        
        # Mocap 데이터는 별도의 인덱스를 가집니다!
        self.target_mocap_idx = self.mj_model.body_mocapid[target_body_id] if target_body_id != -1 else -1
        self.action_mocap_idx = self.mj_model.body_mocapid[action_body_id] if action_body_id != -1 else -1
        self.target_arrow_mocap_idx = self.mj_model.body_mocapid[target_arrow_body_id] if target_arrow_body_id != -1 else -1
        self.action_arrow_mocap_idx = self.mj_model.body_mocapid[action_arrow_body_id] if action_arrow_body_id != -1 else -1
        self.ee_arrow_mocap_idx = self.mj_model.body_mocapid[ee_arrow_body_id] if ee_arrow_body_id != -1 else -1
        self.obstacle_mocap_idx = self.mj_model.body_mocapid[obstacle_body_id] if obstacle_body_id != -1 else -1
        self.contact_marker_mocap_idx = self.mj_model.body_mocapid[contact_marker_body_id] if contact_marker_body_id != -1 else -1
        if self.obstacle_geom_id != -1:
            self.mj_model.geom_margin[self.obstacle_geom_id] = self.obstacle_safety_margin
        
        self.start_time = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 로봇 관절 초기화
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:self.n_arm] = self.q_nominal.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 이번 에피소드의 목표 pose 생성
        self.target_bottle_pos = np.array([
            np.random.uniform(0.35, 0.55),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.25, 0.45)
        ])
        self.target_ee_quat = self._sample_target_quat()

        curr_ee_pos = self._get_ee_pos()
        self.target_ee_pos = np.copy(curr_ee_pos)
        self.subgoal_ee_quat = self._get_ee_quat()
        self.integral_error_pos = np.zeros(3)
        self.integral_error_rot = np.zeros(3)
        self._sample_obstacle_on_path(curr_ee_pos, self.target_bottle_pos)
        self._update_target_visuals()
        self._update_action_visuals(self.target_ee_pos, self.subgoal_ee_quat)
        self._update_ee_visuals(curr_ee_pos, self.subgoal_ee_quat)
        self._update_obstacle_geom()
        self.prev_dist_to_bottle = np.linalg.norm(self.target_bottle_pos - curr_ee_pos)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.start_time is None:
            self.start_time = time.time()
            
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        # 1. 현재 손끝 pose 가져오기
        curr_ee_pos = self._get_ee_pos()
        
        # 2. PPO가 작은 pose sub-goal delta를 생성합니다.
        max_step_size = 0.07 
        max_rot_step = 0.20
        delta_pos = action[:3] * max_step_size
        delta_rot = action[3:] * max_rot_step
        
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
        self.subgoal_ee_quat = self._integrate_quat(self.subgoal_ee_quat, delta_rot)

        # 🌟 [NEW] 시각화 공 위치 업데이트
        if self.target_mocap_idx != -1:
            self.mj_data.mocap_pos[self.target_mocap_idx] = self.target_bottle_pos
        self._update_target_visuals()
        if self.action_mocap_idx != -1:
            self.mj_data.mocap_pos[self.action_mocap_idx] = desired_pos
        self._update_action_visuals(desired_pos, self.subgoal_ee_quat)
        self._update_obstacle_geom()

        # TSID + 시뮬레이션 50번 반복
        for substep_idx in range(self.n_substeps):
            self._run_tsid_control(desired_pos, self.subgoal_ee_quat)
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            if (
                self.viewer is not None
                and (substep_idx + 1) % self.viewer_sync_interval == 0
            ):
                self.viewer.sync()

       # 4. 상태 및 보상 계산
        obs = self._get_obs()
        new_ee_pos = self._get_ee_pos() # 이동 후의 진짜 위치
        new_ee_quat = self._get_ee_quat()
        self._update_ee_visuals(new_ee_pos, new_ee_quat)
        dist_to_bottle = np.linalg.norm(self.target_bottle_pos - new_ee_pos)
        orientation_error = np.linalg.norm(self._orientation_error(self.target_ee_quat, new_ee_quat))
        obstacle_collision, obstacle_contact_count, obstacle_clearance, obstacle_contact_pos = self._get_obstacle_contact_info()
        self._update_contact_marker(obstacle_contact_pos)
        obstacle_penalty = 1.0 if obstacle_collision else 0.0
        progress_reward = self.prev_dist_to_bottle - dist_to_bottle
        obstacle_near_penalty = max(0.0, self.obstacle_safe_dist - obstacle_clearance)
        
        # ----------------------------------------------------
        # 🌟 보상 함수 계산
        # ----------------------------------------------------
        reward = -dist_to_bottle 
        reward -= self.orientation_reward_weight * orientation_error
        reward += self.progress_reward_weight * progress_reward
        reward -= self.action_reward_weight * np.linalg.norm(action) # Jerk 방지
        reward -= wall_penalty                   # 벽 충돌 전기충격
        reward -= self.obstacle_near_weight * obstacle_near_penalty
        reward -= 0.8 * obstacle_penalty
        self.prev_dist_to_bottle = dist_to_bottle
        
        # 🌟 스텝 카운트 증가
        self.current_step += 1 

        terminated = False
        truncated = False
        
        # 실패 조건 (해고)
        if new_ee_pos[2] < 0.0 or dist_to_bottle > 1.0 or obstacle_collision:
            reward -= self.collision_penalty
            terminated = True

        # 성공 조건 (조기 퇴근)
        elif dist_to_bottle < 0.05 and orientation_error < 0.20:
            reward += self.success_bonus
            terminated = True
            
        # 시간 초과 (정시 퇴근)
        elif self.current_step >= self.max_steps:
            truncated = True 

        # ----------------------------------------------------
        # 🌟 뷰어 (관전 모드) 로직
        # ----------------------------------------------------
        elapsed_time = time.time() - self.start_time
        
        if self.is_play_mode:
            if self.viewer is None:
                self._render_viewer()
            
            if self.print_step_info:
                act_str = f"pos[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}] rot[{action[3]:.2f}, {action[4]:.2f}, {action[5]:.2f}]"
                print(f"👀 거리: {dist_to_bottle:.2f}m | 회전오차: {orientation_error:.2f}rad | 장애물여유: {obstacle_clearance:.2f}m | 🎯 AI명령: {act_str} | 💰 보상: {reward:.3f} | ⏰ 스텝: {self.current_step}/{self.max_steps}")
            
        else:
            if self.viewer is not None:
                print("\n⏰ 1분 관전 종료! 초고속 터보 모드로 전환합니다.\n")
                self.close()
                self.viewer = None

        # --------- 여기 수정: 논문용 PPO 평가 그래프를 위한 info 반환 ----------
        info = {
            "ee_pos": new_ee_pos.copy(),                      # 실제 EE 위치
            "subgoal_pos": desired_pos.copy(),                # PPO action으로 갱신된 sub-goal 위치
            "target_pos": self.target_bottle_pos.copy(),      # 최종 목표 위치
            "dist_to_goal": dist_to_bottle,                   # ||target - EE||
            "orientation_error": orientation_error,
            "target_quat_wxyz": self._quat_wxyz(self.target_ee_quat),
            "obstacle_pos": self.obstacle_pos.copy(),
            "obstacle_radius": self.obstacle_radius,
            "obstacle_size": self.obstacle_size.copy(),
            "obstacle_clearance": obstacle_clearance,
            "obstacle_near_penalty": obstacle_near_penalty,
            "obstacle_collision": obstacle_collision,
            "obstacle_contact_count": obstacle_contact_count,
            "obstacle_contact_pos": obstacle_contact_pos.copy(),
            "progress_reward": progress_reward,
            "subgoal_tracking_error": np.linalg.norm(desired_pos - new_ee_pos),
            "goal_tracking_error": np.linalg.norm(self.target_bottle_pos - new_ee_pos),
            "wall_violation_dist": wall_violation_dist,
            "reward": reward,
            "action": np.array(action).copy(),
            "action_norm": np.linalg.norm(action),
            "sim_time": self.mj_data.time,
        }
        return obs, reward, terminated, truncated, info
# ----------------------------------------------------------------------
 
    def _run_tsid_control(self, desired_pos, desired_quat):
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
        e_rot_vec = self._orientation_error(desired_quat, curr_quat)

        self.integral_error_pos += e_pos * self.sim_dt
        self.integral_error_rot += e_rot_vec * self.sim_dt
        self.integral_error_pos = np.clip(self.integral_error_pos, -0.25, 0.25)
        self.integral_error_rot = np.clip(self.integral_error_rot, -0.25, 0.25)
        
        # PD 가속도 지령
        a_pos = (
            (self.Kp_pos * e_pos)
            + (self.Kd_pos * e_vel)
            + (self.Ki_pos * self.integral_error_pos)
        )
        a_rot = (
            (self.Kp_rot * e_rot_vec)
            - (self.Kd_rot * curr_w_pin)
            + (self.Ki_rot * self.integral_error_rot)
        )
        
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

    def _get_ee_quat(self):
        q_pin = np.zeros(self.n_pin)
        q_pin[:self.n_arm] = self.mj_data.qpos[:self.n_arm].copy()

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        quat = pin.Quaternion(self.pin_data.oMf[self.pin_ee_id].rotation)
        quat.normalize()
        return quat

    def _orientation_error(self, target_quat, current_quat):
        return pin.log3((target_quat * current_quat.inverse()).matrix())

    def _quat_wxyz(self, quat):
        coeffs = np.asarray(quat.coeffs()).reshape(-1)
        return np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]])

    def _integrate_quat(self, quat, delta_rotvec):
        if np.linalg.norm(delta_rotvec) < 1e-9:
            new_quat = pin.Quaternion(quat.matrix())
        else:
            new_quat = pin.Quaternion(pin.exp3(delta_rotvec) @ quat.matrix())
        new_quat.normalize()
        return new_quat

    def _sample_target_quat(self):
        # Fully random SO(3)는 초기 학습에 너무 어렵기 때문에 reachable한 RPY 범위에서 샘플링합니다.
        roll = np.deg2rad(np.random.uniform(-20.0, 20.0))
        pitch = np.deg2rad(np.random.uniform(-20.0, 20.0))
        yaw = np.deg2rad(np.random.uniform(-120.0, 120.0))
        target_rot = self._rpy_to_rot(roll, pitch, yaw) @ self.q_ref.matrix()
        quat = pin.Quaternion(target_rot)
        quat.normalize()
        return quat

    def _rpy_to_rot(self, roll, pitch, yaw):
        r_roll = pin.exp3(np.array([roll, 0.0, 0.0]))
        r_pitch = pin.exp3(np.array([0.0, pitch, 0.0]))
        r_yaw = pin.exp3(np.array([0.0, 0.0, yaw]))
        return r_yaw @ r_pitch @ r_roll

    def _sample_obstacle_on_path(self, start_pos, goal_pos):
        path_vec = goal_pos - start_pos
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6:
            path_dir = np.array([1.0, 0.0, 0.0])
        else:
            path_dir = path_vec / path_len

        alpha = np.random.uniform(0.38, 0.55)
        center = start_pos + alpha * path_vec

        rand_vec = np.random.normal(size=3)
        perp = rand_vec - np.dot(rand_vec, path_dir) * path_dir
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-6:
            perp = np.array([0.0, 1.0, 0.0])
        else:
            perp = perp / perp_norm

        center = center + perp * np.random.uniform(-0.015, 0.015)
        center[0] = np.clip(center[0], 0.3, 0.6)
        center[1] = np.clip(center[1], -0.3, 0.3)
        center[2] = np.clip(center[2], 0.2, 0.6)

        self.obstacle_pos = center
        scale = np.random.uniform(0.85, 1.15)
        self.obstacle_size = np.array([0.018, 0.035, 0.035], dtype=float) * scale
        self.obstacle_radius = np.linalg.norm(self.obstacle_size)

    def _update_target_visuals(self):
        if self.target_mocap_idx != -1:
            self.mj_data.mocap_pos[self.target_mocap_idx] = self.target_bottle_pos
        if self.target_arrow_mocap_idx != -1:
            self.mj_data.mocap_pos[self.target_arrow_mocap_idx] = self.target_bottle_pos
            self.mj_data.mocap_quat[self.target_arrow_mocap_idx] = self._quat_wxyz(self.target_ee_quat)

    def _update_action_visuals(self, subgoal_pos, subgoal_quat):
        if self.action_mocap_idx != -1:
            self.mj_data.mocap_pos[self.action_mocap_idx] = subgoal_pos
        if self.action_arrow_mocap_idx != -1:
            self.mj_data.mocap_pos[self.action_arrow_mocap_idx] = subgoal_pos
            self.mj_data.mocap_quat[self.action_arrow_mocap_idx] = self._quat_wxyz(subgoal_quat)

    def _update_ee_visuals(self, ee_pos, ee_quat):
        if self.ee_arrow_mocap_idx != -1:
            self.mj_data.mocap_pos[self.ee_arrow_mocap_idx] = ee_pos
            self.mj_data.mocap_quat[self.ee_arrow_mocap_idx] = self._quat_wxyz(ee_quat)

    def _update_obstacle_geom(self):
        if self.obstacle_mocap_idx != -1:
            self.mj_data.mocap_pos[self.obstacle_mocap_idx] = self.obstacle_pos
        if self.obstacle_geom_id != -1:
            self.mj_model.geom_size[self.obstacle_geom_id, :3] = self.obstacle_size
            self.mj_model.geom_margin[self.obstacle_geom_id] = self.obstacle_safety_margin
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _update_contact_marker(self, contact_pos):
        if self.contact_marker_mocap_idx == -1:
            return
        self.mj_data.mocap_pos[self.contact_marker_mocap_idx] = contact_pos

    def _get_obstacle_contact_info(self):
        if self.obstacle_geom_id == -1:
            ee_clearance = np.linalg.norm(self._get_ee_pos() - self.obstacle_pos) - self.obstacle_radius
            contact_pos = self._get_ee_pos() if ee_clearance < 0.0 else np.array([0.0, 0.0, -1.0])
            return ee_clearance < 0.0, int(ee_clearance < 0.0), ee_clearance, contact_pos

        collision = False
        contact_count = 0
        min_clearance = np.inf
        contact_pos = np.array([0.0, 0.0, -1.0], dtype=float)

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 != self.obstacle_geom_id and geom2 != self.obstacle_geom_id:
                continue

            other_geom = geom2 if geom1 == self.obstacle_geom_id else geom1
            other_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            if other_name == "floor":
                continue

            min_clearance = min(min_clearance, contact.dist)
            if contact.dist <= min_clearance:
                contact_pos = contact.pos.copy()
            if contact.dist <= 0.0:
                collision = True
                contact_count += 1

        if np.isinf(min_clearance):
            min_clearance = self._approx_ee_box_clearance()

        return collision, contact_count, min_clearance, contact_pos

    def _approx_ee_box_clearance(self):
        ee_pos = self._get_ee_pos()
        delta = np.abs(ee_pos - self.obstacle_pos) - self.obstacle_size
        outside_dist = np.linalg.norm(np.maximum(delta, 0.0))
        inside_dist = np.min(self.obstacle_size - np.abs(ee_pos - self.obstacle_pos))
        return outside_dist if np.any(delta > 0.0) else -inside_dist

    def _get_obs(self):
        q = self.mj_data.qpos[:self.n_arm].copy()
        dq = self.mj_data.qvel[:self.n_arm].copy()
        curr_ee_pos = self._get_ee_pos()
        curr_ee_quat = self._get_ee_quat()
        pos_error = self.target_bottle_pos - curr_ee_pos
        ori_error = self._orientation_error(self.target_ee_quat, curr_ee_quat)
        obstacle_rel = self.obstacle_pos - curr_ee_pos
        
        return np.concatenate([
            q,
            dq,
            curr_ee_pos,
            pos_error,
            ori_error,
            obstacle_rel,
            np.array([self.obstacle_radius]),
        ]).astype(np.float32)

    def _render_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
