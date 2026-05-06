# franka_stage2_env.py

import os
import time
from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer
import pinocchio as pin
from qpsolvers import solve_qp

from robot_descriptions.panda_description import URDF_PATH


class FrankaStage2PrecisionEnv(gym.Env):
    """
    Stage 2 PPO + TSID environment.

    목적:
    - Stage1 정책을 이어받아 fine-tuning
    - 목표 근처 위치/방위 정확도 향상
    - 목표 근처 떨림 감소
    - PPO subgoal 변화량 완화

    Stage1 정책 load를 위해 action/observation dimension은 유지합니다.

    Action: shape=(6,)
        action[0:3] = delta position
        action[3:6] = delta rotation vector

    Observation: shape=(29,)
        q(7)
        dq(7)
        ee_pos(3)
        pos_error(3)
        ori_error(3)
        prev_action(6)
    """

    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        print_step_info: bool = False,
        viewer_sync_interval: int = 10,
        max_steps: int = 120,
        fixed_target_pos: Optional[np.ndarray] = None,
        randomize_target_pos: bool = False,
        orientation_level: str = "small",
        use_adaptive_action_scale: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.is_play_mode = render_mode == "human"
        self.viewer = None
        self.print_step_info = print_step_info
        self.viewer_sync_interval = max(1, int(viewer_sync_interval))

        self.max_steps = int(max_steps)
        self.current_step = 0
        self.start_time = None

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32,
        )

        # -------------------------
        # Stage2 action scale
        # -------------------------
        self.use_adaptive_action_scale = use_adaptive_action_scale

        self.base_max_step_size = 0.030
        self.base_max_rot_step = 0.080

        self.mid_max_step_size = 0.020
        self.mid_max_rot_step = 0.060

        self.near_max_step_size = 0.010
        self.near_max_rot_step = 0.035

        # -------------------------
        # Stage2 Reward weights: simple version
        # -------------------------



        # 목표 추종
        self.w_pos = 2.0
        self.w_ori = 0.8
        self.w_progress = 0.5

        # subgoal 부드러움
        self.w_subgoal_jump_pos = 2.0
        self.w_subgoal_jump_rot = 0.5

        # 목표 근처 떨림 억제
        self.w_near_goal_vel = 0.5
        self.near_goal_threshold = 0.05

        self.near_dist_threshold = 0.10
        self.very_near_dist_threshold = 0.05

        # workspace
        self.w_workspace = 5.0

        # 버튼 충돌 금지
        self.button_collision_penalty = 10.0

        # 시간 관련
        self.fast_reach_step_limit = 17      # 1.7초 / rl_dt 0.1초
        self.fast_reach_bonus = 2.0
        self.first_reach_step = None
        self.fast_bonus_given = False

        # 목표 유지
        self.hold_bonus_per_step = 0.05
        self.required_hold_steps = 30        # 3초 / rl_dt 0.1초
        self.hold_counter = 0

        # 성공/실패
        self.success_bonus = 8.0
        self.failure_penalty = 5.0

        # Stage2 성공 기준
        self.success_dist = 0.05             # 5 cm
        self.success_ori = 0.20              # 0.2 rad


        # Workspace clipping
        self.workspace_low = np.array([0.30, -0.30, 0.20], dtype=float)
        self.workspace_high = np.array([0.65, 0.30, 0.65], dtype=float)

        # Target
        self.fixed_target_pos = (
            np.array([0.50, 0.00, 0.44], dtype=float)
            if fixed_target_pos is None
            else np.asarray(fixed_target_pos, dtype=float)
        )
        self.randomize_target_pos = bool(randomize_target_pos)
        self.orientation_level = orientation_level

        self.target_pos = self.fixed_target_pos.copy()
        self.target_quat = None

        # -------------------------
        # MuJoCo / Pinocchio
        # -------------------------
        menagerie_dir = os.path.expanduser(
            "~/ros2_ws_py/src/mujoco_menagerie/franka_emika_panda"
        )
        xml_path = os.path.join(menagerie_dir, "scene.xml")

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()

        self.n_pin = self.pin_model.nq
        self.n_arm = 7

        # -------------------------
        # TSID gains
        # -------------------------
        self.Kp_pos = 260.0
        self.Kd_pos = 2.0 * np.sqrt(self.Kp_pos)
        self.Ki_pos = 30.0

        self.Kp_rot = 260.0
        self.Kd_rot = 2.0 * np.sqrt(self.Kp_rot)
        self.Ki_rot = 15.0

        self.Kp_post = 130.0
        self.Kd_post = 2.0 * np.sqrt(self.Kp_post)

        self.integral_error_pos = np.zeros(3, dtype=float)
        self.integral_error_rot = np.zeros(3, dtype=float)

        self.integral_clip_pos = 0.20
        self.integral_clip_rot = 0.20

        self.w_ee = 1.0
        self.w_post = 0.009

        self.q_nominal = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            dtype=float,
        )

        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

        self.target_body_name = "panda_hand_tcp"
        self.pin_ee_id = self.pin_model.getFrameId(self.target_body_name)

        self.q_ref = pin.Quaternion(0, 1, 0, 0)

        self.sim_dt = self.mj_model.opt.timestep
        self.rl_dt = 0.1
        self.n_substeps = max(1, int(round(self.rl_dt / self.sim_dt)))

        # Mocap visualization
        self.target_mocap_idx = self._get_mocap_idx("target_bottle")
        self.action_mocap_idx = self._get_mocap_idx("ai_action")
        self.target_arrow_mocap_idx = self._get_mocap_idx("target_arrow")
        self.action_arrow_mocap_idx = self._get_mocap_idx("ai_action_arrow")
        self.ee_arrow_mocap_idx = self._get_mocap_idx("ee_arrow")

        #geom id
        self.button_geom_id = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "button_geom",
        )
        self.button_rgba_default = np.array([1.0, 0.4, 0.1, 1.0])
        self.button_rgba_contact = np.array([1.0, 0.0, 0.0, 1.0])
        # State memory
        self.prev_dist = 0.0
        self.prev_ori_err = 0.0
        self.prev_action = np.zeros(6, dtype=float)

        self.prev_desired_pos = None
        self.prev_desired_quat = None

        self.last_reward_terms = {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[: self.n_arm] = self.q_nominal.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.hold_counter = 0
        self.first_reach_step = None
        self.fast_bonus_given = False

        if self.button_geom_id != -1:
            self.mj_model.geom_rgba[self.button_geom_id] = self.button_rgba_default
        
        self.current_step = 0
        self.start_time = time.time()

        self.prev_action[:] = 0.0

        self.integral_error_pos[:] = 0.0
        self.integral_error_rot[:] = 0.0

        self.target_pos = self._sample_target_pos()
        self.target_quat = self._sample_target_quat()

        ee_pos = self._get_ee_pos()
        ee_quat = self._get_ee_quat()

        self.prev_dist = np.linalg.norm(self.target_pos - ee_pos)
        self.prev_ori_err = np.linalg.norm(
            self._orientation_error(self.target_quat, ee_quat)
        )

        self.prev_desired_pos = ee_pos.copy()
        self.prev_desired_quat = pin.Quaternion(ee_quat.matrix())

        self._update_target_visuals()
        self._update_action_visuals(ee_pos, ee_quat)
        self._update_ee_visuals(ee_pos, ee_quat)

        self.last_reward_terms = {}

        return self._get_obs(), {}
    
    def _check_button_collision(self):
        if self.button_geom_id == -1:
            return False

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]

            if contact.geom1 == self.button_geom_id or contact.geom2 == self.button_geom_id:
                return True

        return False
    
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        curr_pos = self._get_ee_pos()
        curr_quat = self._get_ee_quat()
        curr_v, curr_w = self._get_ee_velocity()

        dist_before_action = np.linalg.norm(self.target_pos - curr_pos)

        max_step_size, max_rot_step = self._get_action_scale(dist_before_action)

        delta_pos = action[:3] * max_step_size
        delta_rot = action[3:6] * max_rot_step

        intended_pos = curr_pos + delta_pos
        desired_pos = np.clip(intended_pos, self.workspace_low, self.workspace_high)
        workspace_violation = np.linalg.norm(intended_pos - desired_pos)

        desired_quat = self._integrate_quat(curr_quat, delta_rot)

        if self.prev_desired_pos is None:
            subgoal_jump_pos = 0.0
        else:
            subgoal_jump_pos = np.linalg.norm(desired_pos - self.prev_desired_pos)

        if self.prev_desired_quat is None:
            subgoal_jump_rot = 0.0
        else:
            subgoal_jump_rot = np.linalg.norm(
                self._orientation_error(desired_quat, self.prev_desired_quat)
            )

        self._update_action_visuals(desired_pos, desired_quat)

        for substep_idx in range(self.n_substeps):
            self._run_tsid_control(desired_pos, desired_quat)
            mujoco.mj_step(self.mj_model, self.mj_data)

            if self.is_play_mode and self.viewer is not None:
                if (substep_idx + 1) % self.viewer_sync_interval == 0:
                    self.viewer.sync()

        new_pos = self._get_ee_pos()
        new_quat = self._get_ee_quat()
        new_v, new_w = self._get_ee_velocity()

        self._update_ee_visuals(new_pos, new_quat)

        dist = np.linalg.norm(self.target_pos - new_pos)
        ori_err = np.linalg.norm(self._orientation_error(self.target_quat, new_quat))

        progress = self.prev_dist - dist

        linear_vel_norm = np.linalg.norm(new_v)
        angular_vel_norm = np.linalg.norm(new_w)

        near_goal = dist < self.near_goal_threshold
        near_goal_vel_penalty = linear_vel_norm**2 + 0.2 * angular_vel_norm**2 if near_goal else 0.0

        button_collision = self._check_button_collision()

        if self.button_geom_id != -1:
            if button_collision:
                self.mj_model.geom_rgba[self.button_geom_id] = self.button_rgba_contact
            else:
                self.mj_model.geom_rgba[self.button_geom_id] = self.button_rgba_default

        reach_condition = dist < self.success_dist and ori_err < self.success_ori

        if reach_condition:
            self.hold_counter += 1

            if self.first_reach_step is None:
                self.first_reach_step = self.current_step
        else:
            self.hold_counter = 0

        fast_reach_bonus = 0.0
        if (
            self.first_reach_step is not None
            and self.first_reach_step <= self.fast_reach_step_limit
            and not self.fast_bonus_given
        ):
            fast_reach_bonus = self.fast_reach_bonus
            self.fast_bonus_given = True

        hold_bonus = self.hold_bonus_per_step if reach_condition else 0.0

        reward_terms = {
            # penalties
            "penalty_position_error": -self.w_pos * dist,
            "penalty_orientation_error": -self.w_ori * ori_err,
            "penalty_subgoal_jump_pos": -self.w_subgoal_jump_pos * subgoal_jump_pos**2,
            "penalty_subgoal_jump_rot": -self.w_subgoal_jump_rot * subgoal_jump_rot**2,
            "penalty_near_goal_velocity": -self.w_near_goal_vel * near_goal_vel_penalty,
            "penalty_workspace": -self.w_workspace * workspace_violation,
            "penalty_button_collision": -self.button_collision_penalty if button_collision else 0.0,

            # bonuses
            "bonus_progress": self.w_progress * progress,
            "bonus_hold": hold_bonus,
            "bonus_fast_reach": fast_reach_bonus,

            # terminal terms
            "bonus_success": 0.0,
            "penalty_failure": 0.0,
        }

        reward = float(sum(reward_terms.values()))

        

        self.current_step += 1

        terminated = False
        truncated = False
        success = False
        failure = False


        if not np.isfinite(new_pos).all() or not np.isfinite(ori_err):
            reward -= self.failure_penalty
            reward_terms["penalty_failure"] = -self.failure_penalty
            terminated = True
            failure = True

        elif new_pos[2] < 0.05 or dist > 1.20:
            reward -= self.failure_penalty
            reward_terms["penalty_failure"] = -self.failure_penalty
            terminated = True
            failure = True

        elif self.hold_counter >= self.required_hold_steps:
            reward += self.success_bonus
            reward_terms["bonus_success"] = self.success_bonus
            terminated = True
            success = True

        elif self.current_step >= self.max_steps:
            truncated = True


        self.last_reward_terms = reward_terms.copy()

        obs = self._get_obs()

        info = {
            "ee_pos": new_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "target_quat_wxyz": self._quat_wxyz(self.target_quat),
            "desired_pos": desired_pos.copy(),
            "desired_quat_wxyz": self._quat_wxyz(desired_quat),

            "dist_to_goal": float(dist),
            "orientation_error": float(ori_err),
            "progress": float(progress),

            "linear_vel_norm": float(linear_vel_norm),
            "angular_vel_norm": float(angular_vel_norm),

            "workspace_violation": float(workspace_violation),
            "action_norm": float(np.linalg.norm(action)),

            "subgoal_jump_pos": float(subgoal_jump_pos),
            "subgoal_jump_rot": float(subgoal_jump_rot),

            "max_step_size": float(max_step_size),
            "max_rot_step": float(max_rot_step),

            "success": bool(success),
            "failure": bool(failure),

            "reward_terms": reward_terms.copy(),
            "reward_total": float(reward),
            "sim_time": float(self.mj_data.time),

            "button_collision": bool(button_collision),
            "hold_counter": int(self.hold_counter),
            "first_reach_step": -1 if self.first_reach_step is None else int(self.first_reach_step),
            "reach_condition": bool(reach_condition),
            "near_goal": bool(near_goal),
        }

        if self.is_play_mode:
            if self.viewer is None:
                self._render_viewer()

            if self.print_step_info:
                print(
                    f"dist={dist:.4f} m | "
                    f"ori={ori_err:.4f} rad | "
                    f"v={linear_vel_norm:.4f} m/s | "
                    f"w={angular_vel_norm:.4f} rad/s | "
                    f"reward={reward:.3f} | "
                    f"hold_cnt={self.hold_counter}/{self.required_hold_steps} | "
                    f"step={self.current_step}/{self.max_steps}"
                )

        self.prev_dist = dist
        self.prev_ori_err = ori_err
        self.prev_action = action.copy()
        self.prev_desired_pos = desired_pos.copy()
        self.prev_desired_quat = pin.Quaternion(desired_quat.matrix())

        return obs, reward, terminated, truncated, info

    def _run_tsid_control(self, desired_pos: np.ndarray, desired_quat: pin.Quaternion):
        q_arm = self.mj_data.qpos[: self.n_arm].copy()
        v_arm = self.mj_data.qvel[: self.n_arm].copy()

        q_pin = np.zeros(self.n_pin, dtype=float)
        v_pin = np.zeros(self.n_pin, dtype=float)

        q_pin[: self.n_arm] = q_arm
        v_pin[: self.n_arm] = v_arm

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.computeAllTerms(self.pin_model, self.pin_data, q_pin, v_pin)

        curr_pos = self.pin_data.oMf[self.pin_ee_id].translation.copy()
        curr_quat = pin.Quaternion(self.pin_data.oMf[self.pin_ee_id].rotation)
        curr_quat.normalize()

        J = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.pin_ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:, : self.n_arm]

        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, q_pin, v_pin)

        Jdot_full = pin.getFrameJacobianTimeVariation(
            self.pin_model,
            self.pin_data,
            self.pin_ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        dJv = Jdot_full[:, : self.n_arm] @ v_arm

        frame_vel = pin.getFrameVelocity(
            self.pin_model,
            self.pin_data,
            self.pin_ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        curr_v = frame_vel.linear
        curr_w = frame_vel.angular

        e_pos = desired_pos - curr_pos
        e_vel = -curr_v
        e_rot = self._orientation_error(desired_quat, curr_quat)

        self.integral_error_pos += e_pos * self.sim_dt
        self.integral_error_rot += e_rot * self.sim_dt

        self.integral_error_pos = np.clip(
            self.integral_error_pos,
            -self.integral_clip_pos,
            self.integral_clip_pos,
        )

        self.integral_error_rot = np.clip(
            self.integral_error_rot,
            -self.integral_clip_rot,
            self.integral_clip_rot,
        )

        a_pos = (
            self.Kp_pos * e_pos
            + self.Kd_pos * e_vel
            + self.Ki_pos * self.integral_error_pos
        )

        a_rot = (
            self.Kp_rot * e_rot
            - self.Kd_rot * curr_w
            + self.Ki_rot * self.integral_error_rot
        )

        b_acc = np.concatenate([a_pos, a_rot]) - dJv

        e_post = self.q_nominal - q_arm
        a_post = self.Kp_post * e_post - self.Kd_post * v_arm

        P = (
            self.w_ee * (J.T @ J)
            + self.w_post * np.eye(self.n_arm)
            + 1e-4 * np.eye(self.n_arm)
        )

        P = 0.5 * (P + P.T)

        q_qp = -(self.w_ee * J.T @ b_acc) - self.w_post * a_post

        M = self.pin_data.M[: self.n_arm, : self.n_arm]
        h = self.pin_data.nle[: self.n_arm]

        G = np.vstack([M, -M])
        h_ineq = np.concatenate([self.tau_max - h, self.tau_max + h])

        q_lower = self.mj_model.jnt_range[: self.n_arm, 0]
        q_upper = self.mj_model.jnt_range[: self.n_arm, 1]

        K_lim = 50.0
        D_lim = 2.0 * np.sqrt(K_lim)
        margin = 0.05

        ddq_ub = K_lim * (q_upper - margin - q_arm) - D_lim * v_arm
        ddq_lb = K_lim * (q_lower + margin - q_arm) - D_lim * v_arm

        ddq_ub = np.clip(ddq_ub, -50.0, 50.0)
        ddq_lb = np.clip(ddq_lb, -50.0, 50.0)

        ddq_lb = np.minimum(ddq_lb, ddq_ub - 0.1)

        try:
            ddq = solve_qp(
                P,
                q_qp,
                G,
                h_ineq,
                lb=ddq_lb,
                ub=ddq_ub,
                solver="osqp",
            )
        except Exception:
            ddq = None

        if ddq is None or not np.isfinite(ddq).all():
            self.mj_data.ctrl[: self.n_arm] = 0.0
            return

        tau = M @ ddq + h
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        self.mj_data.ctrl[: self.n_arm] = tau

    def _get_action_scale(self, dist: float):
        if not self.use_adaptive_action_scale:
            return self.base_max_step_size, self.base_max_rot_step

        if dist < self.very_near_dist_threshold:
            return self.near_max_step_size, self.near_max_rot_step

        if dist < self.near_dist_threshold:
            return self.mid_max_step_size, self.mid_max_rot_step

        return self.base_max_step_size, self.base_max_rot_step

    def _get_obs(self):
        q = self.mj_data.qpos[: self.n_arm].copy()
        dq = self.mj_data.qvel[: self.n_arm].copy()

        ee_pos = self._get_ee_pos()
        ee_quat = self._get_ee_quat()

        pos_err = self.target_pos - ee_pos
        ori_err = self._orientation_error(self.target_quat, ee_quat)

        obs = np.concatenate(
            [
                q,
                dq,
                ee_pos,
                pos_err,
                ori_err,
                self.prev_action,
            ]
        )

        return obs.astype(np.float32)

    def _get_ee_pos(self):
        q_pin = np.zeros(self.n_pin, dtype=float)
        q_pin[: self.n_arm] = self.mj_data.qpos[: self.n_arm].copy()

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        return self.pin_data.oMf[self.pin_ee_id].translation.copy()

    def _get_ee_quat(self):
        q_pin = np.zeros(self.n_pin, dtype=float)
        q_pin[: self.n_arm] = self.mj_data.qpos[: self.n_arm].copy()

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        quat = pin.Quaternion(self.pin_data.oMf[self.pin_ee_id].rotation)
        quat.normalize()

        return quat

    def _get_ee_velocity(self):
        q_arm = self.mj_data.qpos[: self.n_arm].copy()
        v_arm = self.mj_data.qvel[: self.n_arm].copy()

        q_pin = np.zeros(self.n_pin, dtype=float)
        v_pin = np.zeros(self.n_pin, dtype=float)

        q_pin[: self.n_arm] = q_arm
        v_pin[: self.n_arm] = v_arm

        pin.forwardKinematics(self.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        frame_vel = pin.getFrameVelocity(
            self.pin_model,
            self.pin_data,
            self.pin_ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        return frame_vel.linear.copy(), frame_vel.angular.copy()

    def _orientation_error(self, target_quat: pin.Quaternion, current_quat: pin.Quaternion):
        return pin.log3((target_quat * current_quat.inverse()).matrix())

    def _integrate_quat(self, quat: pin.Quaternion, delta_rotvec: np.ndarray):
        if np.linalg.norm(delta_rotvec) < 1e-9:
            new_quat = pin.Quaternion(quat.matrix())
        else:
            new_quat = pin.Quaternion(pin.exp3(delta_rotvec) @ quat.matrix())

        new_quat.normalize()
        return new_quat

    def _quat_wxyz(self, quat: pin.Quaternion):
        coeffs = np.asarray(quat.coeffs()).reshape(-1)
        return np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]], dtype=float)

    def _sample_target_pos(self):
        if not self.randomize_target_pos:
            return self.fixed_target_pos.copy()

        low = np.array([0.42, -0.15, 0.34], dtype=float)
        high = np.array([0.58, 0.15, 0.54], dtype=float)

        return self.np_random.uniform(low=low, high=high).astype(float)

    def _sample_target_quat(self):
        if self.orientation_level == "none":
            quat = pin.Quaternion(self.q_ref.matrix())
            quat.normalize()
            return quat

        if self.orientation_level == "small":
            roll_lim, pitch_lim, yaw_lim = 10.0, 10.0, 30.0
        elif self.orientation_level == "medium":
            roll_lim, pitch_lim, yaw_lim = 20.0, 20.0, 90.0
        else:
            raise ValueError("orientation_level must be one of: 'none', 'small', 'medium'")

        roll = np.deg2rad(self.np_random.uniform(-roll_lim, roll_lim))
        pitch = np.deg2rad(self.np_random.uniform(-pitch_lim, pitch_lim))
        yaw = np.deg2rad(self.np_random.uniform(-yaw_lim, yaw_lim))

        target_rot = self._rpy_to_rot(roll, pitch, yaw) @ self.q_ref.matrix()

        quat = pin.Quaternion(target_rot)
        quat.normalize()

        return quat

    def _rpy_to_rot(self, roll: float, pitch: float, yaw: float):
        r_roll = pin.exp3(np.array([roll, 0.0, 0.0]))
        r_pitch = pin.exp3(np.array([0.0, pitch, 0.0]))
        r_yaw = pin.exp3(np.array([0.0, 0.0, yaw]))

        return r_yaw @ r_pitch @ r_roll

    def _get_mocap_idx(self, body_name: str):
        body_id = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id == -1:
            return -1

        return int(self.mj_model.body_mocapid[body_id])

    def _update_target_visuals(self):
        if self.target_mocap_idx != -1:
            self.mj_data.mocap_pos[self.target_mocap_idx] = self.target_pos

        if self.target_arrow_mocap_idx != -1:
            self.mj_data.mocap_pos[self.target_arrow_mocap_idx] = self.target_pos
            self.mj_data.mocap_quat[self.target_arrow_mocap_idx] = self._quat_wxyz(
                self.target_quat
            )

    def _update_action_visuals(self, desired_pos: np.ndarray, desired_quat: pin.Quaternion):
        if self.action_mocap_idx != -1:
            self.mj_data.mocap_pos[self.action_mocap_idx] = desired_pos

        if self.action_arrow_mocap_idx != -1:
            self.mj_data.mocap_pos[self.action_arrow_mocap_idx] = desired_pos
            self.mj_data.mocap_quat[self.action_arrow_mocap_idx] = self._quat_wxyz(
                desired_quat
            )

    def _update_ee_visuals(self, ee_pos: np.ndarray, ee_quat: pin.Quaternion):
        if self.ee_arrow_mocap_idx != -1:
            self.mj_data.mocap_pos[self.ee_arrow_mocap_idx] = ee_pos
            self.mj_data.mocap_quat[self.ee_arrow_mocap_idx] = self._quat_wxyz(ee_quat)

    def _render_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.viewer.sync()

    def render(self):
        if self.is_play_mode:
            self._render_viewer()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                print(f"[Warning] MuJoCo viewer close failed: {e}")
            finally:
                self.viewer = None