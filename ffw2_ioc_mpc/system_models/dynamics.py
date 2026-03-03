import casadi as ca
import numpy as np
import mujoco
from typing import Optional, Sequence, List


class DynamicsModel:
    """
    MuJoCo 물리 엔진의 질량 행렬(M)과 바이어스 힘(C+G)을 파라미터로 주입받아
    로봇 전체의 차세대 상태를 예측하는 CasADi 기반 동역학 모델 클래스입니다.

    변경 사항 (v2):
        - __init__ 시그니처에 input_dim: int 파라미터 추가
          → system_params.yaml의 input_dimension(=31)을 외부에서 주입
        - _build_input_mapping()이 모든 31개 액츄에이터를 처리하도록 수정
          · motor 타입 → actuator_trnid[act_id, 0]으로 dof_id를 조회해 B 행렬에 배치
          · position / velocity 타입 → 직접 조인트 토크를 적용하지 않으므로 B 열은 0
        - 기존 하드코딩된 self.input_dim = 14 제거
    """

    def __init__(
        self,
        dt: float,
        model_xml_path: str,
        input_dim: Optional[int] = None,
        actuator_indices: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            dt            : 제어 주기 (초), system_params.yaml의 time_step
            model_xml_path: ffw_sg2.xml (또는 scene_ffw_sg2.xml) 경로
            input_dim     : 제어 입력 차원 = model.nu = 31
                            system_params.yaml의 input_dimension 값을 전달
        """
        self.dt = dt

        # ── 1. MuJoCo 모델 로드 및 차원 파싱 ───────────────────────
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(model_xml_path)
        except Exception as e:
            raise RuntimeError(f"MuJoCo 모델 로드 실패: {e}")

        self.nq = self.mj_model.nq   # 38 (freejoint 쿼터니언 포함)
        self.nv = self.mj_model.nv   # 37
        print(f"[DynamicsModel] nq={self.nq} / nv={self.nv}")

        # 상태 차원: qpos(nq) + qvel(nv) = 75
        self.state_dim = self.nq + self.nv

        # 입력 차원: 외부 주입 (system_params.yaml → input_dimension)
        # ※ model.nu 와 일치해야 함
        if input_dim is None:
            input_dim = int(self.mj_model.nu)
        if input_dim != self.mj_model.nu:
            print(
                f"[DynamicsModel] 참고: input_dim({input_dim})이 model.nu({self.mj_model.nu})와 다릅니다. "
                f"전달된 input_dim({input_dim})을 사용합니다. (팔 토크만 사용하는 경우 정상)"
            )
            # input_dim = self.mj_model.nu
        self.input_dim = input_dim
        self.input_actuator_indices = self._resolve_input_actuator_indices(actuator_indices)

        # ── 2. CasADi 심볼릭 변수 및 파라미터 정의 ────────────────
        self.x_sym   = ca.MX.sym('x',  self.state_dim)
        self.u_sym   = ca.MX.sym('u',  self.input_dim)

        # 매 타임스텝 외부에서 주입받을 물리 행렬 심볼
        self.M_param  = ca.MX.sym('M',  self.nv, self.nv)  # 질량 행렬
        self.CG_param = ca.MX.sym('CG', self.nv, 1)        # Coriolis + 중력 바이어스

        # ── 3. 입력 매핑 행렬 B 구축 ───────────────────────────────
        # B : (nv, input_dim) — u 벡터를 전체 조인트 토크 tau에 매핑
        self.B = self._build_input_mapping()

        # ── 4. CasADi 동역학 함수 빌드 ─────────────────────────────
        self._f_casadi = self._define_casadi_dynamics()

    # ================================================================
    # 내부 메서드
    # ================================================================

    # def _build_input_mapping(self) -> ca.MX:
    #     """
    #     B 행렬 (nv, input_dim) 구축.

    #     ffw_sg2.xml 액츄에이터 순서 (index 0~30):
    #         0  left_wheel_steer_act       position
    #         1  right_wheel_steer_act      position
    #         2  rear_wheel_steer_act       position
    #         3  left_wheel_drive_act       velocity
    #         4  right_wheel_drive_act      velocity
    #         5  rear_wheel_drive_act       velocity
    #         6  actuator_lift_joint        position
    #         7  actuator_head_joint1       position
    #         8  actuator_head_joint2       position
    #         9  motor_arm_l_joint1         motor  ← 토크 기여 O
    #         ...
    #         15 motor_arm_l_joint7         motor
    #         16 motor_arm_r_joint1         motor
    #         ...
    #         22 motor_arm_r_joint7         motor
    #         23 actuator_gripper_l_joint1  position
    #         ...
    #         30 actuator_gripper_r_joint4  position

    #     motor 타입만 실제 조인트 토크를 생성합니다.
    #     position / velocity 타입은 MuJoCo 내부 서보 제어로 처리되므로
    #     EOM 상의 외력(τ)에 직접 들어가지 않습니다 → B 열 = 0.

    #     model.actuator_trnid[act_id, 0]  : 해당 액츄에이터가 제어하는 joint_id
    #     model.jnt_dofadr[joint_id]       : 해당 joint의 qvel(= dof) 시작 인덱스
    #     model.actuator_gear[act_id, 0]   : 기어비 (robot_torque 클래스는 1.0)
    #     """
    #     B_np = np.zeros((self.nv, self.input_dim))
    #     model = self.mj_model

    #     for act_id in range(self.input_dim):
    #         # MuJoCo 액츄에이터 타입 확인
    #         # mjACTUATOR_MOTOR == 0  (mjtActuator enum)
    #         act_type = model.actuator_dyntype[act_id]   # 동역학 타입

    #         # trntype 0 = joint, trnid[act_id, 0] = joint_id
    #         trn_type = model.actuator_trntype[act_id]
    #         if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
    #             # 조인트가 아닌 대상(예: 텐던)은 스킵
    #             continue

    #         joint_id = int(model.actuator_trnid[act_id, 0])

    #         # motor 타입 액츄에이터만 τ를 직접 생성
    #         # actuator_gaintype == mjGAIN_FIXED (0) + biastype == mjBIAS_NONE (0)
    #         # 가장 확실한 방법: gaintype이 FIXED이고 biastype가 NONE이면 motor
    #         gain_type = model.actuator_gaintype[act_id]
    #         bias_type = model.actuator_biastype[act_id]

    #         is_motor = (
    #             gain_type == mujoco.mjtGain.mjGAIN_FIXED
    #             and bias_type == mujoco.mjtBias.mjBIAS_NONE
    #         )

    #         if not is_motor:
    #             # position / velocity 서보 액츄에이터: B 열 0 유지
    #             continue

    #         dof_id   = int(model.jnt_dofadr[joint_id])
    #         gear_val = float(model.actuator_gear[act_id, 0])

    #         B_np[dof_id, act_id] = gear_val if gear_val != 0.0 else 1.0

    #     n_nonzero = int(np.count_nonzero(B_np))
    #     print(f"[DynamicsModel] B 행렬 구축 완료: shape={B_np.shape}, "
    #           f"비영(motor) 항목 수={n_nonzero}")

    #     return ca.MX(B_np)

    def _resolve_input_actuator_indices(
        self,
        actuator_indices: Optional[Sequence[int]],
    ) -> List[int]:
        """
        입력 벡터 u의 각 채널이 어떤 MuJoCo actuator를 의미하는지 결정합니다.
        """
        if actuator_indices is None:
            if self.input_dim == 7:
                indices = list(range(16, 23))      # arm_r_joint1~7
            elif self.input_dim == 14:
                indices = list(range(9, 23))       # arm_l_joint1~7 + arm_r_joint1~7
            elif self.input_dim == self.mj_model.nu:
                indices = list(range(self.mj_model.nu))
            else:
                raise ValueError(
                    "actuator_indices가 지정되지 않았고 input_dim에 대한 기본 매핑도 없습니다. "
                    f"(input_dim={self.input_dim}, model.nu={self.mj_model.nu})"
                )
        else:
            indices = [int(i) for i in actuator_indices]

        if len(indices) != self.input_dim:
            raise ValueError(
                f"actuator_indices 길이({len(indices)}) != input_dim({self.input_dim})"
            )
        if not indices:
            raise ValueError("actuator_indices가 비어 있습니다.")

        lo = min(indices)
        hi = max(indices)
        if lo < 0 or hi >= self.mj_model.nu:
            raise ValueError(
                "actuator_indices 범위 오류: "
                f"[{lo}, {hi}] not in [0, {self.mj_model.nu - 1}]"
            )
        return indices

    def _is_motor_joint_actuator(self, act_id: int) -> bool:
        model = self.mj_model
        trn_type = model.actuator_trntype[act_id]
        if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
            return False
        gain_type = model.actuator_gaintype[act_id]
        bias_type = model.actuator_biastype[act_id]
        return (
            gain_type == mujoco.mjtGain.mjGAIN_FIXED
            and bias_type == mujoco.mjtBias.mjBIAS_NONE
        )

    def _build_input_mapping(self) -> ca.MX:
        B_np = np.zeros((self.nv, self.input_dim))
        model = self.mj_model

        for col_idx, act_id in enumerate(self.input_actuator_indices):
            if not self._is_motor_joint_actuator(act_id):
                continue

            joint_id = int(model.actuator_trnid[act_id, 0])
            dof_id = int(model.jnt_dofadr[joint_id])
            gear_val = float(model.actuator_gear[act_id, 0])
            B_np[dof_id, col_idx] = gear_val if gear_val != 0.0 else 1.0

        print(
            f"[DynamicsModel] B 행렬 구축 완료: shape={B_np.shape}, "
            f"비영(motor) 항목 수={int(np.count_nonzero(B_np))}, "
            f"actuator_indices={self.input_actuator_indices}"
        )
        return ca.MX(B_np)
    
    def _define_casadi_dynamics(self) -> ca.Function:
        """
        심볼릭 동역학 수식 정의:

            M(q) * ddq = B * u - CG(q, dq)
            ddq = M^{-1} * (B*u - CG)

            dq_next = dq + ddq * dt          (Semi-implicit Euler)
            q_next  = q  + dq_next * dt      (단, freejoint 쿼터니언 처리 포함)
        """
        q  = self.x_sym[:self.nq]   # (38,)
        dq = self.x_sym[self.nq:]   # (37,)

        # ── 가속도 계산 ──────────────────────────────────────────────
        tau_total = ca.mtimes(self.B, self.u_sym)           # (nv,)
        ddq       = ca.solve(self.M_param, tau_total - self.CG_param)  # (nv,)

        # ── 속도 업데이트 (Semi-implicit Euler) ─────────────────────
        dq_next = dq + ddq * self.dt   # (37,)

        # ── 위치 업데이트 ────────────────────────────────────────────
        # freejoint: qpos 7차원 (pos 3 + quat 4), qvel 6차원 (vel 3 + angvel 3)
        # 나머지 조인트: qpos == qvel 차원 (1:1 매핑)

        q_base   = q[:7]          # freejoint qpos  (pos3 + quat4)
        q_joints = q[7:]          # 나머지 조인트 qpos (38-7 = 31)

        dq_base   = dq_next[:6]   # freejoint qvel  (vel3 + angvel3)
        dq_joints = dq_next[6:]   # 나머지 조인트 qvel (37-6 = 31)

        # 나머지 조인트: 단순 적분 (차원 일치)
        q_joints_next = q_joints + dq_joints * self.dt   # (31,)

        # 베이스 선위치: 단순 적분
        pos_base_next  = q_base[:3] + dq_base[:3] * self.dt   # (3,)

        # 쿼터니언: 1차 근사 적분
        # q_next = q + 0.5 * Ω(ω) * q * dt  (단순화: 소각 근사)
        # MPC 내부 예측 용도이므로 정규화 생략 허용
        qw, qx, qy, qz = q_base[3], q_base[4], q_base[5], q_base[6]
        wx, wy, wz      = dq_base[3], dq_base[4], dq_base[5]
        half_dt = 0.5 * self.dt

        dqw = half_dt * (-qx*wx - qy*wy - qz*wz)
        dqx = half_dt * ( qw*wx + qy*wz - qz*wy)
        dqy = half_dt * ( qw*wy - qx*wz + qz*wx)
        dqz = half_dt * ( qw*wz + qx*wy - qy*wx)

        quat_next = ca.vertcat(qw + dqw, qx + dqx, qy + dqy, qz + dqz)   # (4,)

        q_base_next = ca.vertcat(pos_base_next, quat_next)   # (7,)
        q_next      = ca.vertcat(q_base_next, q_joints_next) # (38,)
        x_next      = ca.vertcat(q_next, dq_next)            # (75,)

        # ── CasADi Function 생성 ─────────────────────────────────────
        return ca.Function(
            'f_dynamics',
            [self.x_sym, self.u_sym, self.M_param, self.CG_param],
            [x_next],
            ['x', 'u', 'M', 'CG'],
            ['x_next']
        )

    # ================================================================
    # 공개 인터페이스
    # ================================================================

    def predict(self, x: np.ndarray, u: np.ndarray,
                mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> np.ndarray:
        """
        시뮬레이션 루프에서 실제로 다음 상태를 예측할 때 호출합니다.
        mj_data에서 현재의 M, CG 수치를 직접 추출하여 주입합니다.

        Args:
            x        : (state_dim,) 현재 상태 벡터
            u        : (input_dim,) 제어 입력 벡터 (data.ctrl 전체, 31차원)
            mj_model : MuJoCo MjModel
            mj_data  : MuJoCo MjData (현재 상태가 설정된 상태)

        Returns:
            np.ndarray: (state_dim,) 다음 상태 예측값
        """
        M_np  = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(mj_model, M_np, mj_data.qM)
        CG_np = mj_data.qfrc_bias.copy()   # (nv,)

        result = self._f_casadi(x, u, M_np, CG_np)
        return np.array(result).flatten()

    def get_casadi_func(self) -> ca.Function:
        """KKTBuilder나 MPC Controller에서 수식을 참조할 때 사용합니다."""
        return self._f_casadi


# ====================================================================
# 테스트 (파일 직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path    = os.path.join(current_dir, "mujoco_models", "ffw_sg2.xml")

    if not os.path.exists(xml_path):
        print(f"XML 파일을 찾을 수 없습니다: {os.path.abspath(xml_path)}")
    else:
        model = DynamicsModel(dt=0.02, model_xml_path=xml_path, input_dim=31)
        print(f"\nDynamicsModel 초기화 성공!")
        print(f"  state_dim : {model.state_dim}  (nq={model.nq}, nv={model.nv})")
        print(f"  input_dim : {model.input_dim}")

        # 더미 데이터로 CasADi 함수 호출 테스트
        x_test  = np.zeros(model.state_dim)
        x_test[3 + 3] = 1.0   # quat w = 1 (valid quaternion)
        u_test  = np.zeros(model.input_dim)
        M_test  = np.eye(model.nv)
        CG_test = np.zeros(model.nv)

        f = model.get_casadi_func()
        x_next_sym = f(x_test, u_test, M_test, CG_test)
        print(f"\n  CasADi 함수 호출 성공: x_next shape = {np.array(x_next_sym).shape}")
        print("✅ DynamicsModel 테스트 통과")
