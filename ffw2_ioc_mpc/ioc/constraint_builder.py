"""
src/ioc/constraint_builder.py

IOC 학습을 위한 후보 제약 조건을 생성하는 클래스.

논문 Section II.B.1 기준:
    제약 조건 형태: P @ z <= p   (z = [x; u])

생성 방법:
    1. 도메인 지식 기반  : 관절 위치/속도 한계, 입력 토크 한계
    2. 데이터 기반       : 시연 궤적의 볼록 껍질(Convex Hull) 면(facet)

z 벡터 인덱스 구조 (dynamics.py 기준):
    z[0      : nq       ]  → qpos  (MuJoCo jnt_qposadr 기준)
    z[nq     : nq+nv    ]  → qvel  (MuJoCo jnt_dofadr 기준, +nq 오프셋)
    z[nq+nv  : nq+nv+14 ]  → u     (arm_l_joint1~7, arm_r_joint1~7 순서)
"""

import os
import warnings
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from typing import List, Dict, Any, Optional

import mujoco

try:
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint


class ConstraintBuilder:
    """
    처리된 시연 데이터와 도메인 지식을 바탕으로
    Pz <= p 형태의 후보 제약 조건 리스트를 생성합니다.

    사용 예시:
        builder = ConstraintBuilder(system_params, builder_config)
        constraints = builder.build_candidate_constraints(states_seg, inputs_seg)
        # constraints : List[PolytopeConstraint]
    """

    def __init__(
        self,
        system_params  : Dict[str, Any],
        builder_config : Dict[str, Any],
    ):
        """
        Args:
            system_params (dict):
                configs/system_params.yaml 로드 결과.
                필수 키 구조:
                    system:
                        state_dimension : 75
                        input_dimension : 14
                        nq              : 38
                        nv              : 37
                        model_xml_path  : "src/system_models/mujoco_models/ffw_sg2.xml"

            builder_config (dict):
                configs/experiment_params.yaml의 'constraint_builder' 섹션.
                예시:
                    include_domain_knowledge_constraints: true
                    include_convex_hull_constraints: false
                    convex_hull_pca_dim: null        # null이면 PCA 미적용

                    input_torque_limits_u:
                        min: -100.0
                        max:  100.0

                    joint_limits_qpos:
                        arm_l_joint1: {min: -3.14, max: 3.14}
                        arm_l_joint2: {min:  0.0,  max: 3.14}

                    joint_vel_limits_qvel:
                        arm_l_joint1: {min: -5.0, max: 5.0}
        """
        sys_cfg = system_params['system']

        self.state_dim : int = sys_cfg['state_dimension']
        self.input_dim : int = sys_cfg['input_dimension']
        self.nq        : int = sys_cfg['nq']
        self.nv        : int = sys_cfg['nv']
        self.z_dim     : int = self.state_dim + self.input_dim

        self.cfg = builder_config

        # ── MuJoCo 모델 로드 ────────────────────────────────────────
        self.mj_model: Optional[mujoco.MjModel] = None
        xml_path = sys_cfg.get('model_xml_path', None)

        if xml_path:
            if not os.path.isabs(xml_path):
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                )
                xml_path = os.path.join(project_root, xml_path)
            try:
                self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
                print(f"  [ConstraintBuilder] MuJoCo 모델 로드 성공: {xml_path}")
            except Exception as e:
                warnings.warn(
                    f"[ConstraintBuilder] MuJoCo 모델 로드 실패: {e}\n"
                    "  조인트 이름 기반 제약 조건은 비활성화됩니다."
                )
        else:
            warnings.warn(
                "[ConstraintBuilder] 'model_xml_path'가 없습니다. "
                "조인트 이름 기반 제약 조건은 비활성화됩니다."
            )

        print("=" * 60)
        print("ConstraintBuilder 초기화 완료")
        print(f"  z_dim          : {self.z_dim}  "
              f"(state={self.state_dim}, input={self.input_dim})")
        print(f"  mujoco 모델    : {'로드됨' if self.mj_model else '없음'}")
        print(f"  domain 제약    : {self.cfg.get('include_domain_knowledge_constraints', True)}")
        print(f"  convexhull 제약: {self.cfg.get('include_convex_hull_constraints', False)}")
        print("=" * 60)

    # ================================================================
    # 공개 인터페이스
    # ================================================================

    def build_candidate_constraints(
        self,
        processed_states : np.ndarray,
        processed_inputs : np.ndarray,
    ) -> List[PolytopeConstraint]:
        """
        후보 제약 조건 리스트를 생성합니다.

        Args:
            processed_states : (e+1, state_dim)  DataProcessor 출력
            processed_inputs : (e,   input_dim)  DataProcessor 출력

        Returns:
            List[PolytopeConstraint]: 각 객체의 z_dim == self.z_dim
        """
        self._validate_segment_shapes(processed_states, processed_inputs)

        candidates: List[PolytopeConstraint] = []

        # ── 1. 도메인 지식 기반 ──────────────────────────────────────
        if self.cfg.get('include_domain_knowledge_constraints', True):
            print("  [도메인 지식 제약 조건 생성 중...]")
            domain = self._get_domain_knowledge_constraints()
            candidates.extend(domain)
            print(f"    → {len(domain)}개 추가")

        # ── 2. 볼록 껍질 기반 ────────────────────────────────────────
        # processed_states는 e+1개, processed_inputs는 e개이므로
        # (x_0,u_0) ~ (x_{e-1}, u_{e-1}) 의 e개 쌍을 사용
        if self.cfg.get('include_convex_hull_constraints', False):
            print("  [볼록 껍질 제약 조건 생성 중... (고차원에서 오래 걸릴 수 있습니다)]")
            hull_constraints = self._get_convex_hull_constraints(
                states_for_hull=processed_states[:-1, :],
                inputs_for_hull=processed_inputs,
            )
            candidates.extend(hull_constraints)
            print(f"    → {len(hull_constraints)}개 추가")

        print(f"  총 후보 제약 조건: {len(candidates)}개")
        return candidates

    def get_constraint_summary(
        self,
        constraints: List[PolytopeConstraint],
    ) -> str:
        """생성된 제약 조건 리스트의 요약 문자열을 반환합니다."""
        total_ineq = sum(c.num_ineq for c in constraints)
        lines = [
            f"제약 조건 요약 (PolytopeConstraint {len(constraints)}개 / "
            f"총 부등식 {total_ineq}개):"
        ]
        for i, c in enumerate(constraints):
            lines.append(f"  [{i:3d}] {c.summary()}")
        return "\n".join(lines)

    # ================================================================
    # 내부 메서드: 도메인 지식 기반
    # ================================================================

    def _get_domain_knowledge_constraints(self) -> List[PolytopeConstraint]:
        """
        물리적 한계로부터 PolytopeConstraint를 생성합니다.

        z 벡터 인덱스 매핑:
            qpos  : z[0 : nq]             (MuJoCo jnt_qposadr 직접 사용)
            qvel  : z[nq : nq+nv]         (MuJoCo jnt_dofadr + nq 오프셋)
            u     : z[nq+nv : nq+nv+14]   (입력 순서 그대로)
        """
        constraints: List[PolytopeConstraint] = []

        # ── 입력 토크 한계 ───────────────────────────────────────────
        torque_cfg = self.cfg.get('input_torque_limits_u', None)
        if torque_cfg is not None:
            constraints.extend(
                self._make_block_box_for_u(torque_cfg)
            )

        # ── 관절 위치 한계 (qpos) ────────────────────────────────────
        qpos_cfg = self.cfg.get('joint_limits_qpos', None)
        if qpos_cfg and self.mj_model is not None:
            for joint_name, limits in qpos_cfg.items():
                jid = self._get_joint_id(joint_name)
                if jid is None:
                    continue
                # jnt_qposadr: MuJoCo qpos 배열 내 해당 조인트의 인덱스
                # x = [qpos|qvel] 이므로 z 내 인덱스 = jnt_qposadr 그대로
                z_idx = int(self.mj_model.jnt_qposadr[jid])
                constraints.extend(
                    self._make_scalar_box(z_idx, limits, f"{joint_name} qpos")
                )
                print(f"    - qpos 한계: {joint_name}  "
                      f"[{limits.get('min','−∞')}, {limits.get('max','+∞')}]")

        # ── 관절 속도 한계 (qvel) ────────────────────────────────────
        qvel_cfg = self.cfg.get('joint_vel_limits_qvel', None)
        if qvel_cfg and self.mj_model is not None:
            for joint_name, limits in qvel_cfg.items():
                jid = self._get_joint_id(joint_name)
                if jid is None:
                    continue
                # jnt_dofadr: MuJoCo qvel 배열 내 인덱스
                # x = [qpos(nq) | qvel(nv)] 이므로 z 내 인덱스 = nq + jnt_dofadr
                z_idx = self.nq + int(self.mj_model.jnt_dofadr[jid])
                constraints.extend(
                    self._make_scalar_box(z_idx, limits, f"{joint_name} qvel")
                )
                print(f"    - qvel 한계: {joint_name}  "
                      f"[{limits.get('min','−∞')}, {limits.get('max','+∞')}]")

        return constraints

    def _make_block_box_for_u(
        self,
        limits: Dict[str, float],
    ) -> List[PolytopeConstraint]:
        """
        입력 u 전체(input_dim개)에 대해 박스 제약 조건을 생성합니다.
        (input_dim 행을 묶은 단일 PolytopeConstraint 최대 2개 반환)

        생성 제약:
            +I_m @ u_block <= max  →  u_i <= max
            -I_m @ u_block <= -min →  u_i >= min
        """
        constraints = []
        min_u = limits.get('min', -np.inf)
        max_u = limits.get('max',  np.inf)
        u_start = self.state_dim   # = nq + nv

        if np.isfinite(max_u):
            P_ub = np.zeros((self.input_dim, self.z_dim))
            P_ub[:, u_start : u_start + self.input_dim] = np.eye(self.input_dim)
            p_ub = np.full((self.input_dim, 1), max_u)
            constraints.append(
                PolytopeConstraint(P_ub, p_ub, f"Input u upper ({max_u})")
            )
            print(f"    - u 상한 {max_u}  ({self.input_dim}개 부등식)")

        if np.isfinite(min_u):
            P_lb = np.zeros((self.input_dim, self.z_dim))
            P_lb[:, u_start : u_start + self.input_dim] = -np.eye(self.input_dim)
            p_lb = np.full((self.input_dim, 1), -min_u)
            constraints.append(
                PolytopeConstraint(P_lb, p_lb, f"Input u lower ({min_u})")
            )
            print(f"    - u 하한 {min_u}  ({self.input_dim}개 부등식)")

        return constraints

    def _make_scalar_box(
        self,
        z_idx      : int,
        limits     : Dict[str, float],
        name_prefix: str,
    ) -> List[PolytopeConstraint]:
        """
        z 벡터의 단일 인덱스에 대한 스칼라 상/하한 제약 조건 (최대 2개).

        제약:
            +z[z_idx] <=  max_val
            -z[z_idx] <= -min_val
        """
        constraints = []
        min_val = limits.get('min', -np.inf)
        max_val = limits.get('max',  np.inf)

        if np.isfinite(max_val):
            P_ub = np.zeros((1, self.z_dim))
            P_ub[0, z_idx] = 1.0
            constraints.append(
                PolytopeConstraint(
                    P_ub, np.array([[max_val]]),
                    f"{name_prefix} upper"
                )
            )

        if np.isfinite(min_val):
            P_lb = np.zeros((1, self.z_dim))
            P_lb[0, z_idx] = -1.0
            constraints.append(
                PolytopeConstraint(
                    P_lb, np.array([[-min_val]]),
                    f"{name_prefix} lower"
                )
            )

        return constraints

    # ================================================================
    # 내부 메서드: 볼록 껍질 기반
    # ================================================================

    def _get_convex_hull_constraints(
        self,
        states_for_hull : np.ndarray,
        inputs_for_hull : np.ndarray,
    ) -> List[PolytopeConstraint]:
        """
        z_k = [x_k; u_k] 데이터 포인트의 볼록 껍질 면(facet)으로 제약 조건을 생성합니다.

        고차원(z_dim=89) 문제에서는 반드시 convex_hull_pca_dim을 설정하여
        PCA 차원 축소를 적용하세요.

        Args:
            states_for_hull : (e, state_dim)
            inputs_for_hull : (e, input_dim)
        """
        e = states_for_hull.shape[0]
        Z = np.hstack([states_for_hull, inputs_for_hull])   # (e, z_dim)

        # ── PCA 차원 축소 (선택) ──────────────────────────────────────
        pca_dim = self.cfg.get('convex_hull_pca_dim', None)
        mean, components = None, None

        if pca_dim is not None and 0 < pca_dim < self.z_dim:
            print(f"    PCA 차원 축소: {self.z_dim}D → {pca_dim}D")
            Z_work, mean, components = self._pca_reduce(Z, pca_dim)
        else:
            Z_work = Z

        working_dim = Z_work.shape[1]

        if e < working_dim + 1:
            warnings.warn(
                f"[볼록 껍질] 데이터 포인트({e}개) < 작업 차원({working_dim})+1. "
                "볼록 껍질을 건너뜁니다. "
                "convex_hull_pca_dim을 줄이거나 데이터를 늘려보세요."
            )
            return []

        try:
            hull = ConvexHull(Z_work)
        except QhullError as qe:
            warnings.warn(f"[볼록 껍질] Qhull 오류: {qe}")
            return []

        # hull.equations: 각 행 = [normal (working_dim,), offset]
        # 면의 방정식: normal @ z + offset <= 0  →  normal @ z <= -offset
        constraints = []
        for eq in hull.equations:
            normal = eq[:-1]
            offset = eq[-1]

            if components is not None:
                # PCA 역변환: 축소 공간의 법선 → 원래 z 공간의 법선
                # z_work = (z - mean) @ components.T
                # n^T z_work = n^T (z-mean) @ components.T
                #            = (components n)^T (z - mean)
                #            = (components n)^T z - (components n)^T mean
                normal_z = components.T @ normal           # (z_dim,)
                offset_z = offset - float(normal_z @ mean) # scalar
            else:
                normal_z = normal
                offset_z = offset

            P_row = normal_z.reshape(1, -1)
            p_val = np.array([[-offset_z]])
            constraints.append(
                PolytopeConstraint(P_row, p_val, "ConvexHull facet")
            )

        print(f"    볼록 껍질 면 수: {len(constraints)}")
        return constraints

    @staticmethod
    def _pca_reduce(Z: np.ndarray, n_components: int):
        """
        PCA로 Z를 n_components 차원으로 축소합니다.

        Returns:
            Z_reduced  : (N, n_components)
            mean       : (z_dim,)
            components : (n_components, z_dim)  주성분 행벡터
        """
        mean = Z.mean(axis=0)
        Z_c  = Z - mean
        _, _, Vt = np.linalg.svd(Z_c, full_matrices=False)
        components = Vt[:n_components, :]
        Z_reduced  = Z_c @ components.T
        return Z_reduced, mean, components

    # ================================================================
    # 헬퍼
    # ================================================================

    def _get_joint_id(self, joint_name: str) -> Optional[int]:
        """MuJoCo 모델에서 조인트 ID를 조회합니다. 없으면 None 반환."""
        if self.mj_model is None:
            return None
        jid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if jid == -1:
            warnings.warn(
                f"[ConstraintBuilder] 조인트 '{joint_name}'를 모델에서 찾을 수 없습니다. "
                "건너뜁니다."
            )
            return None
        return int(jid)

    def _validate_segment_shapes(
        self,
        states : np.ndarray,
        inputs : np.ndarray,
    ) -> None:
        """DataProcessor 출력 형상 검증."""
        e_p1, sd = states.shape
        e,    id_ = inputs.shape

        if e_p1 != e + 1:
            raise ValueError(
                f"states.shape[0]({e_p1}) != inputs.shape[0]({e}) + 1"
            )
        if sd != self.state_dim:
            raise ValueError(
                f"states.shape[1]({sd}) != state_dim({self.state_dim})"
            )
        if id_ != self.input_dim:
            raise ValueError(
                f"inputs.shape[1]({id_}) != input_dim({self.input_dim})"
            )


# ====================================================================
# 동작 확인 (직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    # ── 더미 설정 (MuJoCo 모델 없이 u 한계만 테스트) ────────────────
    system_params = {
        'system': {
            'state_dimension': 75,
            'input_dimension': 14,
            'nq': 38,
            'nv': 37,
            'model_xml_path': None,
        }
    }

    builder_config = {
        'include_domain_knowledge_constraints': True,
        'include_convex_hull_constraints': True,

        'input_torque_limits_u': {
            'min': -100.0,
            'max':  100.0,
        },

        # MuJoCo 모델이 있을 때 활성화:
        # 'joint_limits_qpos': {
        #     'arm_l_joint1': {'min': -3.14, 'max': 3.14},
        #     'arm_l_joint2': {'min':  0.0,  'max': 3.14},
        # },
        # 'joint_vel_limits_qvel': {
        #     'arm_l_joint1': {'min': -5.0, 'max': 5.0},
        # },

        # 볼록 껍질: 고차원이므로 PCA 필수
        'convex_hull_pca_dim': 10,
    }

    builder = ConstraintBuilder(system_params, builder_config)

    # ── 더미 세그먼트 (e=120) ─────────────────────────────────────────
    np.random.seed(42)
    e = 120
    states_seg = np.random.randn(e + 1, 75) * 0.1
    inputs_seg = np.random.randn(e,     14) * 5.0

    constraints = builder.build_candidate_constraints(states_seg, inputs_seg)
    print()
    print(builder.get_constraint_summary(constraints))

    # ── 만족도 검증 ───────────────────────────────────────────────────
    z_ok  = np.zeros(89)                        # 토크=0 → 한계 만족
    z_bad = np.zeros(89); z_bad[75] = 200.0    # u[0] = 200 → 상한 위반

    print("\n[z_ok 검증]")
    for c in constraints[:2]:
        print(f"  {c.name:<35} satisfied={c.is_satisfied(z_ok)}")

    print("\n[z_bad 검증 (u[0]=200)]")
    for c in constraints[:2]:
        viol = c.violation(z_bad).max()
        print(f"  {c.name:<35} satisfied={c.is_satisfied(z_bad)}  max_viol={viol:.1f}")

    # P 행렬 형상 확인
    for c in constraints:
        assert c.P.shape == (c.num_ineq, 89), \
            f"P 형상 오류: {c.P.shape}"
    print("\n✅ 모든 P 행렬 형상 검증 통과")
    print("✅ ConstraintBuilder 모든 테스트 통과")