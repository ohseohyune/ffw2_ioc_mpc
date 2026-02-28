"""
src/ioc/kkt_builder.py

논문의 KKT 조건을 CasADi 심볼릭 표현으로 구축하는 클래스.

────────────────────────────────────────────────────────────
논문 Eq.(8) 라그랑지안:
    L(U, λ, ν, θ) =
        ν^T (F_e(U, x(k)) - x(k+e))
      + Σ_{i=0}^{e-1} [ l(F_i(U, x(k)), u_i; θ)
                        + λ_i^T C̄(F_i(U, x(k)), u_i) ]

    여기서
        F_i(U, x(k)) : x(k)에서 u_0,...,u_{i-1}으로 전파된 x_i
        C̄(x_i, u_i) = P [x_i; u_i] - p   (모든 후보 제약 조건)
        λ_i ∈ R^{num_constraints}          (각 타임스텝 라그랑주 승수)
        ν   ∈ R^{state_dim}                (종단 상태 라그랑주 승수)

논문 Eq.(10a) KKT 정지 조건:
    ∇_U L(U, λ, ν, θ)|_{U=U^m} = 0

논문 Eq.(10b) KKT 상보성 여유 조건:
    λ_i^T C̄(x(k+i), u(k+i)) = 0   ∀i

────────────────────────────────────────────────────────────
출력 CasADi Function 시그니처:
    grad_L_U_func(theta, nu, lambda_flat, Xm_data, Um_data,
                   ys_data, M_data, CG_data)
        → gradient  : (e * input_dim,)   ∇_U L

    csc_func(theta, nu, lambda_flat, Xm_data, Um_data,
              ys_data, M_data, CG_data)
        → complementary_slackness : (e,)   λ_i^T C̄_i

────────────────────────────────────────────────────────────
입력 데이터 차원 (ffw_sg2.xml 기준):
    state_dim  = 75  (nq=38, nv=37)
    input_dim  = 14
    nv         = 37  (M, CG 차원 기준)
    e          = 세그먼트 스텝 수

    Xm_data  : (state_dim, e+1)   — CasADi 행렬, 열 = 시간
    Um_data  : (input_dim, e)
    ys_data  : (target_dim, e)
    M_data   : (nv, nv * e)       — 3D 텐서 대신 열 방향으로 이어 붙인 형태
    CG_data  : (nv, e)
"""

import numpy as np
import casadi as ca
from typing import List, Tuple, Dict, Any, Optional

try:
    from ffw2_ioc_mpc.system_models.dynamics import DynamicsModel
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from ffw2_ioc_mpc.system_models.dynamics import DynamicsModel
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint

class KKTBuilder:
    """
    IOC 역최적 제어를 위한 KKT 조건 잔차 함수를 CasADi로 구축합니다.

    핵심 설계 원칙:
        - 시연 데이터(Xm, Um, M, CG, ys)는 CasADi Function의 입력 인자로 처리
          → IOC Optimizer가 여러 시연 데이터로 반복 호출 가능
        - theta(비용 파라미터), nu(종단 승수), lambda(제약 승수)는 최적화 변수
        - 라그랑지안을 U에 대해 CasADi 자동 미분으로 ∇_U L 계산

    사용 순서:
        kkt = KKTBuilder(dynamics, stage_cost, constraints, e=120)
        grad_func, csc_func = kkt.build_kkt_residual_functions()

        # IOC optimizer에서:
        grad_val = grad_func(theta, nu, lambda_flat,
                             Xm_data, Um_data, ys_data, M_data, CG_data)
        csc_val  = csc_func(theta, nu, lambda_flat,
                             Xm_data, Um_data, ys_data, M_data, CG_data)
    """

    def __init__(
        self,
        dynamics_model       : DynamicsModel,
        stage_cost           : ParametricStageCost,
        candidate_constraints: List[PolytopeConstraint],
        e                    : int,
    ):
        """
        Args:
            dynamics_model (DynamicsModel):
                get_casadi_func() → f(x, u, M, CG) → x_next

            stage_cost (ParametricStageCost):
                get_casadi_function() → l(x, u, theta, ys) → scalar

            candidate_constraints (List[PolytopeConstraint]):
                ConstraintBuilder.build_candidate_constraints() 출력.
                각 PolytopeConstraint의 z_dim = state_dim + input_dim.

            e (int):
                궤적 세그먼트 스텝 수.
                DataProcessor.compute_segment_length()[0] 으로 얻을 수 있음.
        """
        self.dyn   = dynamics_model
        self.cost  = stage_cost
        self.cons  = candidate_constraints
        self.e     = e

        self.state_dim  = dynamics_model.state_dim    # 75
        self.input_dim  = dynamics_model.input_dim    # 14
        self.nv         = dynamics_model.nv           # 37
        self.theta_dim  = stage_cost.theta_dim
        self.target_dim = stage_cost.target_dim

        # 모든 PolytopeConstraint의 num_ineq 합산
        self.num_total_constraints: int = sum(
            pc.num_ineq for pc in candidate_constraints
        )

        # CasADi Function 캐시 (build 후 채워짐)
        self._grad_L_U_func: Optional[ca.Function] = None
        self._csc_func      : Optional[ca.Function] = None

        print("=" * 60)
        print("KKTBuilder 초기화 완료")
        print(f"  e                      : {self.e} steps")
        print(f"  state_dim / input_dim  : {self.state_dim} / {self.input_dim}")
        print(f"  theta_dim              : {self.theta_dim}")
        print(f"  후보 제약 조건 (개수)  : {len(candidate_constraints)}")
        print(f"  총 부등식 수           : {self.num_total_constraints}")
        print(f"  lambda 차원            : {self.e * self.num_total_constraints}")
        print("=" * 60)

    # ================================================================
    # 공개 인터페이스
    # ================================================================

    def build_kkt_residual_functions(self) -> Tuple[ca.Function, ca.Function]:
        """
        KKT 정지 조건(∇_U L)과 상보성 여유 조건(λ^T C̄)의 잔차를
        계산하는 CasADi Function 쌍을 반환합니다.

        두 함수 모두 동일한 입력 시그니처:
            theta       : (theta_dim,)
            nu          : (state_dim,)
            lambda_flat : (e * num_total_constraints,)
            Xm_data     : (state_dim, e+1)   — 열 = 타임스텝
            Um_data     : (input_dim, e)
            ys_data     : (target_dim, e)    — 열 = 타임스텝
            M_data      : (nv, nv * e)       — 각 스텝 M을 가로로 이어 붙임
            CG_data     : (nv, e)            — 각 스텝 CG를 열로

        Returns:
            grad_L_U_func : ∇_U L  →  (e * input_dim, 1)
            csc_func      : λ_i^T C̄_i  →  (e, 1)   (각 스텝 스칼라)
        """
        e       = self.e
        n_x     = self.state_dim
        n_u     = self.input_dim
        nv      = self.nv
        n_theta = self.theta_dim
        n_c     = self.num_total_constraints   # 한 타임스텝의 총 부등식 수

        # ── 1. 심볼릭 변수 선언 ────────────────────────────────────────
        # 최적화 변수 (IOC Optimizer가 갱신)
        theta_sym       = ca.MX.sym('theta',       n_theta)
        nu_sym          = ca.MX.sym('nu',           n_x)
        lambda_flat_sym = ca.MX.sym('lambda_flat',  e * n_c)

        # 시연 데이터 (Function 호출 시 수치값으로 제공)
        Xm_data_sym  = ca.MX.sym('Xm_data',  n_x, e + 1)   # (75, e+1)
        Um_data_sym  = ca.MX.sym('Um_data',  n_u, e)        # (14, e)
        ys_data_sym  = ca.MX.sym('ys_data',  self.target_dim, e)   # (target_dim, e)
        M_data_sym   = ca.MX.sym('M_data',   nv,  nv * e)   # (37, 37*e)
        CG_data_sym  = ca.MX.sym('CG_data',  nv,  e)        # (37, e)

        # ── 2. CasADi Function 참조 ────────────────────────────────────
        f_dyn = self.dyn.get_casadi_func()     # f(x, u, M, CG) → x_next
        l_fn  = self.cost.get_casadi_function()  # l(x, u, theta, ys) → scalar

        # ── 3. U 심볼 (∇_U 미분 대상) ─────────────────────────────────
        # Um_data_sym의 모든 열을 flatten한 벡터를 U_sym으로 사용
        # ∇_U L 계산 시 CasADi가 이 심볼에 대해 미분합니다.
        U_sym = ca.MX.sym('U', n_u * e)   # (e * input_dim,)

        # ── 4. 라그랑지안 구성 ─────────────────────────────────────────
        # x(k) : 시연 궤적의 시작 상태 (고정)
        # x(k+e): 시연 궤적의 끝 상태   (고정)
        x_k    = Xm_data_sym[:, 0]           # (n_x,)
        x_k_e  = Xm_data_sym[:, e]           # (n_x,)

        Lagrangian = ca.MX.zeros(1, 1)

        x_prop = x_k   # F_0(U, x(k)) = x(k)

        for i in range(e):
            # 이번 타임스텝의 제어 입력 (U_sym에서 슬라이스)
            u_i = U_sym[i * n_u : (i + 1) * n_u]   # (n_u,)

            # 이번 타임스텝의 M, CG (Xm_data_sym이 아닌 M_data_sym에서 슬라이스)
            M_i  = M_data_sym[:, i * nv : (i + 1) * nv]   # (nv, nv)
            CG_i = CG_data_sym[:, i]                         # (nv,)

            # 스테이지 비용 l(x_i, u_i; theta, ys_i)
            ys_i = ys_data_sym[:, i]   # (target_dim,)
            Lagrangian += l_fn(x_prop, u_i, theta_sym, ys_i)

            # λ_i^T C̄(x_i, u_i)
            # C̄(x_i, u_i) = P [x_i; u_i] - p  (num_total_constraints × 1)
            z_i = ca.vertcat(x_prop, u_i)   # (n_x + n_u,)
            C_i = self._compute_constraint_expr(z_i)   # (n_c, 1)

            # 이 타임스텝의 lambda_i (lambda_flat_sym에서 슬라이스)
            lam_i = lambda_flat_sym[i * n_c : (i + 1) * n_c]   # (n_c,)
            Lagrangian += ca.reshape(lam_i, 1, -1) @ C_i        # scalar

            # 동역학 전파: x_{i+1} = f(x_i, u_i, M_i, CG_i)
            x_prop = f_dyn(x_prop, u_i, M_i, CG_i)

        # 종단 상태 제약 항: ν^T (F_e(U, x(k)) - x(k+e))
        # F_e(U, x(k)) = x_prop (e번 전파 후)
        Lagrangian += ca.reshape(nu_sym, 1, -1) @ (x_prop - x_k_e)

        # ── 5. ∇_U L 심볼릭 미분 ──────────────────────────────────────
        grad_L_U = ca.gradient(Lagrangian, U_sym)   # (e * n_u,)

        # ── 6. 상보성 여유 조건 λ_i^T C̄(x_i^m, u_i^m) ───────────────
        # 주의: 라그랑지안의 x_prop는 U_sym으로 전파된 것이지만,
        # 상보성 여유 조건 계산에는 시연 데이터 Xm_data_sym를 사용합니다.
        # (논문 Eq.10b: 실제 관찰된 데이터에서 제약 만족 확인)
        csc_list = []
        for i in range(e):
            x_i_m = Xm_data_sym[:, i]                      # (n_x,)
            u_i_m = Um_data_sym[:, i]                       # (n_u,)
            z_i_m = ca.vertcat(x_i_m, u_i_m)               # (n_x + n_u,)
            C_i_m = self._compute_constraint_expr(z_i_m)    # (n_c, 1)

            lam_i = lambda_flat_sym[i * n_c : (i + 1) * n_c]
            csc_i = ca.reshape(lam_i, 1, -1) @ C_i_m       # scalar (1, 1)
            csc_list.append(csc_i)

        # n_c == 0 인 경우 csc_i들이 "희소 0 스칼라"로 쌓일 수 있어
        # 이후 IPOPT가 제약벡터 g를 dense vector로 요구할 때 assert에 걸립니다.
        csc_expr = ca.densify(ca.vertcat(*csc_list))   # (e, 1)

        # ── 7. 공통 입력 인자 목록 ─────────────────────────────────────
        # Um_data_sym 대신 U_sym을 라그랑지안에 사용하므로,
        # Function 입력에서 U_sym과 Um_data_sym의 관계를 정리합니다.
        #
        # grad_L_U_func : U_sym이 최적화 변수 → 함수 입력으로 Um_data를 받아
        #                  U_sym = Um_data_flat 로 치환하여 수치 미분 수행
        #
        # 구현 전략: U_sym을 입력 인자에 포함시키고,
        # IOC Optimizer가 Um_data.flatten()을 U 위치에 전달합니다.

        common_inputs = [
            theta_sym,      # 0: (theta_dim,)
            nu_sym,         # 1: (state_dim,)
            lambda_flat_sym,# 2: (e * n_c,)
            U_sym,          # 3: (e * n_u,)  ← Um_data.flatten(order='F') 로 전달
            Xm_data_sym,    # 4: (state_dim, e+1)
            Um_data_sym,    # 5: (input_dim, e)  ← CSC에서만 사용
            ys_data_sym,    # 6: (target_dim, e)
            M_data_sym,     # 7: (nv, nv*e)
            CG_data_sym,    # 8: (nv, e)
        ]
        common_input_names = [
            'theta', 'nu', 'lambda_flat',
            'U', 'Xm_data', 'Um_data', 'ys_data',
            'M_data', 'CG_data'
        ]

        # ── 8. CasADi Function 생성 ────────────────────────────────────
        grad_L_U_func = ca.Function(
            'grad_L_U_func',
            common_inputs,
            [grad_L_U],
            common_input_names,
            ['gradient']
        )

        csc_func = ca.Function(
            'csc_func',
            common_inputs,
            [csc_expr],
            common_input_names,
            ['complementary_slackness']
        )

        # 캐시 저장
        self._grad_L_U_func = grad_L_U_func
        self._csc_func       = csc_func

        print(f"  [KKTBuilder] KKT 함수 빌드 완료")
        print(f"    grad_L_U 차원: ({e * n_u}, 1)")
        print(f"    csc 차원     : ({e}, 1)")

        return grad_L_U_func, csc_func

    def get_input_dims(self) -> Dict[str, Any]:
        """
        IOC Optimizer에서 변수 선언 시 필요한 차원 정보를 반환합니다.

        Returns:
            dict:
                theta_dim       : int
                nu_dim          : int = state_dim
                lambda_flat_dim : int = e * num_total_constraints
                U_dim           : int = e * input_dim
                grad_dim        : int = e * input_dim
                csc_dim         : int = e
        """
        return {
            'theta_dim'       : self.theta_dim,
            'nu_dim'          : self.state_dim,
            'lambda_flat_dim' : self.e * self.num_total_constraints,
            'U_dim'           : self.e * self.input_dim,
            'grad_dim'        : self.e * self.input_dim,
            'csc_dim'         : self.e,
        }

    @staticmethod
    def prepare_M_data(M_list: List[np.ndarray]) -> np.ndarray:
        """
        (e, nv, nv) 형태의 M 행렬 리스트를 CasADi Function 입력용
        (nv, nv * e) 형태로 변환합니다.

        Args:
            M_list : e개의 (nv, nv) 질량 행렬

        Returns:
            np.ndarray: (nv, nv * e)
        """
        return np.hstack(M_list)   # (nv, nv*e)

    @staticmethod
    def prepare_CG_data(CG_list: List[np.ndarray]) -> np.ndarray:
        """
        (e, nv) 또는 (e, nv, 1) 형태의 CG 리스트를 CasADi Function 입력용
        (nv, e) 형태로 변환합니다.

        Args:
            CG_list : e개의 (nv,) 또는 (nv, 1) 바이어스 벡터

        Returns:
            np.ndarray: (nv, e)
        """
        return np.column_stack([cg.flatten() for cg in CG_list])   # (nv, e)

    @staticmethod
    def prepare_U_flat(Um: np.ndarray) -> np.ndarray:
        """
        (e, input_dim) 형태의 입력 행렬을 CasADi Function의 U 인자용
        (e * input_dim,) 열벡터로 변환합니다.

        DataProcessor 출력은 행 = 타임스텝이므로,
        열 방향으로 flatten하여 [u_0; u_1; ...; u_{e-1}] 순서를 만듭니다.

        Args:
            Um : (e, input_dim)

        Returns:
            np.ndarray: (e * input_dim,)
        """
        # Um[i, :] = u_i  →  [u_0; u_1; ...; u_{e-1}]
        return Um.flatten(order='C')   # row-major: u_0 먼저

    # ================================================================
    # 내부 헬퍼
    # ================================================================

    def _compute_constraint_expr(self, z_sym: ca.MX) -> ca.MX:
        """
        z = [x; u] 에 대해 모든 후보 제약 조건의 C̄(z) = Pz - p 를
        하나의 열벡터 (num_total_constraints, 1)로 이어 붙여 반환합니다.

        Args:
            z_sym : ca.MX  (state_dim + input_dim, 1)

        Returns:
            ca.MX: (num_total_constraints, 1)
        """
        if not self.cons:
            return ca.MX.zeros(0, 1)

        parts = []
        for pc in self.cons:
            # base_constraints.py의 get_casadi_expr: P @ z - p → (num_ineq, 1)
            parts.append(pc.get_casadi_expr(z_sym))

        return ca.vertcat(*parts)   # (num_total_constraints, 1)


# ====================================================================
# 동작 확인 (직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost

    # ── 더미 DynamicsModel (MuJoCo 없이) ─────────────────────────────
    class DummyDynamics:
        """선형 동역학 f(x,u,M,CG) ≈ x + A*x*dt + B*u*dt"""
        state_dim = 6
        input_dim = 2
        nq        = 4
        nv        = 2

        def __init__(self):
            np.random.seed(1)
            self._A = np.eye(self.state_dim) * 0.99
            self._B = np.random.randn(self.state_dim, self.input_dim) * 0.01
            x_s  = ca.MX.sym('x',  self.state_dim)
            u_s  = ca.MX.sym('u',  self.input_dim)
            M_s  = ca.MX.sym('M',  self.nv, self.nv)
            CG_s = ca.MX.sym('CG', self.nv, 1)
            x_n  = ca.MX(self._A) @ x_s + ca.MX(self._B) @ u_s
            self._f = ca.Function('f', [x_s, u_s, M_s, CG_s], [x_n],
                                  ['x', 'u', 'M', 'CG'], ['x_next'])

        def get_casadi_func(self): return self._f

    # ── 설정 ──────────────────────────────────────────────────────────
    dyn = DummyDynamics()
    cost_cfg = {'target_state_selection': {'indices': [0, 1]}}
    cost = ParametricStageCost(dyn.state_dim, dyn.input_dim, cost_cfg)

    # ── 더미 제약 조건 (u 박스) ───────────────────────────────────────
    z_dim = dyn.state_dim + dyn.input_dim   # 8
    P_ub = np.zeros((dyn.input_dim, z_dim))
    P_ub[:, dyn.state_dim:] = np.eye(dyn.input_dim)
    P_lb = -P_ub.copy()
    cons = [
        PolytopeConstraint(P_ub, np.ones((dyn.input_dim, 1)) * 10.0, "u upper"),
        PolytopeConstraint(P_lb, np.ones((dyn.input_dim, 1)) * 10.0, "u lower"),
    ]

    # ── KKTBuilder ────────────────────────────────────────────────────
    e = 5
    kkt = KKTBuilder(dyn, cost, cons, e=e)

    grad_func, csc_func = kkt.build_kkt_residual_functions()
    dims = kkt.get_input_dims()
    print(f"\n차원 정보: {dims}")

    # ── 더미 데이터 ───────────────────────────────────────────────────
    np.random.seed(42)
    sd, ud, nv = dyn.state_dim, dyn.input_dim, dyn.nv
    td = cost.target_dim

    Xm = np.random.randn(sd, e + 1) * 0.1   # (state_dim, e+1)
    Um = np.random.randn(ud, e)    * 0.5     # (input_dim, e)
    ys = np.random.randn(td, e)    * 0.1     # (target_dim, e)

    M_list  = [np.eye(nv)] * e
    CG_list = [np.zeros(nv)] * e
    M_data  = KKTBuilder.prepare_M_data(M_list)    # (nv, nv*e)
    CG_data = KKTBuilder.prepare_CG_data(CG_list)  # (nv, e)

    theta0 = cost.theta_init(scale=0.1)
    nu0    = np.zeros(sd)
    lam0   = np.zeros(e * kkt.num_total_constraints)
    U_flat = KKTBuilder.prepare_U_flat(Um.T)  # (e, input_dim) → flatten

    # ── 함수 호출 ────────────────────────────────────────────────────
    kkt_inputs = [theta0, nu0, lam0, U_flat, Xm, Um, ys, M_data, CG_data]

    grad_val = np.array(grad_func(*kkt_inputs)).flatten()
    csc_val  = np.array(csc_func(*kkt_inputs)).flatten()

    print(f"\n∇_U L 형태 : {grad_val.shape}  (기대: ({e * ud},))")
    print(f"∇_U L norm : {np.linalg.norm(grad_val):.6f}")
    print(f"∇_U L 처음 5개: {grad_val[:5].round(6)}")

    print(f"\nCSC 형태   : {csc_val.shape}  (기대: ({e},))")
    print(f"CSC 값     : {csc_val.round(6)}")

    # ── 형상 검증 ────────────────────────────────────────────────────
    assert grad_val.shape == (e * ud,), f"∇_U L 형상 오류: {grad_val.shape}"
    assert csc_val.shape  == (e,),      f"CSC 형상 오류: {csc_val.shape}"
    print("\n✅ KKTBuilder 모든 테스트 통과")
