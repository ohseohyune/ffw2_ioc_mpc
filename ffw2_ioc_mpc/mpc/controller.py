"""
src/mpc/controller.py

IOC 학습 결과(θ*, 식별된 제약 조건)를 MPC 최적화 문제에 통합하는 컨트롤러.

────────────────────────────────────────────────────────────
논문 Eq.(3) MPC 최적화 문제:
    min_{u_0,...,u_{N-1}}  Σ_{i=0}^{N-1} l(x_i, u_i; θ*)

    s.t.
        x_{i+1} = f(x_i, u_i, M_i, CG_i)   i = 0,...,N-1
        P_j x_i + Q_j u_i ≤ q_j              j = 1,...,n_c  (식별된 제약 조건)
        x_0 = x(k)                            (현재 상태 초기 조건)

────────────────────────────────────────────────────────────
설계 원칙:
    - 학습된 θ*는 `set_learned_parameters()` 호출 시 NLP에 수치값으로 고정
    - M, CG (질량 행렬 / 바이어스 힘)는 매 호출마다 런타임에 파라미터로 주입
    - 제약 조건은 PolytopeConstraint.get_casadi_expr(z) 를 통해 CasADi 심볼로 변환
    - 핫스타트(warm start): 이전 해를 초기 추정값으로 사용하여 수렴 가속

────────────────────────────────────────────────────────────
파라미터 팩킹 순서:
    p = [x0 (state_dim,)
         | ys (target_dim,)
         | M_flat (nv * nv * N_p,)   ← 각 스텝 M을 column-major flatten
         | CG_flat (nv * N_p,)]
"""

import time
import warnings
import numpy as np
import casadi as ca
from typing import List, Dict, Any, Tuple, Optional

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


class MPCController:
    """
    IOC 학습 결과를 활용하는 모델 예측 컨트롤러.

    사용 흐름:
        1. MPCController(dynamics, stage_cost, identified_constraints, config) 생성
        2. set_learned_parameters(theta_star) → NLP 빌드 & 솔버 생성
        3. 루프에서: optimal_u, info = get_control_action(x, ys, M_data, CG_data)
        4. optimal_u[:input_dim] 을 로봇에 적용

    Notes:
        - 비선형 동역학(MuJoCo 기반) + 비선형 비용 → IPOPT 사용 권장
        - 실시간성: N_p × e_step 수가 클수록 계산 시간 증가
        - M, CG는 MPC 예측 구간 N_p 전체에 대해 필요하나,
          실용적으로 현재 스텝 값을 N_p번 반복하는 것도 허용됨
    """

    def __init__(
        self,
        dynamics_model        : DynamicsModel,
        stage_cost            : ParametricStageCost,
        identified_constraints: List[PolytopeConstraint],
        mpc_config            : Dict[str, Any],
    ):
        """
        Args:
            dynamics_model (DynamicsModel):
                get_casadi_func() → f(x, u, M, CG) → x_next

            stage_cost (ParametricStageCost):
                get_casadi_function() → l(x, u, theta, ys) → scalar
                get_Q_R_functions()   → Q_func, R_func

            identified_constraints (List[PolytopeConstraint]):
                ConstraintIdentifier.identify_active_constraints() 출력.
                빈 리스트이면 제약 없는 MPC로 동작합니다.

            mpc_config (Dict[str, Any]):
                지원 키:
                    prediction_horizon   : int    N_p (기본: 20)
                    control_horizon      : int    N_c (기본: 1, N_c ≤ N_p)
                    solver_name          : str    'ipopt' (기본)
                    ipopt_options        : dict   IPOPT 솔버 옵션
                        print_level      : int    (기본: 0)
                        max_iter         : int    (기본: 100)
                        tol              : float  (기본: 1e-6)
                    enable_terminal_cost : bool   종단 비용 추가 여부 (기본: False)
                    terminal_cost_weight : float  종단 비용 가중치 (기본: 1.0)
        """
        self.dyn   = dynamics_model
        self.cost  = stage_cost
        self.cons  = identified_constraints
        self.cfg   = mpc_config

        self.state_dim  = dynamics_model.state_dim    # 75
        self.input_dim  = dynamics_model.input_dim    # 14
        self.nv         = dynamics_model.nv           # 37
        self.target_dim = stage_cost.target_dim       # 2 (예)

        self.N_p = mpc_config.get('prediction_horizon', 20)
        self.N_c = min(mpc_config.get('control_horizon', 1), self.N_p)

        # 학습된 θ* (set_learned_parameters 호출 전까지 None)
        self.learned_theta: Optional[np.ndarray] = None     

        # CasADi NLP 솔버 (set_learned_parameters → _build_nlp 호출 후 생성)
        self._solver            : Optional[ca.Function] = None
        self._lbg               : Optional[ca.DM]       = None
        self._ubg               : Optional[ca.DM]       = None
        self._n_g               : int                    = 0

        # 파라미터 벡터 차원 (빌드 후 설정)
        self._p_dim : int = 0

        # 핫스타트용 이전 해 저장
        self._prev_u_sol: Optional[np.ndarray] = None

        print("=" * 60)
        print("MPCController 초기화 완료")
        print(f"  예측 구간 N_p               : {self.N_p}")
        print(f"  제어 구간 N_c               : {self.N_c}")
        print(f"  최적화 변수 dim             : {self.input_dim * self.N_c}")
        print(f"  솔버                        : {self.cfg.get('solver_name', 'ipopt')}")
        print(f"  식별된 제약 조건            : {len(self.cons)}개 "
              f"({sum(pc.num_ineq for pc in self.cons)}개 부등식)")
        print(f"  종단 비용                   : {self.cfg.get('enable_terminal_cost', False)}")
        print("=" * 60)

    # ================================================================
    # 공개 인터페이스: 학습된 파라미터 설정 및 NLP 빌드
    # ================================================================

    def set_learned_parameters(self, learned_theta: np.ndarray) -> None:
        """
        IOC 학습된 비용 파라미터 θ*를 설정하고 MPC NLP를 빌드합니다.

        이 메서드를 호출한 후에야 get_control_action()을 사용할 수 있습니다.

        Args:
            learned_theta (np.ndarray): shape (theta_dim,)
        """
        if learned_theta.shape[0] != self.cost.theta_dim:
            raise ValueError(
                f"learned_theta 차원({learned_theta.shape[0]}) ≠ "
                f"theta_dim({self.cost.theta_dim})"
            )

        self.learned_theta = learned_theta.copy()
        print(f"\nMPCController: θ* 설정 완료  shape={learned_theta.shape}")
        print(f"  θ*[:5] = {learned_theta[:5].round(6)}")

        # NLP 빌드
        self._build_nlp() # NLP를 새로 구성해. 왜냐면 θ가 바뀌면 Q,R이 바뀌니까.

    # ================================================================
    # 공개 인터페이스: MPC 최적화 실행
    # ================================================================

    def get_control_action(
        self,
        current_state : np.ndarray,
        target_ys     : np.ndarray,
        M_data_mpc    : np.ndarray,
        CG_data_mpc   : np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        현재 상태에 대한 최적 제어 입력을 계산합니다.

        Args:
            current_state (np.ndarray):
                현재 시스템 상태. shape: (state_dim,)

            target_ys (np.ndarray):
                MPC가 추적할 목표. shape: (target_dim,)

            M_data_mpc (np.ndarray):
                예측 구간 N_p에 대한 질량 행렬.
                shape: (N_p, nv, nv)

                * 실용적 근사: 현재 스텝 M을 N_p번 복사
                  M_data_mpc = np.tile(M_curr, (N_p, 1, 1))

            CG_data_mpc (np.ndarray):
                예측 구간 N_p에 대한 바이어스 힘.
                shape: (N_p, nv) 또는 (N_p, nv, 1)

        Returns:
            optimal_u (np.ndarray):
                최적 제어 입력의 첫 번째 스텝. shape: (input_dim,)
                실제 로봇에 적용할 토크 명령.

            solver_info (Dict[str, Any]):
                'f'             : float   최종 목적 함수 값
                'success'       : bool    수렴 성공 여부
                'return_status' : str     IPOPT 상태 문자열
                'solve_time_ms' : float   풀이 소요 시간 (ms)
                'g_max'         : float   최대 제약 위반량 (≤0이 이상적)

        Raises:
            RuntimeError: set_learned_parameters()가 호출되지 않았을 때.
        """
        if self._solver is None:
            raise RuntimeError(
                "MPC 솔버가 준비되지 않았습니다. "
                "set_learned_parameters()를 먼저 호출하세요."
            )

        # ── 입력 검증 ───────────────────────────────────────────────
        current_state  = np.asarray(current_state,  dtype=float).flatten()
        target_ys      = np.asarray(target_ys,      dtype=float).flatten()
        M_data_mpc     = np.asarray(M_data_mpc,     dtype=float)
        CG_data_mpc    = np.asarray(CG_data_mpc,    dtype=float)

        if current_state.shape[0] != self.state_dim:
            raise ValueError(f"current_state 차원 오류: {current_state.shape}")
        if target_ys.shape[0] != self.target_dim:
            raise ValueError(f"target_ys 차원 오류: {target_ys.shape}")

        # M_data_mpc: (N_p, nv, nv) 보장
        if M_data_mpc.ndim == 2:
            M_data_mpc = np.tile(M_data_mpc, (self.N_p, 1, 1))
        elif M_data_mpc.ndim == 3 and M_data_mpc.shape[0] != self.N_p:
            raise ValueError(f"M_data_mpc.shape[0]={M_data_mpc.shape[0]} ≠ N_p={self.N_p}")

        # CG_data_mpc: (N_p, nv) 보장
        CG_data_mpc = CG_data_mpc.reshape(self.N_p, -1)
        if CG_data_mpc.shape[1] != self.nv:
            raise ValueError(f"CG_data_mpc.shape[1]={CG_data_mpc.shape[1]} ≠ nv={self.nv}")

        # ── 파라미터 팩킹 ────────────────────────────────────────────
        # p = [x0 | ys | M_flat (N_p * nv * nv,) | CG_flat (N_p * nv,)]
        M_flat  = M_data_mpc.flatten(order='C')    # (N_p * nv * nv,)
        CG_flat = CG_data_mpc.flatten(order='C')   # (N_p * nv,)

        p_val = np.concatenate([current_state, target_ys, M_flat, CG_flat])

        # ── 초기 추정값 (핫스타트) ───────────────────────────────────
        u_init = self._get_initial_guess()

        # ── MPC 풀이 ─────────────────────────────────────────────────
        t_start = time.perf_counter()
        sol = self._solver(
            x0   = u_init,
            p    = p_val,
            lbg  = self._lbg,
            ubg  = self._ubg,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── 결과 파싱 ────────────────────────────────────────────────
        u_opt_flat = np.array(sol['x']).flatten()
        optimal_u  = u_opt_flat[:self.input_dim]   # 첫 번째 제어 스텝

        # 다음 호출을 위한 핫스타트 저장
        self._prev_u_sol = u_opt_flat.copy()

        stats = self._solver.stats()
        g_val = np.array(sol['g']).flatten() if self._n_g > 0 else np.array([])

        solver_info = {
            'f'            : float(sol['f']),
            'success'      : stats.get('success', False),
            'return_status': stats.get('return_status', ''),
            'solve_time_ms': elapsed_ms,
            'g_max'        : float(g_val.max()) if g_val.size > 0 else 0.0,
        }   

        return optimal_u, solver_info

    # ================================================================
    # NLP 빌드
    # ================================================================

    def _build_nlp(self) -> None:
        """
        CasADi NLP 문제를 구성하고 IPOPT 솔버를 생성합니다.

        최적화 변수: U_ctrl = [u_0; u_1; ...; u_{N_c-1}]  shape: (N_c * input_dim,)
        파라미터:    p = [x0 | ys | M_flat | CG_flat]
        """
        print("\n[MPCController] NLP 빌드 중...")

        n_x   = self.state_dim
        n_u   = self.input_dim
        nv    = self.nv
        N_p   = self.N_p
        N_c   = self.N_c
        n_td  = self.target_dim

        # ── 심볼릭 최적화 변수 ────────────────────────────────────────
        # U_ctrl : 제어 구간 N_c개의 입력 (N_c * n_u,)
        U_ctrl_sym = ca.MX.sym('U_ctrl', n_u * N_c) # 최적화 변수

        # ── 심볼릭 파라미터 ───────────────────────────────────────────
        x0_sym  = ca.MX.sym('x0',  n_x)
        ys_sym  = ca.MX.sym('ys',  n_td)
        M_sym   = ca.MX.sym('M',   nv * nv * N_p)    # (N_p * nv * nv,)
        CG_sym  = ca.MX.sym('CG',  nv * N_p)         # (N_p * nv,)
        p_sym   = ca.vertcat(x0_sym, ys_sym, M_sym, CG_sym)

        self._p_dim = int(p_sym.shape[0])

        # ── CasADi 함수 참조 ──────────────────────────────────────────
        f_dyn = self.dyn.get_casadi_func()    # f(x, u, M, CG) → x_next
        l_fn  = self.cost.get_casadi_function()  # l(x, u, theta, ys) → scalar
        theta_dm = ca.DM(self.learned_theta)  # 고정된 수치값

        # ── 목적 함수 구성 ────────────────────────────────────────────
        objective = ca.MX.zeros(1, 1)
        g_list    = []   # 부등식 제약: expr <= 0

        x_curr = x0_sym  # 현재 상태에서 시작

        for i in range(N_p):
            # 제어 구간 고려: i >= N_c 이면 마지막 제어 입력 반복
            ctrl_idx = min(i, N_c - 1)
            u_i = U_ctrl_sym[ctrl_idx * n_u : (ctrl_idx + 1) * n_u]

            # 이번 스텝의 M, CG 추출 (파라미터 벡터에서 슬라이스)
            M_i_flat = M_sym[i * nv * nv : (i + 1) * nv * nv]
            M_i      = ca.reshape(M_i_flat, nv, nv)
            CG_i     = CG_sym[i * nv : (i + 1) * nv]

            # 스테이지 비용 l(x_i, u_i; θ*, ys)
            objective += l_fn(x_curr, u_i, theta_dm, ys_sym)

            # 식별된 제약 조건 C(z_i) = P[x_i;u_i] - p ≤ 0
            if self.cons:
                z_i = ca.vertcat(x_curr, u_i)   # (state_dim + input_dim,)
                for pc in self.cons:
                    g_list.append(pc.get_casadi_expr(z_i))   # (num_ineq, 1)

            # 동역학 전파
            x_curr = f_dyn(x_curr, u_i, M_i, CG_i)

        # 종단 비용 (선택)
        if self.cfg.get('enable_terminal_cost', False):
            w_term = float(self.cfg.get('terminal_cost_weight', 1.0))
            Q_func, _ = self.cost.get_Q_R_functions()
            Q_term     = Q_func(theta_dm)   # (target_dim, target_dim) DM

            # S * x_term : 상태에서 target 인덱스만 선택
            x_selected = ca.vertcat(*[
                x_curr[idx] for idx in self.cost.target_state_indices
            ])
            err_term   = x_selected - ys_sym
            objective += w_term * (err_term.T @ Q_term @ err_term)

        # ── 제약 조건 합치기 ──────────────────────────────────────────
        if g_list:
            g_expr  = ca.vertcat(*g_list)   # (total_ineq * N_p,)
            self._n_g = int(g_expr.shape[0])
            lbg = -np.inf * np.ones(self._n_g)
            ubg = np.zeros(self._n_g)       # P z - p <= 0
        else:
            g_expr    = ca.MX.zeros(0, 1)
            self._n_g = 0
            lbg = np.zeros(0)
            ubg = np.zeros(0)

        # ── NLP 정의 & 솔버 생성 ─────────────────────────────────────
        nlp = {
            'x': U_ctrl_sym,
            'f': objective,
            'p': p_sym,
            'g': g_expr,
        }

        ipopt_user = self.cfg.get('ipopt_options', {})
        solver_opts = {
            'ipopt.print_level' : ipopt_user.get('print_level', 0),
            'ipopt.max_iter'    : ipopt_user.get('max_iter',    100),
            'ipopt.tol'         : ipopt_user.get('tol',         1e-6),
            'ipopt.sb'          : 'yes',
            'print_time'        : 0,
        }

        solver_name = self.cfg.get('solver_name', 'ipopt')
        self._solver = ca.nlpsol('mpc_solver', solver_name, nlp, solver_opts)
        self._lbg    = ca.DM(lbg)
        self._ubg    = ca.DM(ubg)

        n_opt = n_u * N_c
        print(f"  NLP 빌드 완료")
        print(f"    최적화 변수 : {n_opt}  (N_c={N_c}, n_u={n_u})")
        print(f"    파라미터 dim: {self._p_dim}")
        print(f"    부등식 수   : {self._n_g}  "
              f"({len(self.cons)}개 PC × {N_p} steps)")

    # ================================================================
    # 유틸리티
    # ================================================================

    def _get_initial_guess(self) -> np.ndarray:
        """
        핫스타트: 이전 해의 첫 번째 제어 스텝을 제거하고 나머지를 앞으로 이동.
        이전 해가 없으면 0으로 초기화.
        """
        n_opt = self.input_dim * self.N_c

        if self._prev_u_sol is None:
            return np.zeros(n_opt)

        # 핫스타트: shift by one control step
        prev = self._prev_u_sol
        if len(prev) < n_opt:
            return np.zeros(n_opt)

        u_init = np.zeros(n_opt)
        if self.N_c > 1:
            # u_1,...,u_{N_c-1}, u_{N_c-1} (마지막 반복)
            u_init[: (self.N_c - 1) * self.input_dim] = \
                prev[self.input_dim : self.N_c * self.input_dim]
            u_init[(self.N_c - 1) * self.input_dim :] = \
                prev[(self.N_c - 1) * self.input_dim : self.N_c * self.input_dim]
        else:
            u_init = prev[:n_opt].copy()

        return u_init

    def reset_warm_start(self) -> None:
        """핫스타트 이력을 초기화합니다 (새 태스크 시작 시 사용)."""
        self._prev_u_sol = None
        print("[MPCController] 핫스타트 이력 초기화")

    def predict_trajectory(
        self,
        current_state: np.ndarray,
        M_data_mpc   : np.ndarray,
        CG_data_mpc  : np.ndarray,
        u_sequence   : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        주어진 제어 시퀀스로 예측 궤적을 계산합니다 (디버깅 / 시각화용).

        Args:
            current_state : (state_dim,)
            M_data_mpc    : (N_p, nv, nv)
            CG_data_mpc   : (N_p, nv)
            u_sequence    : (N_p, input_dim) 또는 None (이전 해 사용)

        Returns:
            np.ndarray: (N_p + 1, state_dim)  예측 상태 궤적
        """
        if u_sequence is None:
            if self._prev_u_sol is None:
                u_sequence = np.zeros((self.N_p, self.input_dim))
            else:
                u_flat = self._prev_u_sol
                u_sequence = np.zeros((self.N_p, self.input_dim))
                for i in range(self.N_p):
                    ctrl_idx = min(i, self.N_c - 1)
                    u_sequence[i] = u_flat[ctrl_idx * self.input_dim:
                                           (ctrl_idx + 1) * self.input_dim]

        f_dyn = self.dyn.get_casadi_func()
        M_data_mpc  = np.asarray(M_data_mpc).reshape(self.N_p, self.nv, self.nv)
        CG_data_mpc = np.asarray(CG_data_mpc).reshape(self.N_p, self.nv)

        traj = [current_state.copy()]
        x_curr = current_state.copy()

        for i in range(self.N_p):
            x_next = np.array(
                f_dyn(x_curr, u_sequence[i], M_data_mpc[i], CG_data_mpc[i])
            ).flatten()
            traj.append(x_next)
            x_curr = x_next

        return np.array(traj)   # (N_p + 1, state_dim)

    def get_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습된 θ*로부터 Q, R 행렬을 반환합니다 (분석용).

        Returns:
            Q : (target_dim, target_dim)
            R : (input_dim, input_dim)

        Raises:
            RuntimeError: set_learned_parameters()가 호출되지 않았을 때.
        """
        if self.learned_theta is None:
            raise RuntimeError("set_learned_parameters()를 먼저 호출하세요.")
        Q_func, R_func = self.cost.get_Q_R_functions()
        Q = np.array(Q_func(self.learned_theta))
        R = np.array(R_func(self.learned_theta))
        return Q, R


# ====================================================================
# 동작 확인 (직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    import casadi as ca
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost

    # ── 더미 DynamicsModel ───────────────────────────────────────────
    class DummyDyn:
        state_dim = 4
        input_dim = 2
        nv        = 2

        def __init__(self):
            np.random.seed(3)
            A = np.eye(self.state_dim) * 0.98
            B = np.random.randn(self.state_dim, self.input_dim) * 0.05
            xs = ca.MX.sym('x', self.state_dim)
            us = ca.MX.sym('u', self.input_dim)
            Ms = ca.MX.sym('M', self.nv, self.nv)
            Gs = ca.MX.sym('G', self.nv, 1)
            xn = ca.MX(A) @ xs + ca.MX(B) @ us
            self._f = ca.Function('f', [xs, us, Ms, Gs], [xn],
                                  ['x', 'u', 'M', 'CG'], ['x_next'])

        def get_casadi_func(self):
            return self._f

    dyn = DummyDyn()
    cost_cfg = {'target_state_selection': {'indices': [0, 1]}}
    cost = ParametricStageCost(dyn.state_dim, dyn.input_dim, cost_cfg)

    # ── 더미 제약 조건 (u 박스) ───────────────────────────────────────
    z_dim = dyn.state_dim + dyn.input_dim   # 6
    P_ub  = np.zeros((dyn.input_dim, z_dim))
    P_ub[:, dyn.state_dim:] = np.eye(dyn.input_dim)
    identified_cons = [
        PolytopeConstraint(P_ub,  np.ones((dyn.input_dim, 1)) * 5.0, "u upper"),
        PolytopeConstraint(-P_ub, np.ones((dyn.input_dim, 1)) * 5.0, "u lower"),
    ]

    # ── MPC 설정 ─────────────────────────────────────────────────────
    mpc_cfg = {
        'prediction_horizon' : 5,
        'control_horizon'    : 1,
        'solver_name'        : 'ipopt',
        'ipopt_options'      : {'print_level': 0, 'max_iter': 200, 'tol': 1e-6},
        'enable_terminal_cost': True,
        'terminal_cost_weight': 2.0,
    }

    controller = MPCController(dyn, cost, identified_cons, mpc_cfg)

    # ── 학습된 θ* 설정 (더미) ────────────────────────────────────────
    theta_star = cost.theta_init(scale=0.3)
    # R trace ≈ 1 맞추기 (간단히 스케일)
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost as PSC
    _, R_fn = cost.get_Q_R_functions()
    R_cur   = np.array(R_fn(theta_star))
    scale   = 1.0 / max(float(np.trace(R_cur)), 1e-6)
    theta_star *= np.sqrt(scale)
    controller.set_learned_parameters(theta_star)

    # ── 더미 M, CG 데이터 ────────────────────────────────────────────
    N_p = mpc_cfg['prediction_horizon']
    nv  = dyn.nv

    M_data  = np.tile(np.eye(nv), (N_p, 1, 1))      # (N_p, nv, nv)
    CG_data = np.zeros((N_p, nv))                     # (N_p, nv)

    # ── 제어 동작 계산 ────────────────────────────────────────────────
    x0 = np.array([0.5, -0.3, 0.1, 0.0])
    ys = np.array([0.0, 0.0])

    optimal_u, info = controller.get_control_action(x0, ys, M_data, CG_data)

    print(f"\n[결과]")
    print(f"  최적 u (첫 스텝) : {optimal_u.round(6)}")
    print(f"  목적 함수 f      : {info['f']:.4e}")
    print(f"  성공 여부        : {info['success']}")
    print(f"  풀이 시간        : {info['solve_time_ms']:.1f} ms")
    print(f"  최대 제약 위반   : {info['g_max']:.4e}  (≤ 0이 이상적)")

    assert optimal_u.shape == (dyn.input_dim,), f"u 형상 오류: {optimal_u.shape}"
    assert info['success'], f"MPC 수렴 실패: {info['return_status']}"
    print(f"\n  제약 조건 만족 : {'✅' if info['g_max'] <= 1e-4 else '⚠️'}")

    # ── 예측 궤적 ────────────────────────────────────────────────────
    traj = controller.predict_trajectory(x0, M_data, CG_data)
    print(f"\n  예측 궤적 shape: {traj.shape}  (기대: ({N_p+1}, {dyn.state_dim}))")
    print(f"  초기 상태  : {traj[0].round(4)}")
    print(f"  최종 상태  : {traj[-1].round(4)}")

    # ── Q, R 행렬 확인 ────────────────────────────────────────────────
    Q, R = controller.get_cost_matrices()
    print(f"\n  Q:\n{Q.round(4)}")
    print(f"  R (trace={np.trace(R):.4f}):\n{np.diag(R).round(4)}")

    # ── 핫스타트 테스트 ──────────────────────────────────────────────
    optimal_u2, info2 = controller.get_control_action(x0 * 0.9, ys, M_data, CG_data)
    print(f"\n  [핫스타트] 2번째 호출 풀이 시간: {info2['solve_time_ms']:.1f} ms  "
          f"(1번째: {info['solve_time_ms']:.1f} ms)")

    print("\n✅ MPCController 모든 테스트 통과")