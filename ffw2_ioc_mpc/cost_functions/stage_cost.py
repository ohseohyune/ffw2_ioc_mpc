"""
src/cost_functions/stage_cost.py

논문 Section II에 정의된 매개변수화된 스테이지 비용 함수를 CasADi로 구현합니다.
    l(x, u; L) = (S*x - y_s)^T Q (S*x - y_s) + u^T R u

Q = L_Q @ L_Q^T, R = L_R @ L_R^T 형태로 양의 준정부호성(PSD)을 보장합니다.
"""

import casadi as ca
import numpy as np
from typing import List


class ParametricStageCost:
    """
    학습 가능한 매개변수 theta를 사용하는 스테이지 비용 함수.

    theta 벡터 구성:
        - theta[0 : num_elements_LQ]                → L_Q의 하삼각 요소 (row-major)
        - theta[num_elements_LQ : theta_dim]         → L_R의 하삼각 요소 (row-major)

    양의 준정부호성 보장:
        Q = L_Q @ L_Q^T,  R = L_R @ L_R^T

    스케일링 정규화 조건 (sum(R_ii) == 1) 은 이 클래스 외부에서
    IOC 최적화 단계의 제약 조건으로 추가해야 합니다.
    """

    def __init__(self, state_dim: int, input_dim: int, cost_params_config: dict):
        """
        Args:
            state_dim (int):
                시스템 상태 벡터 x의 차원 (n).
                dynamics.py의 DynamicsModel.state_dim 값을 전달하세요.
            input_dim (int):
                시스템 입력 벡터 u의 차원 (m).
                dynamics.py의 DynamicsModel.input_dim 값을 전달하세요. (기본 14)
            cost_params_config (dict):
                configs/cost_params.yaml에서 로드된 딕셔너리.
                반드시 'target_state_selection.indices' 키를 포함해야 합니다.

                예시 cost_params.yaml:
                    target_state_selection:
                        indices: [20, 21]   # x 내 arm_l_joint1, arm_l_joint2 위치 (0-indexed)
                        description: "left arm joints 1 and 2 positions"
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.cost_params_config = cost_params_config

        # --- S 행렬 구성에 사용할 타겟 인덱스 파싱 ---
        target_cfg = cost_params_config.get("target_state_selection", {})
        self.target_state_indices: List[int] = target_cfg.get("indices", [])
        if not self.target_state_indices:
            raise ValueError(
                "cost_params_config에 'target_state_selection.indices' 가 비어있거나 없습니다. "
                "configs/cost_params.yaml을 확인하세요."
            )

        self.target_dim: int = len(self.target_state_indices)  # y_s 차원 (논문에서 2)

        # --- theta 차원 계산 ---
        # L_Q : target_dim × target_dim 하삼각 행렬  →  target_dim*(target_dim+1)/2 개 파라미터
        # L_R : input_dim  × input_dim  하삼각 행렬  →  input_dim *(input_dim +1)/2 개 파라미터
        self.num_elements_LQ: int = self.target_dim * (self.target_dim + 1) // 2
        self.num_elements_LR: int = self.input_dim  * (self.input_dim  + 1) // 2
        self.theta_dim:       int = self.num_elements_LQ + self.num_elements_LR

        print("=" * 60)
        print("ParametricStageCost 초기화 완료")
        print(f"  state_dim  : {self.state_dim}")
        print(f"  input_dim  : {self.input_dim}")
        print(f"  target_dim : {self.target_dim}  (y_s 차원)")
        print(f"  target_state_indices : {self.target_state_indices}")
        print(f"  theta_dim  : {self.theta_dim}")
        print(f"    ├── L_Q 요소 수 : {self.num_elements_LQ}")
        print(f"    └── L_R 요소 수 : {self.num_elements_LR}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 내부 헬퍼: 하삼각 행렬 구성
    # ------------------------------------------------------------------
    @staticmethod
    def _build_lower_triangular(flat_params: ca.MX, dim: int) -> ca.MX:
        """
        flat_params (dim*(dim+1)/2 개의 심볼) 로부터
        dim×dim 하삼각 행렬(CasADi MX)을 반환합니다.

        저장 순서 (row-major, lower-triangular):
            L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], ...
        """
        L = ca.MX.zeros(dim, dim)
        flat_idx = 0
        for i in range(dim):
            for j in range(i + 1):   # j <= i  →  하삼각
                L[i, j] = flat_params[flat_idx]
                flat_idx += 1
        return L

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------
    def get_casadi_function(self) -> ca.Function:
        """
        스테이지 비용을 계산하는 CasADi Function을 반환합니다.

        시그니처:
            l_func(x, u, theta, ys) -> scalar cost

        입력:
            x     (state_dim,)  : 현재 상태
            u     (input_dim,)  : 현재 제어 입력
            theta (theta_dim,)  : 비용 파라미터 [L_Q 요소들 | L_R 요소들]
            ys    (target_dim,) : 목표 상태 (레퍼런스)

        출력:
            cost (scalar) : l(x, u; theta, ys)
        """
        # ── 심볼릭 변수 선언 ────────────────────────────────────────────
        x_sym     = ca.MX.sym('x',     self.state_dim)
        u_sym     = ca.MX.sym('u',     self.input_dim)
        theta_sym = ca.MX.sym('theta', self.theta_dim)
        ys_sym    = ca.MX.sym('ys',    self.target_dim)

        # ── 1. S*x : 상태에서 타겟 관절만 선택 ──────────────────────────
        # S ∈ R^{target_dim × state_dim},  S*x = x[target_state_indices]
        selected_x = ca.vertcat(*[x_sym[idx] for idx in self.target_state_indices])
        # shape: (target_dim, 1)

        # ── 2. theta 분해 → L_Q, L_R ────────────────────────────────────
        theta_LQ = theta_sym[: self.num_elements_LQ]
        theta_LR = theta_sym[self.num_elements_LQ: self.num_elements_LQ + self.num_elements_LR]

        LQ = self._build_lower_triangular(theta_LQ, self.target_dim)
        LR = self._build_lower_triangular(theta_LR, self.input_dim)

        # Q = L_Q @ L_Q^T  ≥ 0  (PSD 보장)
        # R = L_R @ L_R^T  ≥ 0  (PSD 보장)
        Q_mat = LQ @ LQ.T
        R_mat = LR @ LR.T

        # ── 3. 스테이지 비용 계산 ────────────────────────────────────────
        err = selected_x - ys_sym                                    # (target_dim, 1)
        state_cost   = err.T @ Q_mat @ err                           # scalar
        control_cost = u_sym.T @ R_mat @ u_sym                       # scalar
        cost_expr    = state_cost + control_cost                      # scalar

        # ── 4. CasADi Function 생성 ──────────────────────────────────────
        l_func = ca.Function(
            'stage_cost_func',
            [x_sym, u_sym, theta_sym, ys_sym],
            [cost_expr],
            ['x', 'u', 'theta', 'ys'],
            ['cost']
        )
        return l_func

    def get_Q_R_functions(self):
        """
        (선택 사항) Q 및 R 행렬 자체를 theta의 함수로 반환하는 CasADi Function 쌍.
        IOC 디버깅이나 학습된 파라미터 시각화에 유용합니다.

        반환:
            Q_func : theta -> Q (target_dim × target_dim)
            R_func : theta -> R (input_dim  × input_dim)
        """
        theta_sym = ca.MX.sym('theta', self.theta_dim)

        theta_LQ = theta_sym[: self.num_elements_LQ]
        theta_LR = theta_sym[self.num_elements_LQ: self.num_elements_LQ + self.num_elements_LR]

        LQ = self._build_lower_triangular(theta_LQ, self.target_dim)
        LR = self._build_lower_triangular(theta_LR, self.input_dim)

        Q_mat = LQ @ LQ.T
        R_mat = LR @ LR.T

        Q_func = ca.Function('Q_from_theta', [theta_sym], [Q_mat], ['theta'], ['Q'])
        R_func = ca.Function('R_from_theta', [theta_sym], [R_mat], ['theta'], ['R'])
        return Q_func, R_func

    def normalization_constraint_expr(self, theta_sym: ca.MX) -> ca.MX:
        """
        논문의 스케일링 정규화 조건을 CasADi 심볼 표현식으로 반환합니다.
            sum(diag(R)) == 1   →   g(theta) = sum(R_ii) - 1 == 0

        이 표현식은 IOC 최적화 단계에서 등식 제약 조건으로 추가하세요:
            opti.subject_to(stage_cost.normalization_constraint_expr(theta_var) == 0)

        Args:
            theta_sym (ca.MX): theta 심볼릭 변수 (theta_dim,)

        Returns:
            ca.MX: scalar 심볼 표현식  sum(R_ii) - 1
        """
        theta_LR = theta_sym[self.num_elements_LQ: self.num_elements_LQ + self.num_elements_LR]
        LR = self._build_lower_triangular(theta_LR, self.input_dim)
        R_mat = LR @ LR.T
        trace_R = ca.trace(R_mat)   # sum of diagonal elements
        return trace_R - 1.0

    def theta_init(self, scale: float = 0.1) -> np.ndarray:
        """
        theta의 초기값을 반환합니다 (작은 양의 값으로 초기화).
        L_Q, L_R의 대각 요소만 scale로 설정하고 나머지는 0으로 초기화합니다.

        Args:
            scale (float): 대각 요소 초기값

        Returns:
            np.ndarray: shape (theta_dim,)
        """
        theta_init = np.zeros(self.theta_dim)

        # L_Q 대각 요소 위치 계산 (row-major lower-triangular에서 대각 위치)
        flat_idx = 0
        for i in range(self.target_dim):
            for j in range(i + 1):
                if i == j:                               # 대각
                    theta_init[flat_idx] = scale
                flat_idx += 1

        # L_R 대각 요소 위치 계산
        flat_idx = self.num_elements_LQ
        for i in range(self.input_dim):
            for j in range(i + 1):
                if i == j:                               # 대각
                    theta_init[flat_idx] = scale
                flat_idx += 1

        return theta_init


# ---------------------------------------------------------------------------
# 간단한 동작 확인 (직접 실행 시)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, os

    # ── 최소한의 더미 설정으로 테스트 ──────────────────────────────────
    dummy_cost_params = {
        "target_state_selection": {
            "indices": [20, 21],          # 예: arm_l_joint1, arm_l_joint2의 qpos 인덱스
            "description": "left arm joints 1 and 2 (position part of state)"
        }
    }

    state_dim  = 75   # DynamicsModel.state_dim (nq=38, nv=37 → 75)
    input_dim  = 14   # DynamicsModel.input_dim

    psc = ParametricStageCost(
        state_dim=state_dim,
        input_dim=input_dim,
        cost_params_config=dummy_cost_params
    )

    # CasADi 함수 생성
    l_func = psc.get_casadi_function()
    print(f"\nCasADi Function 생성 성공: {l_func}")

    # 수치 테스트
    theta0 = psc.theta_init(scale=0.1)
    x_test = np.zeros(state_dim)
    u_test = np.ones(input_dim) * 0.5
    ys_test = np.array([0.3, -0.2])   # 목표 관절 각도

    cost_val = float(l_func(x_test, u_test, theta0, ys_test))
    print(f"\n수치 테스트:")
    print(f"  theta_init (처음 5개): {theta0[:5]}")
    print(f"  cost value           : {cost_val:.6f}")

    # Q, R 함수 테스트
    Q_func, R_func = psc.get_Q_R_functions()
    Q_val = np.array(Q_func(theta0))
    R_val = np.array(R_func(theta0))
    print(f"\n  Q 행렬:\n{Q_val}")
    print(f"  R 대각 합 (정규화 전): {np.trace(R_val):.4f}")

    # 정규화 제약 확인
    theta_sym_test = ca.MX.sym('theta', psc.theta_dim)
    norm_expr = psc.normalization_constraint_expr(theta_sym_test)
    norm_func = ca.Function('norm_check', [theta_sym_test], [norm_expr])
    print(f"  정규화 제약값 (0이어야 정상): {float(norm_func(theta0)):.4f}")
    print("\n✅ ParametricStageCost 모든 테스트 통과")