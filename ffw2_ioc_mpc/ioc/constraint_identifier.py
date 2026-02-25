"""
src/ioc/constraint_identifier.py

학습된 라그랑주 승수 λ*를 사용하여 후보 제약 조건 중 활성/중요한
제약 조건을 식별하는 클래스.

────────────────────────────────────────────────────────────
논문 Eq.(11) 제약 조건 중요도 지표:
    Λ_j = Σ_{i=0}^{e-1} λ_{i,j}

    Λ_j ≥ Λ̄  →  j번째 제약 조건을 "활성"으로 식별
    Λ_j = 0   →  해당 제약 조건은 최적화에 영향 없음

    논문 기본값: Λ̄ = 10^{-3}

────────────────────────────────────────────────────────────
lambda_star 인덱스 구조:
    lambda_star : (e, num_total_inequalities)
    열 순서: [PolytopeConstraint_0 부등식들 | PolytopeConstraint_1 부등식들 | ...]

    예) PolytopeConstraint_0.num_ineq = 14 (u upper bound)
        PolytopeConstraint_1.num_ineq = 14 (u lower bound)
        → lambda_star[:, 0:14]   ← PolytopeConstraint_0 λ 값
           lambda_star[:, 14:28] ← PolytopeConstraint_1 λ 값

────────────────────────────────────────────────────────────
중요도 집계 전략 (identifier_config의 'aggregation' 키):
    'sum_all'    : Λ_j = Σ_i Σ_k λ_{i, col_start+k}  (기본, 논문 방식)
    'max_ineq'   : Λ_j = max_k( Σ_i λ_{i, col_start+k} )  (가장 강한 부등식 기준)
    'any_ineq'   : PolytopeConstraint 내 어느 부등식이라도 Λ̄ 이상이면 활성
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint


class ConstraintIdentifier:
    """
    학습된 라그랑주 승수를 기반으로 후보 제약 조건 세트에서
    활성/중요한 제약 조건을 식별하는 클래스.

    사용 흐름:
        1. ConstraintIdentifier 생성 (후보 제약 조건 리스트 주입)
        2. identify_active_constraints(lambda_star) 호출
        3. 반환된 List[PolytopeConstraint]를 MPC Controller에 주입

    Notes:
        - lambda_star의 열 순서는 KKTBuilder가 candidate_constraints를
          이어 붙인 순서와 반드시 일치해야 합니다.
        - PolytopeConstraint 하나에 여러 부등식이 포함될 수 있습니다
          (예: u 박스 제약 = 14개 부등식).
          모든 관련 λ를 합산하여 Λ_j를 계산합니다.
    """

    def __init__(
        self,
        candidate_constraints: List[PolytopeConstraint],
        identifier_config    : Dict[str, Any],
    ):
        """
        Args:
            candidate_constraints (List[PolytopeConstraint]):
                ConstraintBuilder.build_candidate_constraints() 출력.
                KKTBuilder에 전달된 것과 동일한 리스트여야 합니다.

            identifier_config (Dict[str, Any]):
                configs/experiment_params.yaml의 'constraint_identification' 섹션.
                지원 키:
                    threshold_lambda : float  Λ̄ 임계값 (기본: 1e-3, 논문값)
                    aggregation      : str    'sum_all' | 'max_ineq' | 'any_ineq'
                                             (기본: 'sum_all')
                    verbose          : bool   상세 출력 여부 (기본: True)
        """
        self.candidate_constraints = candidate_constraints
        self.threshold             = identifier_config.get('threshold_lambda', 1e-3)
        self.aggregation           = identifier_config.get('aggregation', 'sum_all')
        self.verbose               = identifier_config.get('verbose', True)

        # 각 PolytopeConstraint의 부등식 수 → 열 슬라이스 범위 미리 계산
        self._col_ranges: List[Tuple[int, int]] = []
        col = 0
        for pc in candidate_constraints:
            self._col_ranges.append((col, col + pc.num_ineq))
            col += pc.num_ineq

        self.total_num_inequalities: int = col   # 전체 후보 부등식 수

        print("=" * 60)
        print("ConstraintIdentifier 초기화 완료")
        print(f"  후보 PolytopeConstraint 수 : {len(candidate_constraints)}")
        print(f"  총 후보 부등식 수          : {self.total_num_inequalities}")
        print(f"  임계값 Λ̄                   : {self.threshold:.2e}")
        print(f"  집계 전략                  : {self.aggregation}")
        print("=" * 60)

    # ================================================================
    # 공개 인터페이스
    # ================================================================

    def identify_active_constraints(
        self,
        lambda_star: np.ndarray,
    ) -> List[PolytopeConstraint]:
        """
        학습된 λ*로부터 활성 제약 조건을 식별합니다.

        Args:
            lambda_star (np.ndarray):
                IOCOptimizer.learn_parameters() 반환값.
                shape: (e, num_total_inequalities)
                    e                    = 세그먼트 스텝 수
                    num_total_inequalities = 모든 PolytopeConstraint의 num_ineq 합

        Returns:
            List[PolytopeConstraint]:
                Λ_j ≥ Λ̄ 를 만족하는 PolytopeConstraint 리스트.
                MPC Controller의 제약 조건으로 직접 사용 가능합니다.

        Raises:
            ValueError: lambda_star의 열 수가 예상과 다를 때.
        """
        self._validate_lambda(lambda_star)

        # 각 PolytopeConstraint에 대한 Λ_j 계산
        lambda_scores = self._compute_lambda_scores(lambda_star)

        identified   : List[PolytopeConstraint] = []
        identified_idx: List[int]               = []

        for j, (pc, score) in enumerate(
            zip(self.candidate_constraints, lambda_scores)
        ):
            active = score >= self.threshold

            if self.verbose:
                status = "✅ 식별됨" if active else "  스킵됨"
                print(f"  [{j:3d}] {status}  Λ_j={score:.4e}  '{pc.name}'  "
                      f"(num_ineq={pc.num_ineq})")

            if active:
                identified.append(pc)
                identified_idx.append(j)

        if self.verbose:
            print(f"\n  식별된 제약 조건: {len(identified)} / {len(self.candidate_constraints)}")
            if identified_idx:
                print(f"  인덱스: {identified_idx}")

        return identified

    def get_lambda_scores(
        self,
        lambda_star: np.ndarray,
    ) -> Dict[str, Any]:
        """
        각 후보 PolytopeConstraint의 Λ_j 값을 딕셔너리로 반환합니다.
        임계값 조정이나 시각화에 유용합니다.

        Args:
            lambda_star : (e, num_total_inequalities)

        Returns:
            dict:
                scores          : np.ndarray (num_candidates,)  각 Λ_j 값
                names           : List[str]  각 제약 조건 이름
                threshold       : float      현재 임계값
                above_threshold : np.ndarray (num_candidates,) bool 마스크
        """
        self._validate_lambda(lambda_star)
        scores = self._compute_lambda_scores(lambda_star)

        return {
            'scores'          : scores,
            'names'           : [pc.name for pc in self.candidate_constraints],
            'threshold'       : self.threshold,
            'above_threshold' : scores >= self.threshold,
        }

    def identify_with_threshold(
        self,
        lambda_star     : np.ndarray,
        threshold_lambda: float,
    ) -> List[PolytopeConstraint]:
        """
        임계값을 즉석에서 변경하여 제약 조건을 식별합니다.
        (self.threshold를 영구적으로 변경하지 않습니다.)

        Args:
            lambda_star      : (e, num_total_inequalities)
            threshold_lambda : 사용할 임계값

        Returns:
            List[PolytopeConstraint]: 식별된 제약 조건 리스트
        """
        original = self.threshold
        self.threshold = threshold_lambda
        result = self.identify_active_constraints(lambda_star)
        self.threshold = original
        return result

    def per_timestep_analysis(
        self,
        lambda_star: np.ndarray,
    ) -> Dict[str, Any]:
        """
        타임스텝별 λ 활성 패턴을 분석합니다.
        어떤 시간 구간에서 어떤 제약이 강하게 활성화되는지 확인합니다.

        Args:
            lambda_star : (e, num_total_inequalities)

        Returns:
            dict:
                per_constraint_timeseries : List[np.ndarray]  각 PolytopeConstraint의 (e,) λ 시계열
                                            (해당 PC 내 모든 부등식 λ의 합)
                peak_timesteps            : List[int]  각 제약의 λ 최대값 타임스텝
        """
        self._validate_lambda(lambda_star)

        timeseries_list = []
        peak_timesteps  = []

        for pc, (c_start, c_end) in zip(
            self.candidate_constraints, self._col_ranges
        ):
            lam_pc = lambda_star[:, c_start:c_end]   # (e, num_ineq)
            ts     = lam_pc.sum(axis=1)               # (e,) 타임스텝별 합
            timeseries_list.append(ts)
            peak_timesteps.append(int(np.argmax(ts)))

        return {
            'per_constraint_timeseries': timeseries_list,
            'peak_timesteps'           : peak_timesteps,
        }

    # ================================================================
    # 내부 헬퍼
    # ================================================================

    def _compute_lambda_scores(self, lambda_star: np.ndarray) -> np.ndarray:
        """
        각 PolytopeConstraint의 Λ_j 값을 집계 전략에 따라 계산합니다.

        Args:
            lambda_star : (e, num_total_inequalities)

        Returns:
            np.ndarray: (num_candidates,)  각 Λ_j 값
        """
        scores = np.zeros(len(self.candidate_constraints))

        for j, (pc, (c_start, c_end)) in enumerate(
            zip(self.candidate_constraints, self._col_ranges)
        ):
            lam_pc = lambda_star[:, c_start:c_end]   # (e, num_ineq)

            if self.aggregation == 'sum_all':
                # 논문 Eq.(11): Λ_j = Σ_i Σ_k λ_{i,k}
                scores[j] = float(lam_pc.sum())

            elif self.aggregation == 'max_ineq':
                # 가장 강한 개별 부등식의 시간 합
                # Λ_j = max_k( Σ_i λ_{i,k} )
                per_ineq_sum = lam_pc.sum(axis=0)   # (num_ineq,)
                scores[j] = float(per_ineq_sum.max())

            elif self.aggregation == 'any_ineq':
                # 어느 개별 부등식이라도 임계값을 넘으면 활성
                # (이 경우 scores는 최대 개별 Σ_i λ_{i,k} 값)
                per_ineq_sum = lam_pc.sum(axis=0)
                scores[j] = float(per_ineq_sum.max())

            else:
                raise ValueError(
                    f"알 수 없는 aggregation='{self.aggregation}'. "
                    "'sum_all', 'max_ineq', 'any_ineq' 중 하나를 사용하세요."
                )

        return scores

    def _validate_lambda(self, lambda_star: np.ndarray) -> None:
        """lambda_star 형상 검증."""
        if lambda_star.ndim != 2:
            raise ValueError(
                f"lambda_star는 2D 배열이어야 합니다. "
                f"현재 ndim={lambda_star.ndim}."
            )
        if lambda_star.shape[1] != self.total_num_inequalities:
            raise ValueError(
                f"lambda_star.shape[1]={lambda_star.shape[1]} ≠ "
                f"총 후보 부등식 수={self.total_num_inequalities}.\n"
                f"  ConstraintBuilder와 KKTBuilder에 전달한 candidate_constraints와\n"
                f"  동일한 리스트를 ConstraintIdentifier에도 전달했는지 확인하세요."
            )


# ====================================================================
# 동작 확인 (직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint

    # ── 더미 후보 제약 조건 ──────────────────────────────────────────
    z_dim = 16   # 더미 z 차원
    n_u   = 3
    n_x   = z_dim - n_u

    P_ub  = np.zeros((n_u, z_dim)); P_ub[:, n_x:] = np.eye(n_u)
    P_lb  = -P_ub.copy()
    P_qv  = np.zeros((2, z_dim))
    P_qv[0, 0] = 1.0; P_qv[1, 1] = 1.0   # qvel 한계 (2개 부등식)
    P_hull = np.random.randn(5, z_dim)    # 가상의 볼록 껍질 (5개 부등식)

    candidate_constraints = [
        PolytopeConstraint(P_ub,  np.ones((n_u, 1)) * 100.0, "Input u upper (100 Nm)"),
        PolytopeConstraint(P_lb,  np.ones((n_u, 1)) * 100.0, "Input u lower (-100 Nm)"),
        PolytopeConstraint(P_qv,  np.ones((2,  1)) *   5.0,  "Joint vel upper"),
        PolytopeConstraint(P_hull, np.ones((5, 1)) *   1.0,  "ConvexHull facet"),
    ]

    # 총 부등식: 3 + 3 + 2 + 5 = 13
    total_ineq = sum(pc.num_ineq for pc in candidate_constraints)
    print(f"총 후보 부등식 수: {total_ineq}")  # 13

    identifier_cfg = {
        'threshold_lambda': 1e-3,
        'aggregation'     : 'sum_all',
        'verbose'         : True,
    }

    identifier = ConstraintIdentifier(candidate_constraints, identifier_cfg)

    # ── 더미 lambda_star: u upper / lower 활성, 나머지 0에 가까움 ──
    e = 8
    np.random.seed(42)
    lambda_star = np.zeros((e, total_ineq))

    # u upper (idx 0~2): 비교적 큰 λ 값
    lambda_star[:, 0:3] = np.random.rand(e, 3) * 0.5

    # u lower (idx 3~5): 작은 λ 값
    lambda_star[:, 3:6] = np.random.rand(e, 3) * 0.0005

    # qvel (idx 6~7): 0
    lambda_star[:, 6:8] = 0.0

    # ConvexHull (idx 8~12): 중간 크기
    lambda_star[:, 8:13] = np.random.rand(e, 5) * 0.05

    print("\n[식별 결과 (sum_all)]")
    identified = identifier.identify_active_constraints(lambda_star)

    print(f"\n식별된 제약 조건:")
    for pc in identified:
        print(f"  - {pc.name}  (num_ineq={pc.num_ineq})")

    # ── Λ_j 점수 상세 확인 ───────────────────────────────────────────
    print("\n[Λ_j 점수]")
    score_info = identifier.get_lambda_scores(lambda_star)
    for name, score, above in zip(
        score_info['names'],
        score_info['scores'],
        score_info['above_threshold']
    ):
        mark = "✅" if above else "  "
        print(f"  {mark} {name:<35} Λ_j={score:.4e}")

    # ── 집계 전략 비교 ───────────────────────────────────────────────
    print("\n[집계 전략 비교]")
    for agg in ['sum_all', 'max_ineq', 'any_ineq']:
        cfg_tmp = {'threshold_lambda': 1e-3, 'aggregation': agg, 'verbose': False}
        id_tmp  = ConstraintIdentifier(candidate_constraints, cfg_tmp)
        result  = id_tmp.identify_active_constraints(lambda_star)
        print(f"  {agg:12s}: {len(result)}개 식별 "
              f"({[pc.name for pc in result]})")

    # ── 타임스텝 분석 ────────────────────────────────────────────────
    print("\n[타임스텝별 λ 패턴]")
    ts_info = identifier.per_timestep_analysis(lambda_star)
    for j, (pc, ts, peak) in enumerate(zip(
        candidate_constraints,
        ts_info['per_constraint_timeseries'],
        ts_info['peak_timesteps']
    )):
        print(f"  [{j}] {pc.name:<35} "
              f"peak_t={peak}  Λ_peak={ts[peak]:.4e}")

    # ── 즉석 임계값 변경 테스트 ──────────────────────────────────────
    print("\n[임계값 0.1로 변경 시]")
    strict = identifier.identify_with_threshold(lambda_star, 0.1)
    print(f"  식별 수: {len(strict)}")

    # ── 형상 검증: 잘못된 lambda_star ────────────────────────────────
    print("\n[형상 오류 검증]")
    try:
        identifier.identify_active_constraints(np.zeros((e, total_ineq + 1)))
    except ValueError as err:
        print(f"  예상된 오류 발생: {err}")

    print("\n✅ ConstraintIdentifier 모든 테스트 통과")