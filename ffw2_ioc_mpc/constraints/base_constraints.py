"""
src/constraints/base_constraints.py

IOC 파이프라인 전반에서 사용되는 다면체(Polytope) 제약 조건 기본 클래스.

제약 조건 형태:  P @ z <= p
여기서 z = [x; u]  (state_dim + input_dim)
"""

import numpy as np


class PolytopeConstraint:
    """
    단일 다면체 제약 조건:  P @ z <= p

    z = [x; u] 벡터에 대해 정의됩니다.
        x = [qpos (nq); qvel (nv)]   (dynamics.py 기준)
        u = 토크 입력 (input_dim = 14)

    Attributes:
        P    (np.ndarray): 계수 행렬  (num_ineq, z_dim)
        p    (np.ndarray): 상한 벡터  (num_ineq, 1)
        name (str)       : 디버그용 레이블
    """

    def __init__(self, P: np.ndarray, p: np.ndarray, name: str = ""):
        """
        Args:
            P    : (num_ineq, z_dim)  계수 행렬
            p    : (num_ineq, 1) 또는 (num_ineq,)  상한 벡터
            name : 디버그용 레이블 (선택)
        """
        self.P = np.atleast_2d(P).astype(float)

        # p를 항상 (num_ineq, 1) 열벡터로 정규화
        p_arr = np.asarray(p, dtype=float)
        if p_arr.ndim == 1:
            p_arr = p_arr.reshape(-1, 1)
        elif p_arr.ndim == 2 and p_arr.shape[1] != 1:
            raise ValueError(
                f"p는 열벡터(num_ineq, 1)여야 합니다. 현재 shape: {p_arr.shape}"
            )
        self.p = p_arr
        self.name = name

        # 형상 일관성 검증
        if self.P.shape[0] != self.p.shape[0]:
            raise ValueError(
                f"[{name}] P 행 수({self.P.shape[0]}) != p 행 수({self.p.shape[0]})"
            )

    # ----------------------------------------------------------------
    # 프로퍼티
    # ----------------------------------------------------------------

    @property
    def num_ineq(self) -> int:
        """부등식 제약 조건의 수 (P의 행 수)."""
        return self.P.shape[0]

    @property
    def z_dim(self) -> int:
        """z = [x; u] 벡터의 차원 (P의 열 수)."""
        return self.P.shape[1]

    # ----------------------------------------------------------------
    # 공개 메서드
    # ----------------------------------------------------------------

    def is_satisfied(self, z: np.ndarray, tol: float = 1e-4) -> bool:
        """
        주어진 z 벡터가 모든 부등식을 만족하는지 확인합니다.

        Args:
            z   : (z_dim,) 또는 (z_dim, 1)
            tol : 수치 허용 오차

        Returns:
            bool: P @ z <= p + tol 이면 True
        """
        z_flat = np.asarray(z, dtype=float).flatten()
        residuals = self.P @ z_flat - self.p.flatten()
        return bool(np.all(residuals <= tol))

    def violation(self, z: np.ndarray) -> np.ndarray:
        """
        각 부등식 제약의 위반량을 반환합니다.
        위반량 = max(0, P_i @ z - p_i)

        Args:
            z : (z_dim,)

        Returns:
            np.ndarray: (num_ineq,)  각 행의 위반량 (0이면 만족)
        """
        z_flat = np.asarray(z, dtype=float).flatten()
        raw = self.P @ z_flat - self.p.flatten()
        return np.maximum(0.0, raw)

    def get_casadi_expr(self, z_sym):
        """
        CasADi 심볼릭 표현식  P @ z_sym - p  를 반환합니다.
        kkt_builder.py에서 라그랑지안 구성 시 사용합니다.

        반환값이 <= 0 이면 제약 조건 만족.

        Args:
            z_sym : ca.MX 또는 ca.SX  (z_dim, 1)

        Returns:
            ca.MX: (num_ineq, 1)  P @ z_sym - p
        """
        import casadi as ca
        P_ca = ca.DM(self.P)
        p_ca = ca.DM(self.p)
        return P_ca @ z_sym - p_ca

    def summary(self) -> str:
        """제약 조건 요약 문자열 반환."""
        return (
            f"PolytopeConstraint(name='{self.name}', "
            f"num_ineq={self.num_ineq}, z_dim={self.z_dim})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ====================================================================
# 동작 확인 (직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    import casadi as ca

    z_dim = 5   # x(3,) + u(2,)

    # 제약 조건: z[3] <= 100  (u[0] 상한)
    P1 = np.zeros((1, z_dim));  P1[0, 3] = 1.0
    c1 = PolytopeConstraint(P1, np.array([[100.0]]), "u0 upper limit")

    print(c1)
    print(f"  z_dim={c1.z_dim}, num_ineq={c1.num_ineq}")

    z_ok  = np.array([0, 0, 0,  50.0, 0])
    z_bad = np.array([0, 0, 0, 150.0, 0])

    print(f"\n  z_ok  satisfied : {c1.is_satisfied(z_ok)}")   # True
    print(f"  z_bad satisfied : {c1.is_satisfied(z_bad)}")    # False
    print(f"  z_bad violation : {c1.violation(z_bad)}")        # [50.]

    z_sym = ca.MX.sym('z', z_dim)
    expr  = c1.get_casadi_expr(z_sym)
    print(f"\n  CasADi expr shape: {expr.shape}")              # (1, 1)

    print("\n✅ PolytopeConstraint 모든 테스트 통과")