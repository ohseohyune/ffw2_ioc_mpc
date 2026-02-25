#!/usr/bin/env python3
"""Standalone smoke test for IOCOptimizer with dummy dynamics."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import casadi as ca
import numpy as np

from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
from ffw2_ioc_mpc.ioc.kkt_builder import KKTBuilder
from ffw2_ioc_mpc.ioc.optimizer import IOCOptimizer


class DummyDyn:
    state_dim = 4
    input_dim = 2
    nq = 3
    nv = 2

    def __init__(self):
        np.random.seed(7)
        A = np.eye(self.state_dim) * 0.98
        B = np.random.randn(self.state_dim, self.input_dim) * 0.05
        xs = ca.MX.sym("x", self.state_dim)
        us = ca.MX.sym("u", self.input_dim)
        Ms = ca.MX.sym("M", self.nv, self.nv)
        Gs = ca.MX.sym("G", self.nv, 1)
        xn = ca.MX(A) @ xs + ca.MX(B) @ us
        self._f = ca.Function(
            "f",
            [xs, us, Ms, Gs],
            [xn],
            ["x", "u", "M", "CG"],
            ["x_next"],
        )

    def get_casadi_func(self):
        return self._f


def main() -> None:
    dyn = DummyDyn()
    cost_cfg = {"target_state_selection": {"indices": [0, 1]}}
    cost = ParametricStageCost(dyn.state_dim, dyn.input_dim, cost_cfg)

    z_dim = dyn.state_dim + dyn.input_dim
    P_ub = np.zeros((dyn.input_dim, z_dim))
    P_ub[:, dyn.state_dim :] = np.eye(dyn.input_dim)
    cons = [
        PolytopeConstraint(P_ub, np.ones((dyn.input_dim, 1)) * 5.0, "u upper"),
        PolytopeConstraint(-P_ub, np.ones((dyn.input_dim, 1)) * 5.0, "u lower"),
    ]

    e = 4
    kkt = KKTBuilder(dyn, cost, cons, e=e)

    opt_cfg = {
        "solver": "scipy_slsqp",
        "max_iterations": 300,
        "num_initial_guesses": 2,
        "initial_guesses_strategy": "random",
        "random_init_scale": 0.05,
        "l1_reg_lambda": 1e-4,
        "l1_reg_nu": 0.0,
        "tol": 1e-8,
    }

    optimizer = IOCOptimizer(kkt, cost, opt_cfg)

    np.random.seed(42)
    sd, ud, nv, td = dyn.state_dim, dyn.input_dim, dyn.nv, cost.target_dim

    Xm = np.random.randn(sd, e + 1) * 0.1
    Um = np.random.randn(ud, e) * 0.3
    ys = np.random.randn(td, e) * 0.1

    M_data = KKTBuilder.prepare_M_data([np.eye(nv)] * e)
    CG_data = KKTBuilder.prepare_CG_data([np.zeros(nv)] * e)

    theta_s, nu_s, lam_s, obj = optimizer.learn_parameters(
        Xm, Um, ys, M_data, CG_data
    )

    print(f"학습된 theta (처음 5개): {theta_s[:5].round(6)}")
    print(f"최종 objective        : {obj:.4e}")
    print(
        f"lambda_star shape     : {lam_s.shape}  "
        f"(기대: ({e},{kkt.num_total_constraints}))"
    )

    optimizer.analyze_result(theta_s, lam_s)
    print("\n✅ IOCOptimizer 테스트 완료")


if __name__ == "__main__":
    main()
