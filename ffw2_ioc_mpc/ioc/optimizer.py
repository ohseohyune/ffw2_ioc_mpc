"""
src/ioc/optimizer.py

논문 Eq.(12)의 이완된 최적화 문제를 풀어 비용 파라미터 θ와
라그랑주 승수 (λ, ν)를 학습하는 IOC 최적화 클래스.

────────────────────────────────────────────────────────────
논문 Eq.(12) 최적화 문제:
    min_{θ, ν, λ}  ‖∇_U L(U^m, λ, ν, θ)‖²

    s.t.
        λ_i^T C̄(x(k+i), u(k+i)) = 0    i = 0,...,e-1   (상보성 여유)
        λ_i ≥ 0                          i = 0,...,e-1   (쌍대 가능성)
        Σ R_{ii} = 1                                      (스케일 정규화)

솔버: CasADi + IPOPT (기본) / scipy SLSQP (fallback)
"""

import sys
import time
import threading
import warnings
import numpy as np
import casadi as ca
from typing import List, Dict, Any, Tuple, Optional

try:
    from ffw2_ioc_mpc.ioc.kkt_builder import KKTBuilder
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from ffw2_ioc_mpc.ioc.kkt_builder import KKTBuilder
    from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost


class IOCOptimizer:
    """
    시연 데이터로부터 역최적 제어 비용 파라미터를 학습합니다.

    사용 흐름:
        1. IOCOptimizer 생성 (KKTBuilder, ParametricStageCost 주입)
        2. learn_parameters() 호출 → (θ*, ν*, λ*, obj_val) 반환
        3. 학습된 θ*를 MPC Controller에 전달하여 재현 실험 수행

    Notes:
        - 비볼록(non-convex) 문제이므로 num_initial_guesses > 1 권장
        - IPOPT 사용 시 HSL MA27/MA57 라이선스가 있으면 수렴이 훨씬 빨라짐
        - 실제 로봇 데이터 없이 랜덤 더미 데이터로는 수렴을 기대하기 어려움
    """

    def __init__(
        self,
        kkt_builder     : KKTBuilder,
        stage_cost      : ParametricStageCost,
        optimizer_config: Dict[str, Any],
    ):
        """
        Args:
            kkt_builder (KKTBuilder):
                build_kkt_residual_functions()를 내부에서 자동 호출합니다.

            stage_cost (ParametricStageCost):
                R 정규화 제약 조건 생성에 사용.

            optimizer_config (Dict[str, Any]):
                지원 키:
                    solver              : 'ipopt' | 'scipy_slsqp'  (기본: 'ipopt')
                    max_iterations      : int    (기본: 2000)
                    num_initial_guesses : int    (기본: 3)
                    initial_guesses_strategy: 'random' | 'zeros'
                    random_init_scale   : float  (기본: 0.01)
                    l1_reg_lambda       : float  (기본: 0.0)
                    l1_reg_nu           : float  (기본: 0.0)
                    ipopt_print_level   : int    (기본: 3)
                    ipopt_hessian_approximation: str | None (기본: 'limited-memory')
                    ipopt_sb            : 'yes' | 'no' | None
                    casadi_print_time   : bool   (기본: False)
                    casadi_verbose      : bool   (기본: False)
                    casadi_verbose_init : bool   (기본: False)
                    show_solver_progress: bool   (기본: True)
                    suppress_casadi_output: bool (기본: False)
                    tol                 : float  (기본: 1e-6)
                    enable_chunked_ipopt: bool   (기본: True)
                    ipopt_chunk_max_iter: int    (기본: 50)
                    log_grad_every_sec  : float  (기본: 10.0)
        """
        self.kkt  = kkt_builder
        self.cost = stage_cost
        self.cfg  = optimizer_config
        self._last_solver_stats: Dict[str, Any] = {}

        # KKT 잔차 함수 빌드
        self.grad_func, self.csc_func = kkt_builder.build_kkt_residual_functions()

        # 차원
        self.theta_dim       = kkt_builder.theta_dim
        self.nu_dim          = kkt_builder.state_dim
        self.lambda_flat_dim = kkt_builder.e * kkt_builder.num_total_constraints
        self.U_dim           = kkt_builder.e * kkt_builder.input_dim
        self.e               = kkt_builder.e
        self.n_c             = kkt_builder.num_total_constraints

        # 총 최적화 변수: [theta | nu | lambda_flat]
        self.total_vars = self.theta_dim + self.nu_dim + self.lambda_flat_dim

        # R 정규화 CasADi 함수
        theta_sym_tmp = ca.MX.sym('th', self.theta_dim)
        self._r_norm_func = ca.Function(
            'r_norm',
            [theta_sym_tmp],
            [stage_cost.normalization_constraint_expr(theta_sym_tmp)]
        )

        print("=" * 60)
        print("IOCOptimizer 초기화 완료")
        print(f"  솔버            : {self.cfg.get('solver', 'ipopt')}")
        print(f"  theta_dim       : {self.theta_dim}")
        print(f"  nu_dim          : {self.nu_dim}")
        print(f"  lambda_flat_dim : {self.lambda_flat_dim}")
        print(f"  총 최적화 변수  : {self.total_vars}")
        print(f"  초기 추정 횟수  : {self.cfg.get('num_initial_guesses', 3)}")
        print("=" * 60)

    def _start_solver_spinner(self, label: str = "IPOPT solve"):
        """
        솔버 호출 중 TTY에 간단한 진행 표시(스피너 + 경과시간)를 출력합니다.
        정확한 진행률(%)은 IPOPT/CasADi에서 총 반복 수를 미리 알 수 없어 제공하지 않습니다.
        """
        if not self.cfg.get('show_solver_progress', True):
            return None
        if not sys.stdout.isatty():
            return None

        stop_event = threading.Event()
        frames = "|/-\\"
        started = time.perf_counter()

        def _worker():
            i = 0
            while not stop_event.wait(0.15):
                elapsed = time.perf_counter() - started
                sys.stdout.write(
                    f"\r  {label} {frames[i % len(frames)]}  elapsed={elapsed:6.1f}s"
                )
                sys.stdout.flush()
                i += 1

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return (stop_event, thread)

    @staticmethod
    def _stop_solver_spinner(spinner_handle) -> None:
        if spinner_handle is None:
            return
        stop_event, thread = spinner_handle
        stop_event.set()
        thread.join(timeout=0.3)
        # 현재 줄 정리
        sys.stdout.write("\r" + (" " * 80) + "\r")
        sys.stdout.flush()

    def _call_casadi_solver(
        self,
        solver_fn,
        x0: np.ndarray,
        lbw: np.ndarray,
        ubw: np.ndarray,
        lbg: np.ndarray,
        ubg: np.ndarray,
    ):
        """
        CasADi solver 호출 래퍼.
        필요 시 C/C++ 레벨 stdout/stderr를 /dev/null로 리다이렉트합니다.
        """
        if self.cfg.get('suppress_casadi_output', False):
            import os
            devnull_fd = os.open(os.devnull, os.O_RDWR)
            orig_stdout_fd = os.dup(1)
            orig_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                return solver_fn(x0=x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            finally:
                os.dup2(orig_stdout_fd, 1)
                os.dup2(orig_stderr_fd, 2)
                os.close(orig_stdout_fd)
                os.close(orig_stderr_fd)
                os.close(devnull_fd)
        return solver_fn(x0=x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # ================================================================
    # 공개 인터페이스
    # ================================================================

    def learn_parameters(
        self,
        Xm_data : np.ndarray,
        Um_data : np.ndarray,
        ys_data : np.ndarray,
        M_data  : np.ndarray,
        CG_data : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        시연 데이터로부터 비용 파라미터와 라그랑주 승수를 학습합니다.

        Args:
            Xm_data : (state_dim, e+1)   열 = 타임스텝
            Um_data : (input_dim, e)
            ys_data : (target_dim, e)
            M_data  : (nv, nv * e)       KKTBuilder.prepare_M_data() 출력
            CG_data : (nv, e)            KKTBuilder.prepare_CG_data() 출력

        Returns:
            theta_star    : (theta_dim,)   학습된 비용 파라미터
            nu_star       : (nu_dim,)      학습된 종단 승수
            lambda_star   : (e, n_c)       학습된 부등식 승수
            final_obj_val : float          최종 ‖∇_U L‖² 값
        """
        # Um_data: (input_dim, e) → U_flat: (e * input_dim,)
        U_flat = Um_data.T.flatten(order='C')

        fixed_data = (Xm_data, Um_data, ys_data, M_data, CG_data, U_flat)

        solver    = self.cfg.get('solver', 'ipopt')
        num_tries = self.cfg.get('num_initial_guesses', 3)

        best_result = None
        best_obj    = np.inf

        for trial in range(num_tries):
            print(f"\n{'─'*55}")
            print(f"  [Trial {trial+1}/{num_tries}]  솔버={solver}")
            print(f"{'─'*55}")

            x0 = self._generate_initial_guess()

            try:
                t0 = time.perf_counter()

                if solver == 'ipopt':
                    opt_vars, obj_val, success, msg = \
                        self._solve_ipopt(x0, fixed_data)
                else:
                    opt_vars, obj_val, success, msg = \
                        self._solve_scipy(x0, fixed_data)

                elapsed = time.perf_counter() - t0
                print(f"  완료: success={success}  obj={obj_val:.4e}  "
                      f"({elapsed:.1f}s)  msg={msg}")

                diag = self._compute_trial_diagnostics(opt_vars, fixed_data)
                print(
                    "    진단: "
                    f"||grad||2={diag['grad_norm_l2']:.3e}  "
                    f"||grad||inf={diag['grad_norm_inf']:.3e}  "
                    f"csc_inf={diag['csc_inf']:.3e}  "
                    f"|r_norm|={diag['r_norm_abs']:.3e}  "
                    f"lam_min={diag['lam_min']:.3e}"
                )
                print(
                    "    norm: "
                    f"||theta||2={diag['theta_norm_l2']:.3e}  "
                    f"||nu||2={diag['nu_norm_l2']:.3e}  "
                    f"||lam||2={diag['lam_norm_l2']:.3e}"
                )

                iter_count = self._last_solver_stats.get('iter_count', None)
                t_wall = self._last_solver_stats.get('t_wall_total', None)
                t_proc = self._last_solver_stats.get('t_proc_total', None)
                if iter_count is not None or t_wall is not None or t_proc is not None:
                    extra_parts = []
                    if iter_count is not None:
                        extra_parts.append(f"iter={iter_count}")
                    if t_wall is not None:
                        extra_parts.append(f"t_wall={float(t_wall):.3f}s")
                    if t_proc is not None:
                        extra_parts.append(f"t_proc={float(t_proc):.3f}s")
                    print("    stats: " + "  ".join(extra_parts))

                if obj_val < best_obj:
                    best_obj    = obj_val
                    best_result = opt_vars.copy()

            except Exception as ex:
                warnings.warn(f"  [Trial {trial+1}] 예외 발생: {ex}")
                continue

        if best_result is None:
            raise RuntimeError(
                f"모든 {num_tries}번의 시도에서 최적화에 실패했습니다."
            )

        theta_star = best_result[:self.theta_dim]
        nu_star    = best_result[self.theta_dim : self.theta_dim + self.nu_dim]
        lam_flat   = best_result[self.theta_dim + self.nu_dim:]
        lambda_star = lam_flat.reshape(self.e, self.n_c)

        print(f"\n{'='*55}")
        print(f"  학습 완료  최적 obj = {best_obj:.4e}")
        print(f"  theta_star (처음 5개) : {theta_star[:5].round(6)}")
        print(f"  λ 합 (활성 제약 지표): {lam_flat.sum():.4f}")

        r_norm_val = float(self._r_norm_func(theta_star))
        print(f"  R 정규화 잔차        : {r_norm_val:.6f}  (0에 가까워야 함)")
        print(f"{'='*55}\n")

        return theta_star, nu_star, lambda_star, best_obj

    # ================================================================
    # 솔버 구현 A: CasADi + IPOPT
    # ================================================================

    def _solve_ipopt(
        self,
        x0        : np.ndarray,
        fixed_data: tuple,
    ) -> Tuple[np.ndarray, float, bool, str]:
        """
        CasADi nlpsol + IPOPT 로 NLP를 풉니다.

        NLP 구조:
            변수: w = [theta | nu | lambda_flat]
            목적: ‖∇_U L‖² + L1 정규화
            등식: [CSC(e개); R_norm(1개)]
            경계: lambda_flat >= 0
        """
        Xm, Um, ys, M, CG, U_flat = fixed_data

        n_theta = self.theta_dim
        n_nu    = self.nu_dim
        n_lam   = self.lambda_flat_dim
        e       = self.e

        # 심볼릭 최적화 변수
        w_sym      = ca.MX.sym('w', self.total_vars)
        theta_s    = w_sym[:n_theta]
        nu_s       = w_sym[n_theta : n_theta + n_nu]
        lam_flat_s = w_sym[n_theta + n_nu:]

        # 고정 데이터 → DM
        Xm_dm = ca.DM(Xm);  Um_dm = ca.DM(Um)
        ys_dm = ca.DM(ys);  M_dm  = ca.DM(M)
        CG_dm = ca.DM(CG);  U_dm  = ca.DM(U_flat)

        # 목적 함수
        grad_val = self.grad_func(
            theta_s, nu_s, lam_flat_s,
            U_dm, Xm_dm, Um_dm, ys_dm, M_dm, CG_dm
        )
        obj = ca.sumsqr(grad_val)

        l1_lam = self.cfg.get('l1_reg_lambda', 0.0)
        l1_nu  = self.cfg.get('l1_reg_nu',  0.0)
        if l1_lam > 0:
            obj += l1_lam * ca.sum1(ca.fabs(lam_flat_s))
        if l1_nu > 0:
            obj += l1_nu * ca.sum1(ca.fabs(nu_s))

        # 등식 제약
        csc_expr    = self.csc_func(
            theta_s, nu_s, lam_flat_s,
            U_dm, Xm_dm, Um_dm, ys_dm, M_dm, CG_dm
        )
        r_norm_expr = self.cost.normalization_constraint_expr(theta_s)
        # IPOPT(CasADi)는 NLPSOL_G가 dense vector이길 기대합니다.
        # csc_expr가 희소 zero 벡터인 경우(후보 제약 0개) assert가 발생할 수 있어 densify 처리.
        g_expr      = ca.densify(ca.vertcat(csc_expr, r_norm_expr))   # (e+1, 1)

        # 변수 경계
        lbw = np.concatenate([
            np.full(n_theta + n_nu, -np.inf),
            np.zeros(n_lam)
        ])
        ubw = np.full(self.total_vars, np.inf)

        # 등식 제약 경계
        n_g   = e + 1
        lbg   = np.zeros(n_g)
        ubg   = np.zeros(n_g)

        # NLP 정의
        nlp = {'x': w_sym, 'f': obj, 'g': g_expr}
        max_total_iter = int(self.cfg.get('max_iterations', 2000))
        chunk_max_iter = int(self.cfg.get('ipopt_chunk_max_iter', 150))
        if chunk_max_iter < 1:
            chunk_max_iter = 1
        enable_chunked = bool(self.cfg.get('enable_chunked_ipopt', True))
        use_chunked = enable_chunked and (max_total_iter > chunk_max_iter)

        ipopt_opts = {
            'ipopt.max_iter'             : chunk_max_iter if use_chunked else max_total_iter,
            'ipopt.tol'                  : self.cfg.get('tol'),
            'ipopt.print_level'          : self.cfg.get('ipopt_print_level', 0),
            'ipopt.print_frequency_iter' : self.cfg.get('print_frequency_iter', 10),
            # CasADi-level options: allow configuration from optimizer config
            'print_time'                 : 1 if self.cfg.get('casadi_print_time', False) else 0,
            'verbose'                    : bool(self.cfg.get('casadi_verbose', False)),
            'verbose_init'               : bool(self.cfg.get('casadi_verbose_init', False)),
        }

        hessian_approx = self.cfg.get('ipopt_hessian_approximation', 'limited-memory')
        if hessian_approx:
            # Dense exact Hessian codegen can dominate initialization time for this IOC NLP.
            ipopt_opts['ipopt.hessian_approximation'] = hessian_approx

        ipopt_sb = self.cfg.get('ipopt_sb', 'yes')
        if ipopt_sb is not None:
            ipopt_opts['ipopt.sb'] = ipopt_sb

        solver_fn = ca.nlpsol('ioc_nlp', 'ipopt', nlp, ipopt_opts)
        sys.stdout.flush()

        if use_chunked:
            w_curr = np.array(x0, dtype=float).copy()
            obj_val = np.inf
            success = False
            msg = ""
            iter_accum = 0
            t_wall_total = 0.0
            t_proc_total = 0.0
            last_stats: Dict[str, Any] = {}

            log_every_sec = float(self.cfg.get('log_grad_every_sec', 10.0))
            if log_every_sec <= 0:
                log_every_sec = 10.0

            t_start = time.perf_counter()
            last_log_t = t_start

            while iter_accum < max_total_iter:
                t_wall_0 = time.perf_counter()
                t_proc_0 = time.process_time()
                sol = self._call_casadi_solver(
                    solver_fn, w_curr, lbw, ubw, lbg, ubg
                )
                t_wall_total += time.perf_counter() - t_wall_0
                t_proc_total += time.process_time() - t_proc_0

                w_curr = np.array(sol['x']).flatten()
                obj_val = float(sol['f'])
                stats = solver_fn.stats()
                last_stats = dict(stats)

                iter_chunk = int(stats.get('iter_count', chunk_max_iter))
                if iter_chunk <= 0:
                    iter_chunk = chunk_max_iter
                iter_accum += iter_chunk

                success = bool(stats.get('success', False))
                msg = str(stats.get('return_status', ''))

                now = time.perf_counter()
                should_log = (now - last_log_t) >= log_every_sec
                if should_log or success or iter_accum >= max_total_iter:
                    theta_now = w_curr[:n_theta]
                    nu_now = w_curr[n_theta : n_theta + n_nu]
                    lam_now = w_curr[n_theta + n_nu:]
                    grad_now = np.array(
                        self.grad_func(theta_now, nu_now, lam_now, U_flat, Xm, Um, ys, M, CG)
                    ).astype(float).flatten()
                    g2 = float(np.linalg.norm(grad_now, ord=2))
                    ginf = float(np.linalg.norm(grad_now, ord=np.inf))
                    elapsed = now - t_start
                    iter_disp = min(iter_accum, max_total_iter)
                    print(
                        "    [IPOPT-progress] "
                        f"elapsed={elapsed:7.1f}s  "
                        f"iter={iter_disp}/{max_total_iter}  "
                        f"||grad||2={g2:.3e}  ||grad||inf={ginf:.3e}"
                    )
                    last_log_t = now

                if success:
                    break

                status_lower = msg.lower()
                max_iter_status = (
                    'maximum_iterations_exceeded' in status_lower
                    or 'maximum iterations exceeded' in status_lower
                )
                if not max_iter_status:
                    break

            self._last_solver_stats = dict(last_stats)
            self._last_solver_stats['iter_count'] = int(iter_accum)
            self._last_solver_stats['t_wall_total'] = float(t_wall_total)
            self._last_solver_stats['t_proc_total'] = float(t_proc_total)
            return w_curr, float(obj_val), bool(success), msg

        spinner = self._start_solver_spinner("IPOPT solving")
        try:
            sol = self._call_casadi_solver(
                solver_fn, x0, lbw, ubw, lbg, ubg
            )
        finally:
            self._stop_solver_spinner(spinner)

        w_opt = np.array(sol['x']).flatten()
        obj_val = float(sol['f'])
        stats = solver_fn.stats()
        self._last_solver_stats = dict(stats)
        success = bool(stats.get('success', False))
        msg = str(stats.get('return_status', ''))

        t_wall = stats.get('t_wall_total', None)
        t_proc = stats.get('t_proc_total', None)
        if t_wall is not None:
            self._last_solver_stats['t_wall_total'] = float(t_wall)
        if t_proc is not None:
            self._last_solver_stats['t_proc_total'] = float(t_proc)

        return w_opt, obj_val, success, msg

    # ================================================================
    # 솔버 구현 B: scipy SLSQP
    # ================================================================

    def _solve_scipy(
        self,
        x0        : np.ndarray,
        fixed_data: tuple,
    ) -> Tuple[np.ndarray, float, bool, str]:
        """scipy.optimize.minimize (SLSQP) fallback 솔버."""
        from scipy.optimize import minimize

        Xm, Um, ys, M, CG, U_flat = fixed_data

        bounds = (
            [(None, None)] * (self.theta_dim + self.nu_dim)
            + [(0.0, None)] * self.lambda_flat_dim
        )
        constraints = [
            {
                'type': 'eq',
                'fun' : self._scipy_csc,
                'args': (Xm, Um, ys, M, CG, U_flat),
            },
            {
                'type': 'eq',
                'fun' : self._scipy_r_norm,
            },
        ]

        res = minimize(
            fun        = self._scipy_obj,
            x0         = x0,
            args       = (Xm, Um, ys, M, CG, U_flat),
            method     = 'SLSQP',
            bounds     = bounds,
            constraints= constraints,
            options    = {
                'maxiter': self.cfg.get('max_iterations'),
                'ftol'   : self.cfg.get('tol', 1e-9),
                'disp'   : False,
            },
        )

        self._last_solver_stats = {
            'iter_count': int(getattr(res, 'nit', -1)),
            'return_status': str(res.message),
            'success': bool(res.success),
        }

        return res.x, float(res.fun), res.success, res.message

    def _scipy_obj(self, w, Xm, Um, ys, M, CG, U_flat) -> float:
        theta = w[:self.theta_dim]
        nu    = w[self.theta_dim : self.theta_dim + self.nu_dim]
        lam   = w[self.theta_dim + self.nu_dim:]
        g = np.array(
            self.grad_func(theta, nu, lam, U_flat, Xm, Um, ys, M, CG)
        ).flatten()
        obj = float(g @ g)
        l1_lam = self.cfg.get('l1_reg_lambda', 0.0)
        l1_nu  = self.cfg.get('l1_reg_nu', 0.0)
        if l1_lam > 0: obj += l1_lam * float(np.abs(lam).sum())
        if l1_nu  > 0: obj += l1_nu  * float(np.abs(nu).sum())
        return obj

    def _scipy_csc(self, w, Xm, Um, ys, M, CG, U_flat) -> np.ndarray:
        theta = w[:self.theta_dim]
        nu    = w[self.theta_dim : self.theta_dim + self.nu_dim]
        lam   = w[self.theta_dim + self.nu_dim:]
        return np.array(
            self.csc_func(theta, nu, lam, U_flat, Xm, Um, ys, M, CG)
        ).flatten()

    def _scipy_r_norm(self, w) -> float:
        return float(self._r_norm_func(w[:self.theta_dim]))

    def _compute_trial_diagnostics(
        self,
        opt_vars: np.ndarray,
        fixed_data: tuple,
    ) -> Dict[str, float]:
        """
        Trial 종료 시 병목 위치 파악을 위한 잔차/노름 요약치를 계산합니다.
        """
        Xm, Um, ys, M, CG, U_flat = fixed_data

        theta = opt_vars[:self.theta_dim]
        nu    = opt_vars[self.theta_dim : self.theta_dim + self.nu_dim]
        lam   = opt_vars[self.theta_dim + self.nu_dim:]

        grad = np.array(
            self.grad_func(theta, nu, lam, U_flat, Xm, Um, ys, M, CG)
        ).astype(float).flatten()
        csc = np.array(
            self.csc_func(theta, nu, lam, U_flat, Xm, Um, ys, M, CG)
        ).astype(float).flatten()
        r_norm = float(self._r_norm_func(theta))

        grad_norm_l2  = float(np.linalg.norm(grad, ord=2))
        grad_norm_inf = float(np.linalg.norm(grad, ord=np.inf))
        csc_inf       = float(np.max(np.abs(csc))) if csc.size > 0 else 0.0
        lam_min       = float(np.min(lam)) if lam.size > 0 else 0.0

        return {
            'grad_norm_l2' : grad_norm_l2,
            'grad_norm_inf': grad_norm_inf,
            'csc_inf'      : csc_inf,
            'r_norm_abs'   : abs(r_norm),
            'lam_min'      : lam_min,
            'theta_norm_l2': float(np.linalg.norm(theta, ord=2)),
            'nu_norm_l2'   : float(np.linalg.norm(nu, ord=2)),
            'lam_norm_l2'  : float(np.linalg.norm(lam, ord=2)),
        }

    # ================================================================
    # 초기 추정값 생성
    # ================================================================

    def _generate_initial_guess(self) -> np.ndarray:
        strategy = self.cfg.get('initial_guesses_strategy', 'random')
        scale    = self.cfg.get('random_init_scale', 0.01)

        theta = self.cost.theta_init(scale=scale)

        if strategy == 'random':
            nu  = np.random.randn(self.nu_dim) * scale
            lam = np.abs(np.random.randn(self.lambda_flat_dim)) * scale
        elif strategy == 'zeros':
            nu  = np.zeros(self.nu_dim)
            lam = np.ones(self.lambda_flat_dim) * 1e-4
        else:
            raise ValueError(f"알 수 없는 strategy: '{strategy}'")

        return np.concatenate([theta, nu, lam])

    # ================================================================
    # 학습 결과 분석
    # ================================================================

    def analyze_result(
        self,
        theta_star  : np.ndarray,
        lambda_star : np.ndarray,
        active_tol  : float = 1e-4,
    ) -> Dict[str, Any]:
        """
        학습 결과를 분석하여 Q, R 행렬과 활성 제약 통계를 출력합니다.

        Args:
            theta_star  : (theta_dim,)
            lambda_star : (e, n_c)
            active_tol  : λ_ij > active_tol 이면 활성 제약으로 판단

        Returns:
            dict with Q, R, R_trace, active_mask, n_active
        """
        Q_fn, R_fn = self.cost.get_Q_R_functions()
        Q_val   = np.array(Q_fn(theta_star))
        R_val   = np.array(R_fn(theta_star))
        R_trace = float(np.trace(R_val))

        active_mask = lambda_star > active_tol
        n_active    = int(active_mask.sum())

        print(f"\n[분석 결과]")
        print(f"  Q:\n{Q_val.round(6)}")
        print(f"  R 대각: {np.diag(R_val).round(6)}")
        print(f"  R_trace (≈1.0 목표): {R_trace:.6f}")
        print(f"  활성 부등식: {n_active} / {self.e * self.n_c}")

        return {
            'Q'          : Q_val,
            'R'          : R_val,   
            'R_trace'    : R_trace,
            'active_mask': active_mask,
            'n_active'   : n_active,
        }


# # ====================================================================
# # 동작 확인 (직접 실행 시)
# # ====================================================================
# if __name__ == "__main__":
#     import sys, os
#     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

#     import casadi as ca
#     from ffw2_ioc_mpc.constraints.base_constraints import PolytopeConstraint
#     from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
#     from ffw2_ioc_mpc.ioc.kkt_builder import KKTBuilder

#     # ── 더미 DynamicsModel ───────────────────────────────────────────
#     class DummyDyn:
#         state_dim = 4; input_dim = 2; nq = 3; nv = 2

#         def __init__(self):
#             np.random.seed(7)
#             A = np.eye(self.state_dim) * 0.98
#             B = np.random.randn(self.state_dim, self.input_dim) * 0.05
#             xs = ca.MX.sym('x', self.state_dim)
#             us = ca.MX.sym('u', self.input_dim)
#             Ms = ca.MX.sym('M', self.nv, self.nv)
#             Gs = ca.MX.sym('G', self.nv, 1)
#             xn = ca.MX(A) @ xs + ca.MX(B) @ us
#             self._f = ca.Function('f', [xs,us,Ms,Gs],[xn],
#                                   ['x','u','M','CG'],['x_next'])

#         def get_casadi_func(self): return self._f

#     dyn = DummyDyn()
#     cost_cfg = {'target_state_selection': {'indices': [0, 1]}}
#     cost = ParametricStageCost(dyn.state_dim, dyn.input_dim, cost_cfg)

#     z_dim = dyn.state_dim + dyn.input_dim
#     P_ub  = np.zeros((dyn.input_dim, z_dim))
#     P_ub[:, dyn.state_dim:] = np.eye(dyn.input_dim)
#     cons  = [
#         PolytopeConstraint(P_ub,  np.ones((dyn.input_dim, 1)) * 5.0, "u upper"),
#         PolytopeConstraint(-P_ub, np.ones((dyn.input_dim, 1)) * 5.0, "u lower"),
#     ]

#     e   = 4
#     kkt = KKTBuilder(dyn, cost, cons, e=e)

#     opt_cfg = {
#         'solver'                  : 'scipy_slsqp',
#         'max_iterations'          : 300,
#         'num_initial_guesses'     : 2,
#         'initial_guesses_strategy': 'random',
#         'random_init_scale'       : 0.05,
#         'l1_reg_lambda'           : 1e-4,
#         'l1_reg_nu'               : 0.0,
#         'tol'                     : 1e-8,
#     }

#     optimizer = IOCOptimizer(kkt, cost, opt_cfg)

#     np.random.seed(42)
#     sd, ud, nv, td = dyn.state_dim, dyn.input_dim, dyn.nv, cost.target_dim

#     Xm = np.random.randn(sd, e+1) * 0.1
#     Um = np.random.randn(ud, e)   * 0.3
#     ys = np.random.randn(td, e)   * 0.1

#     M_data  = KKTBuilder.prepare_M_data([np.eye(nv)] * e)
#     CG_data = KKTBuilder.prepare_CG_data([np.zeros(nv)] * e)

#     theta_s, nu_s, lam_s, obj = optimizer.learn_parameters(
#         Xm, Um, ys, M_data, CG_data
#     )

#     print(f"학습된 theta (처음 5개): {theta_s[:5].round(6)}")
#     print(f"최종 objective        : {obj:.4e}")
#     print(f"lambda_star shape     : {lam_s.shape}  (기대: ({e},{kkt.num_total_constraints}))")

#     optimizer.analyze_result(theta_s, lam_s)
#     print("\n✅ IOCOptimizer 테스트 완료")
