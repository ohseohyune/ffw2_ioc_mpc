"""
src/ioc/data_processor.py

MuJoCo 시연 데이터를 IOC 학습에 적합한 형태로 전처리하는 클래스.

변경 사항 (v2):
    - self.input_dim을 system_params['system']['input_dimension'] 에서 가져옴
      (기존 하드코딩 14 → yaml 주입 31)
    - _validate_raw_inputs 은 self.input_dim 기반이므로 별도 수정 불필요

처리 순서:
    1. raw qpos + qvel → 상태 벡터 x = [qpos, qvel] 결합
    2. 노이즈 필터링 (Savitzky-Golay / UKF / none)
    3. 고정 길이 세그먼트 추출

출력 형식 (논문 Eq.3 기준):
    states_segment : (e+1, state_dim)   x(k), x(k+1), ..., x(k+e)
    inputs_segment : (e,   input_dim)   u(k), u(k+1), ..., u(k+e-1)
"""

import numpy as np
import scipy.signal
from typing import Tuple, Dict, Any, Optional
import warnings


class DataProcessor:
    """
    MuJoCo 시연 데이터를 IOC 학습용 세그먼트로 변환합니다.

    dynamics.py 연동:
        - state_dim  = nq + nv  (DynamicsModel.state_dim = 75)
        - input_dim  = 31       (DynamicsModel.input_dim, yaml에서 로드)
        - x = ca.vertcat(qpos, qvel) 순서와 일치하도록 np.hstack([qpos, qvel]) 사용

    stage_cost.py 연동:
        - target_state_indices 가 유효하도록 qpos 부분이 x[0:nq] 에 위치함을 보장
    """

    def __init__(self, system_params: Dict[str, Any], processing_config: Dict[str, Any]):
        """
        Args:
            system_params (dict):
                configs/system_params.yaml 로드 결과.
                필수 키 구조:
                    system:
                        state_dimension: 75   # nq + nv
                        input_dimension: 31   # ← yaml에서 주입 (기존 하드코딩 14 제거)
                        time_step: 0.02
                        nq: 38
                        nv: 37

            processing_config (dict):
                configs/experiment_params.yaml의 'processing' 섹션.
                필수 키:
                    segment_strategy       : 'fixed_length'
                    segment_length_seconds : 1.2
                    filter_type            : 'savgol' | 'none' | 'ukf'
                선택 키 (savgol):
                    savgol_window_length   : 11  (홀수)
                    savgol_polyorder       : 3
                선택 키 (fixed_length):
                    segment_start_seconds  : 0.0  (기본: 궤적 시작)
        """
        sys = system_params['system']
        self.state_dim : int   = sys['state_dimension']   # 75
        self.input_dim : int   = sys['input_dimension']   # 31 ← yaml 주입
        self.dt        : float = sys['time_step']         # 0.02
        self.nq        : int   = sys['nq']                # 38
        self.nv        : int   = sys['nv']                # 37

        # 차원 일관성 검증
        assert self.state_dim == self.nq + self.nv, (
            f"state_dimension({self.state_dim}) != nq({self.nq}) + nv({self.nv}). "
            "system_params.yaml을 확인하세요."
        )

        self.cfg = processing_config

        print("=" * 60)
        print("DataProcessor 초기화 완료")
        print(f"  state_dim  : {self.state_dim}  (nq={self.nq}, nv={self.nv})")
        print(f"  input_dim  : {self.input_dim}  ← system_params.yaml 주입")
        print(f"  dt         : {self.dt} s")
        print(f"  filter     : {self.cfg.get('filter_type', 'none')}")
        print(f"  segment    : {self.cfg.get('segment_strategy', 'fixed_length')} "
              f"({self.cfg.get('segment_length_seconds', 1.2)} s)")
        print("=" * 60)

    # ================================================================
    # 공개 인터페이스
    # ================================================================

    def process_demonstration(
        self,
        raw_qpos   : np.ndarray,
        raw_qvel   : np.ndarray,
        raw_inputs : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 시연 궤적을 전처리하여 IOC 학습용 세그먼트를 반환합니다.

        Args:
            raw_qpos   : (N, nq=38)         MuJoCo qpos 궤적
            raw_qvel   : (N, nv=37)         MuJoCo qvel 궤적
            raw_inputs : (N, input_dim=31)  data.ctrl 전체 궤적
                         * u[k]가 x[k] → x[k+1] 전이에 사용된다고 가정

        Returns:
            states_segment : (e+1, state_dim)  x(k) ~ x(k+e)
            inputs_segment : (e,   input_dim)  u(k) ~ u(k+e-1)
        """
        self._validate_raw_inputs(raw_qpos, raw_qvel, raw_inputs)

        # ── 1. x = [qpos | qvel] 결합 ────────────────────────────────
        raw_states = np.hstack([raw_qpos, raw_qvel])   # (N, 75)

        # ── 2. 필터링 ─────────────────────────────────────────────────
        filtered_states = self._apply_filter(raw_states, label='states')
        filtered_inputs = self._apply_filter(raw_inputs, label='inputs')

        # ── 3. 세그먼트 추출 ──────────────────────────────────────────
        states_seg, inputs_seg = self._segment(filtered_states, filtered_inputs)

        return states_seg, inputs_seg

    def process_batch(
        self,
        demonstrations: list,
    ) -> list:
        """
        여러 시연을 일괄 처리합니다.

        Args:
            demonstrations: [
                {'qpos': np.ndarray, 'qvel': np.ndarray, 'inputs': np.ndarray},
                ...
            ]

        Returns:
            processed: [
                {'states': np.ndarray (e+1, state_dim),
                 'inputs': np.ndarray (e,   input_dim)},
                ...
            ]
        """
        processed = []
        for idx, demo in enumerate(demonstrations):
            try:
                s, u = self.process_demonstration(
                    demo['qpos'], demo['qvel'], demo['inputs']
                )
                processed.append({'states': s, 'inputs': u})
                print(f"  [Demo {idx+1}/{len(demonstrations)}] OK  "
                      f"states={s.shape}, inputs={u.shape}")
            except Exception as ex:
                warnings.warn(f"  [Demo {idx+1}] 처리 실패: {ex}")
        return processed

    # ================================================================
    # 내부 메서드
    # ================================================================

    def _validate_raw_inputs(
        self,
        raw_qpos   : np.ndarray,
        raw_qvel   : np.ndarray,
        raw_inputs : np.ndarray,
    ) -> None:
        """입력 데이터 형상 및 차원 검증."""
        N = raw_qpos.shape[0]

        if raw_qpos.shape != (N, self.nq):
            raise ValueError(f"raw_qpos shape {raw_qpos.shape} != (N, {self.nq})")
        if raw_qvel.shape != (N, self.nv):
            raise ValueError(f"raw_qvel shape {raw_qvel.shape} != (N, {self.nv})")
        if raw_inputs.shape[0] != N:
            raise ValueError(
                f"raw_inputs 길이({raw_inputs.shape[0]}) != "
                f"raw_qpos 길이({N}). 동일한 N이어야 합니다."
            )
        if raw_inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"raw_inputs dim({raw_inputs.shape[1]}) != input_dim({self.input_dim})\n"
                f"  → input_traj.npy shape이 (N, {raw_inputs.shape[1]})인데 "
                f"system_params.yaml의 input_dimension={self.input_dim}과 불일치합니다."
            )

        min_required = int(self.cfg.get('segment_length_seconds', 1.2) / self.dt) + 1
        if N < min_required:
            raise ValueError(
                f"데이터 길이({N} steps = {N*self.dt:.2f}s)가 "
                f"요청된 세그먼트({min_required} steps)보다 짧습니다."
            )

    def _apply_filter(self, data: np.ndarray, label: str = '') -> np.ndarray:
        """각 열(차원)에 독립적으로 필터를 적용합니다."""
        filter_type = self.cfg.get('filter_type', 'none')

        if filter_type == 'savgol':
            return self._savgol_filter(data, label)
        elif filter_type == 'ukf':
            warnings.warn(
                f"[{label}] UKF 필터가 선택되었으나 아직 구현되지 않았습니다. "
                "원본 데이터를 반환합니다.",
                UserWarning
            )
            return data.copy()
        elif filter_type == 'none':
            return data.copy()
        else:
            warnings.warn(f"[{label}] 알 수 없는 filter_type='{filter_type}'. 원본 반환.")
            return data.copy()

    def _savgol_filter(self, data: np.ndarray, label: str = '') -> np.ndarray:
        """Savitzky-Golay 필터를 모든 열에 적용합니다."""
        N, dim = data.shape
        window = self.cfg.get('savgol_window_length', 11)
        poly   = self.cfg.get('savgol_polyorder', 3)

        if window % 2 == 0:
            window -= 1
        if window > N:
            window = N if N % 2 == 1 else N - 1
            if window < 3:
                warnings.warn(
                    f"[{label}] 데이터({N})가 너무 짧아 SavGol 필터 적용 불가. 원본 반환."
                )
                return data.copy()
        poly = min(poly, window - 1)
        if poly < 1:
            poly = 1

        filtered = np.zeros_like(data)
        for col in range(dim):
            filtered[:, col] = scipy.signal.savgol_filter(
                data[:, col], window_length=window, polyorder=poly
            )
        return filtered

    def _segment(
        self,
        states : np.ndarray,
        inputs : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """세그먼트 전략에 따라 데이터를 잘라 반환합니다."""
        strategy = self.cfg.get('segment_strategy', 'fixed_length')

        if strategy == 'fixed_length':
            return self._fixed_length_segment(states, inputs)
        else:
            raise NotImplementedError(
                f"segment_strategy='{strategy}'은 아직 구현되지 않았습니다. "
                "'fixed_length'를 사용하세요."
            )

    def _fixed_length_segment(
        self,
        states : np.ndarray,
        inputs : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        논문 Eq.(3)의 세그먼트 추출:
            x(k), x(k+1), ..., x(k+e)    → states_segment  (e+1 행)
            u(k), u(k+1), ..., u(k+e-1)  → inputs_segment  (e   행)
        """
        length_sec = self.cfg.get('segment_length_seconds', 1.2)
        start_sec  = self.cfg.get('segment_start_seconds',  0.0)

        e = int(round(length_sec / self.dt))
        k = int(round(start_sec  / self.dt))
        N = states.shape[0]

        max_e = min(N - k - 1, inputs.shape[0] - k)

        if e > max_e:
            warnings.warn(
                f"요청 세그먼트({e} steps = {e*self.dt:.2f}s)가 "
                f"가용 데이터({max_e} steps)를 초과합니다. "
                f"{max_e} steps로 조정합니다."
            )
            e = max_e

        if e <= 0:
            raise ValueError(
                f"유효한 세그먼트를 추출할 수 없습니다. "
                f"start={k}, N={N}, 요청 e={int(round(length_sec/self.dt))}."
            )

        states_seg = states[k : k + e + 1, :]
        inputs_seg = inputs[k : k + e,     :]

        print(f"  ✔ 세그먼트 추출: start={k}({k*self.dt:.3f}s), "
              f"e={e} steps ({e*self.dt:.3f}s)")
        print(f"    states : {states_seg.shape}  inputs : {inputs_seg.shape}")

        return states_seg, inputs_seg

    # ================================================================
    # 유틸리티
    # ================================================================

    def split_state(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """상태 배열을 qpos와 qvel로 분리합니다."""
        return states[:, :self.nq], states[:, self.nq:]

    def compute_segment_length(self) -> Tuple[int, float]:
        """현재 설정의 세그먼트 스텝 수와 실제 시간(초) 반환."""
        e = int(round(self.cfg.get('segment_length_seconds', 1.2) / self.dt))
        return e, e * self.dt


# ====================================================================
# 동작 확인 (직접 실행 시)
# ====================================================================
if __name__ == "__main__":
    system_params = {
        'system': {
            'state_dimension': 75,
            'input_dimension': 31,   # ← 실제 데이터 shape
            'time_step': 0.02,
            'nq': 38,
            'nv': 37,
        }
    }

    processing_config = {
        'segment_strategy':        'fixed_length',
        'segment_length_seconds':  1.2,
        'segment_start_seconds':   0.0,
        'filter_type':             'savgol',
        'savgol_window_length':    11,
        'savgol_polyorder':        3,
    }

    proc = DataProcessor(system_params, processing_config)

    # 실제 데이터 shape에 맞는 더미 궤적 (1136 steps = 22.72s)
    N = 1136
    nq, nv, nu = 38, 37, 31

    np.random.seed(42)
    raw_qpos   = np.cumsum(np.random.randn(N, nq) * 0.005, axis=0)
    raw_qvel   = np.random.randn(N, nv) * 0.02
    raw_inputs = np.random.randn(N, nu) * 5.0   # 31차원

    states_seg, inputs_seg = proc.process_demonstration(raw_qpos, raw_qvel, raw_inputs)

    e, e_sec = proc.compute_segment_length()
    print(f"\n예상 e={e} ({e_sec:.3f}s)")
    print(f"states_seg shape : {states_seg.shape}  (기대: ({e+1}, 75))")
    print(f"inputs_seg shape : {inputs_seg.shape}  (기대: ({e},   31))")
    print("\n✅ DataProcessor (input_dim=31) 테스트 통과")