"""
run_mpc_with_learned_params.py

IOC 학습 결과(configs/learned_params)에 저장된 theta_star.npy를 사용해
MPCController를 1회 실행하는 오프라인 테스트 스크립트.

사용 예시:
    python -m ffw2_ioc_mpc.mpc.run_mpc_with_learned_params \
        --learned_dir configs/learned_params \
        --data_dir data/raw/episode_000_20260223_204925 \
        --step_idx 0

동작 개요:
    1) theta_star.npy 로드
    2) (선택) active_constraint_indices.npy 로 식별 제약 재구성
    3) 에피소드에서 x(k), ys(k), M/CG(k:k+N_p-1) 로드
    4) MPCController.set_learned_parameters(theta*)
    5) get_control_action(...) 1회 호출
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ffw2_ioc_mpc.system_models.dynamics import DynamicsModel
from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
from ffw2_ioc_mpc.ioc.data_processor import DataProcessor
from ffw2_ioc_mpc.ioc.constraint_builder import ConstraintBuilder
from ffw2_ioc_mpc.mpc.controller import MPCController


DEFAULT_PROCESSING_CONFIG = {
    "segment_strategy": "fixed_length",
    "segment_length_seconds": 0.6,
    "segment_start_seconds": 0.0,
    "filter_type": "savgol",
    "savgol_window_length": 11,
    "savgol_polyorder": 3,
}

DEFAULT_BUILDER_CONFIG = {
    "include_domain_knowledge_constraints": True,
    "include_convex_hull_constraints": False,
    "input_torque_limits_u": {
        "min": -100.0,
        "max": 100.0,
    },
}

DEFAULT_MPC_CONFIG = {
    "prediction_horizon": 10,
    "control_horizon": 1,
    "solver_name": "ipopt",
    "ipopt_options": {
        "print_level": 0,
        "max_iter": 100,
        "tol": 1e-4,
    },
    "enable_terminal_cost": False,
    "terminal_cost_weight": 1.0,
}

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """중첩 dict를 재귀적으로 병합합니다."""
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def resolve_input_actuator_indices(sys_cfg: dict, raw_input_dim: int) -> list[int]:
    indices = sys_cfg.get("input_actuator_indices", None)
    if indices is not None:
        idx = [int(i) for i in indices]
    else:
        input_dim = int(sys_cfg["input_dimension"])
        if input_dim == 7:
            idx = list(range(16, 23))
        elif input_dim == 14:
            idx = list(range(9, 23))
        elif input_dim == raw_input_dim:
            idx = list(range(raw_input_dim))
        else:
            raise ValueError(
                "input_actuator_indices가 없고 input_dimension에 대한 기본 매핑을 결정할 수 없습니다. "
                f"(input_dimension={input_dim}, raw_input_dim={raw_input_dim})"
            )

    if len(idx) != int(sys_cfg["input_dimension"]):
        raise ValueError(
            f"입력 인덱스 개수({len(idx)}) != input_dimension({sys_cfg['input_dimension']})"
        )
    if min(idx) < 0 or max(idx) >= raw_input_dim:
        raise ValueError(
            f"입력 인덱스 범위 오류: [{min(idx)}, {max(idx)}], raw_input_dim={raw_input_dim}"
        )
    return idx


def get_default_data_dir_from_learning_summary(learned_dir: str) -> str | None:
    summary_path = os.path.join(learned_dir, "learning_summary.yaml")
    if not os.path.exists(summary_path):
        return None
    try:
        summary = load_yaml(summary_path)
        data_dir = summary.get("data_dir", None)
        if isinstance(data_dir, str) and data_dir:
            return data_dir
    except Exception as ex:
        warnings.warn(f"learning_summary.yaml 읽기 실패: {ex}")
    return None


def load_episode_data(data_dir: str) -> dict:
    files = {
        "qpos": "qpos_traj.npy",
        "qvel": "qvel_traj.npy",
        "input": "input_traj.npy",
        "M": "M_traj.npy",
        "CG": "CG_traj.npy",
        "ys": "ys_traj.npy",
    }
    data = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {fpath}")
        data[key] = np.load(fpath)

    print("\n[에피소드 데이터 로드 완료]")
    for key, arr in data.items():
        print(f"  {key:6s}: {arr.shape}")
    return data


def get_xml_path_from_project() -> str:
    return os.path.join(
        PROJECT_ROOT,
        "ffw2_ioc_mpc",
        "system_models",
        "mujoco_models",
        "scene_ffw_sg2.xml",
    )


def _slice_horizon_with_pad(arr: np.ndarray, start_idx: int, horizon: int) -> np.ndarray:
    """arr의 첫 축 기준으로 [start_idx : start_idx+horizon) 슬라이스 후 부족분은 마지막 값 반복."""
    n = arr.shape[0]
    if not (0 <= start_idx < n):
        raise IndexError(f"start_idx={start_idx} 범위 오류 (0 <= idx < {n})")

    end_idx = min(start_idx + horizon, n)
    out = arr[start_idx:end_idx].copy()
    if out.shape[0] == 0:
        raise RuntimeError("horizon 슬라이스 결과가 비었습니다.")
    if out.shape[0] < horizon:
        pad_count = horizon - out.shape[0]
        pad = np.repeat(out[-1:, ...], pad_count, axis=0)
        out = np.concatenate([out, pad], axis=0)
        print(f"  [참고] horizon 부족분 {pad_count} step은 마지막 값으로 패딩했습니다.")
    return out


def build_identified_constraints(
    raw: dict,
    system_params: dict,
    processing_config: dict,
    builder_config: dict,
    learned_dir: str,
    use_identified_constraints: bool,
    input_actuator_indices: List[int],
) -> List[Any]:
    """
    학습 결과 폴더의 active_constraint_indices.npy를 읽어 식별된 제약을 재구성합니다.

    active indices는 후보 제약 순서에 대한 인덱스이므로, candidate list를 동일하게 다시 생성해야 합니다.
    """
    if not use_identified_constraints:
        print("\n[제약 재구성] 비활성화됨 (--no_identified_constraints)")
        return []

    idx_path = os.path.join(learned_dir, "active_constraint_indices.npy")
    if not os.path.exists(idx_path):
        print("\n[제약 재구성] active_constraint_indices.npy 없음 → 제약 없이 MPC 실행")
        return []

    active_indices = np.load(idx_path).astype(int).flatten()
    print(f"\n[제약 재구성] active_constraint_indices: {active_indices.tolist()}")
    if active_indices.size == 0:
        print("  식별된 활성 제약 없음 → 제약 없이 MPC 실행")
        return []

    # IOC와 동일한 입력 채널 슬라이스
    raw_input_arm = raw["input"][:, input_actuator_indices]

    data_processor = DataProcessor(system_params, processing_config)
    states_seg, inputs_seg = data_processor.process_demonstration(
        raw_qpos=raw["qpos"],
        raw_qvel=raw["qvel"],
        raw_inputs=raw_input_arm,
    )

    xml_path = get_xml_path_from_project()
    system_params_with_xml = {
        "system": {
            **system_params["system"],
            "model_xml_path": xml_path,
        }
    }
    builder = ConstraintBuilder(system_params_with_xml, builder_config)
    candidates = builder.build_candidate_constraints(
        processed_states=states_seg,
        processed_inputs=inputs_seg,
    )

    if np.any(active_indices < 0) or np.any(active_indices >= len(candidates)):
        raise IndexError(
            "active_constraint_indices.npy에 후보 제약 범위를 벗어난 인덱스가 있습니다. "
            "학습 당시 experiment_params.yaml과 현재 설정이 달라졌을 수 있습니다."
        )

    identified = [candidates[i] for i in active_indices.tolist()]
    print(f"  재구성된 활성 제약 수: {len(identified)}")
    for i, c in zip(active_indices.tolist(), identified):
        print(f"    - [{i}] {c.name} (num_ineq={c.num_ineq})")
    return identified


def build_controller_and_runtime(
    system_params: dict,
    cost_params: dict,
    mpc_config: dict,
    identified_constraints: List[Any],
    theta_star: np.ndarray,
    raw: dict,
    step_idx: int,
    input_actuator_indices: List[int],
) -> Tuple[MPCController, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    DynamicsModel / StageCost / MPCController를 구성하고,
    get_control_action()에 넣을 런타임 입력(x, ys, M_h, CG_h)을 반환합니다.
    """
    sys_cfg = system_params["system"]
    xml_path = get_xml_path_from_project()

    dynamics_model = DynamicsModel(
        dt=sys_cfg["time_step"],
        model_xml_path=xml_path,
        input_dim=sys_cfg["input_dimension"],
        actuator_indices=input_actuator_indices,
    )

    stage_cost = ParametricStageCost(
        state_dim=dynamics_model.state_dim,
        input_dim=dynamics_model.input_dim,
        cost_params_config=cost_params["cost_function"],
    )

    controller = MPCController(
        dynamics_model=dynamics_model,
        stage_cost=stage_cost,
        identified_constraints=identified_constraints,
        mpc_config=mpc_config,
    )
    controller.set_learned_parameters(theta_star)

    n_steps = raw["qpos"].shape[0]
    if not (0 <= step_idx < n_steps):
        raise IndexError(f"--step_idx={step_idx} 범위 오류 (0 ~ {n_steps-1})")

    # 현재 상태 x(k) = [qpos(k), qvel(k)]
    current_state = np.hstack([raw["qpos"][step_idx], raw["qvel"][step_idx]]).astype(float)

    # 목표 ys(k)
    target_ys = np.asarray(raw["ys"][step_idx], dtype=float).flatten()
    if target_ys.shape[0] != stage_cost.target_dim:
        raise ValueError(
            f"ys 차원 불일치: raw['ys'][{step_idx}].shape={raw['ys'][step_idx].shape}, "
            f"target_dim={stage_cost.target_dim}"
        )

    # 예측 구간에 사용할 M, CG (에피소드에서 꺼내오고 부족하면 패딩)
    N_p = int(mpc_config.get("prediction_horizon", 10))
    M_data_mpc = _slice_horizon_with_pad(raw["M"], step_idx, N_p)
    CG_data_mpc = _slice_horizon_with_pad(raw["CG"], step_idx, N_p)

    return controller, current_state, target_ys, M_data_mpc, CG_data_mpc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 IOC 파라미터(theta*)로 MPC 1회 실행")
    parser.add_argument(
        "--learned_dir",
        type=str,
        default=os.path.join("configs", "learned_params"),
        help="IOC 학습 결과 디렉토리 (theta_star.npy 포함)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="에피소드 데이터 디렉토리. 생략 시 learned_dir/learning_summary.yaml의 data_dir 사용 시도",
    )
    parser.add_argument(
        "--step_idx",
        type=int,
        default=0,
        help="에피소드에서 MPC를 실행할 기준 시점 인덱스 k",
    )
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=None,
        help="MPC 예측 구간 N_p (없으면 기본값/experiment_params 사용)",
    )
    parser.add_argument(
        "--control_horizon",
        type=int,
        default=None,
        help="MPC 제어 구간 N_c (없으면 기본값/experiment_params 사용)",
    )
    parser.add_argument(
        "--mpc_max_iter",
        type=int,
        default=None,
        help="MPC IPOPT max_iter override",
    )
    parser.add_argument(
        "--mpc_tol",
        type=float,
        default=None,
        help="MPC IPOPT tol override (예: 1e-4)",
    )
    parser.add_argument(
        "--ipopt_print_level",
        type=int,
        default=None,
        help="MPC IPOPT print_level override",
    )
    parser.add_argument(
        "--enable_terminal_cost",
        action="store_true",
        help="종단 비용 사용",
    )
    parser.add_argument(
        "--terminal_cost_weight",
        type=float,
        default=None,
        help="종단 비용 가중치 (enable_terminal_cost 사용 시)",
    )
    parser.add_argument(
        "--no_identified_constraints",
        action="store_true",
        help="active_constraint_indices.npy를 무시하고 제약 없는 MPC로 실행",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="MPC 1회 실행 결과를 learned_dir에 npy/yaml로 저장",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  학습된 theta*로 MPC 1회 실행")
    print("=" * 70)

    learned_dir = resolve_project_path(args.learned_dir)
    if not os.path.isdir(learned_dir):
        raise FileNotFoundError(f"learned_dir를 찾을 수 없습니다: {learned_dir}")

    if args.data_dir is None:
        inferred_data_dir = get_default_data_dir_from_learning_summary(learned_dir)
        if inferred_data_dir is None:
            raise ValueError(
                "--data_dir를 지정하세요. (learned_dir/learning_summary.yaml에서 data_dir를 찾지 못함)"
            )
        data_dir = inferred_data_dir
        print(f"[data_dir 자동추론] {data_dir}")
    else:
        data_dir = resolve_project_path(args.data_dir)

    config_dir = os.path.join(PROJECT_ROOT, "configs")
    system_params = load_yaml(os.path.join(config_dir, "system_params.yaml"))
    cost_params = load_yaml(os.path.join(config_dir, "cost_params.yaml"))

    exp_path = os.path.join(config_dir, "experiment_params.yaml")
    exp_params = load_yaml(exp_path) if os.path.exists(exp_path) else {}
    processing_config = deep_merge_dict(
        DEFAULT_PROCESSING_CONFIG, exp_params.get("processing", {})
    )
    builder_config = deep_merge_dict(
        DEFAULT_BUILDER_CONFIG, exp_params.get("constraint_builder", {})
    )

    # experiment_params.yaml에 mpc 섹션이 있으면 반영, 없으면 기본값
    mpc_config = deep_merge_dict(DEFAULT_MPC_CONFIG, exp_params.get("mpc", {}))
    if args.prediction_horizon is not None:
        mpc_config["prediction_horizon"] = args.prediction_horizon
    if args.control_horizon is not None:
        mpc_config["control_horizon"] = args.control_horizon
    if args.mpc_max_iter is not None:
        mpc_config.setdefault("ipopt_options", {})
        mpc_config["ipopt_options"]["max_iter"] = args.mpc_max_iter
    if args.mpc_tol is not None:
        mpc_config.setdefault("ipopt_options", {})
        mpc_config["ipopt_options"]["tol"] = args.mpc_tol
    if args.ipopt_print_level is not None:
        mpc_config.setdefault("ipopt_options", {})
        mpc_config["ipopt_options"]["print_level"] = args.ipopt_print_level
    if args.enable_terminal_cost:
        mpc_config["enable_terminal_cost"] = True
    if args.terminal_cost_weight is not None:
        mpc_config["terminal_cost_weight"] = args.terminal_cost_weight

    theta_path = os.path.join(learned_dir, "theta_star.npy")
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"theta_star.npy를 찾을 수 없습니다: {theta_path}")
    theta_star = np.load(theta_path).astype(float).flatten()

    print("\n[입력 설정]")
    print(f"  learned_dir : {learned_dir}")
    print(f"  data_dir    : {data_dir}")
    print(f"  step_idx    : {args.step_idx}")
    print(f"  theta_star  : {theta_star.shape}")
    print(f"  MPC config  : {mpc_config}")

    raw = load_episode_data(data_dir)
    input_actuator_indices = resolve_input_actuator_indices(
        sys_cfg=system_params["system"],
        raw_input_dim=raw["input"].shape[1],
    )
    print(f"  input_actuator_indices: {input_actuator_indices}")

    identified_constraints = build_identified_constraints(
        raw=raw,
        system_params=system_params,
        processing_config=processing_config,
        builder_config=builder_config,
        learned_dir=learned_dir,
        use_identified_constraints=(not args.no_identified_constraints),
        input_actuator_indices=input_actuator_indices,
    )

    controller, current_state, target_ys, M_data_mpc, CG_data_mpc = build_controller_and_runtime(
        system_params=system_params,
        cost_params=cost_params,
        mpc_config=mpc_config,
        identified_constraints=identified_constraints,
        theta_star=theta_star,
        raw=raw,
        step_idx=args.step_idx,
        input_actuator_indices=input_actuator_indices,
    )

    print("\n[MPC 실행 입력 요약]")
    print(f"  current_state : {current_state.shape}")
    print(f"  target_ys     : {target_ys.shape}  값={target_ys.round(6)}")
    print(f"  M_data_mpc    : {M_data_mpc.shape}")
    print(f"  CG_data_mpc   : {CG_data_mpc.shape}")

    optimal_u, info = controller.get_control_action(
        current_state=current_state,
        target_ys=target_ys,
        M_data_mpc=M_data_mpc,
        CG_data_mpc=CG_data_mpc,
    )

    Q_val, R_val = controller.get_cost_matrices()

    print("\n" + "=" * 70)
    print("  MPC 1회 실행 결과")
    print("=" * 70)
    print(f"  optimal_u shape     : {optimal_u.shape}")
    print(f"  optimal_u           : {np.round(optimal_u, 6)}")
    print(f"  solver success      : {info.get('success', False)}")
    print(f"  return_status       : {info.get('return_status', '')}")
    print(f"  objective f         : {info.get('f', np.nan):.6e}")
    print(f"  solve_time_ms       : {info.get('solve_time_ms', np.nan):.2f}")
    print(f"  g_max (<=0 ideal)   : {info.get('g_max', np.nan):.6e}")
    print(f"  Q shape             : {Q_val.shape}")
    print(f"  R shape             : {R_val.shape}")
    print(f"  R diag              : {np.diag(R_val).round(6)}")
    print(f"  trace(R)            : {float(np.trace(R_val)):.6f}")

    if args.save_result:
        os.makedirs(learned_dir, exist_ok=True)
        np.save(os.path.join(learned_dir, "mpc_one_step_u.npy"), optimal_u)
        result_yaml = {
            "step_idx": int(args.step_idx),
            "solver_success": bool(info.get("success", False)),
            "return_status": str(info.get("return_status", "")),
            "objective_f": float(info.get("f", np.nan)),
            "solve_time_ms": float(info.get("solve_time_ms", np.nan)),
            "g_max": float(info.get("g_max", np.nan)),
            "data_dir": data_dir,
            "learned_dir": learned_dir,
            "num_identified_constraints": int(len(identified_constraints)),
            "prediction_horizon": int(mpc_config.get("prediction_horizon", 0)),
            "control_horizon": int(mpc_config.get("control_horizon", 0)),
        }
        with open(os.path.join(learned_dir, "mpc_one_step_result.yaml"), "w") as f:
            yaml.dump(result_yaml, f, allow_unicode=True, default_flow_style=False)
        print(f"\n[결과 저장] {os.path.join(learned_dir, 'mpc_one_step_u.npy')}")
        print(f"[결과 저장] {os.path.join(learned_dir, 'mpc_one_step_result.yaml')}")

    print("\n완료.")


if __name__ == "__main__":
    main()
