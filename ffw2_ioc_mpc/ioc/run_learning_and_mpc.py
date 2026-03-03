"""
scripts/run_learning_and_mpc.py

IOC 학습 파이프라인 전체 실행 스크립트.

실행 순서:
    1. 설정 파일 로드 (system_params.yaml, cost_params.yaml, experiment_params.yaml)
    2. DynamicsModel / DataProcessor / ParametricStageCost 초기화
    3. 시연 데이터 로드 및 전처리
    4. ConstraintBuilder → 후보 제약 조건 생성
    5. KKTBuilder → KKT 잔차 함수 빌드
    6. IOCOptimizer → θ*, λ*, ν* 학습
    7. ConstraintIdentifier → 활성 제약 조건 식별
    8. 결과 저장 (configs/learned_params/)

사용법:
    python -m ffw2_ioc_mpc.ioc.run_learning_and_mpc \
        --data_dir  data/raw/episode_000_20260224_153814 \
        --output_dir configs/learned_params

"""

import os
import sys
import argparse
import yaml
import numpy as np

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ffw2_ioc_mpc.system_models.dynamics import DynamicsModel
from ffw2_ioc_mpc.ioc.data_processor import DataProcessor
from ffw2_ioc_mpc.cost_functions.stage_cost import ParametricStageCost
from ffw2_ioc_mpc.ioc.constraint_builder import ConstraintBuilder
from ffw2_ioc_mpc.ioc.kkt_builder import KKTBuilder
from ffw2_ioc_mpc.ioc.optimizer import IOCOptimizer
from ffw2_ioc_mpc.ioc.constraint_identifier import ConstraintIdentifier


# ====================================================================
# 설정 파일 기본값 (experiment_params.yaml 없을 때 사용)
# ====================================================================

DEFAULT_PROCESSING_CONFIG = {
    'segment_strategy'      : 'fixed_length',
    'segment_length_seconds': 0.6,
    'segment_start_seconds' : 0.0,
    'filter_type'           : 'savgol',
    'savgol_window_length'  : 11,
    'savgol_polyorder'      : 3,
}

DEFAULT_BUILDER_CONFIG = {
    'include_domain_knowledge_constraints': True,
    'include_convex_hull_constraints'     : False,
    'input_torque_limits_u': {
        'min': -100.0,
        'max':  100.0,
    },
}

DEFAULT_OPTIMIZER_CONFIG = {
    'solver'                  : 'ipopt',
    'max_iterations'          : 5000,
    'num_initial_guesses'     : 3,
    'initial_guesses_strategy': 'random',
    'random_init_scale'       : 0.01,
    'l1_reg_lambda'           : 1e-4,
    'l1_reg_nu'               : 0.0,
    'tol'                     : 1e-6,
    'ipopt_print_level'       : 0,
    'print_frequency_iter'    : 10,
    # large IOC NLP에서는 exact Hessian 빌드가 매우 느릴 수 있으므로 기본값은 L-BFGS
    'ipopt_hessian_approximation': 'limited-memory',
    'ipopt_sb'                : 'yes',
    'casadi_print_time'       : False,
    'casadi_verbose'          : False,
    'casadi_verbose_init'     : False,
    # 정확한 % 진행률은 아니고, solver 호출 중 스피너 + 경과시간 표시
    'show_solver_progress'    : True,
    # C/C++ 레벨 stdout/stderr까지 완전히 숨기고 싶을 때 사용 (스피너도 숨겨질 수 있음)
    'suppress_casadi_output'  : False,
}

DEFAULT_IDENTIFIER_CONFIG = {
    'threshold_lambda': 1e-3,
    'aggregation'     : 'sum_all',
    'verbose'         : True,
}


# ====================================================================
# 헬퍼
# ====================================================================

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_episode_data(data_dir: str) -> dict:
    """에피소드 디렉토리에서 npy 파일들을 로드합니다."""
    files = {
        'qpos' : 'qpos_traj.npy',
        'qvel' : 'qvel_traj.npy',
        'input': 'input_traj.npy',
        'M'    : 'M_traj.npy',
        'CG'   : 'CG_traj.npy',
        'ys'   : 'ys_traj.npy',
    }
    data = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {fpath}")
        data[key] = np.load(fpath)

    print("\n[데이터 로드 완료]")
    for key, arr in data.items():
        print(f"  {key:6s}: {arr.shape}")
    return data


def resolve_input_actuator_indices(sys_cfg: dict, raw_input_dim: int) -> list[int]:
    """
    IOC 학습에서 사용할 input_traj 채널 인덱스를 결정합니다.
    """
    indices = sys_cfg.get("input_actuator_indices", None)
    if indices is not None:
        idx = [int(i) for i in indices]
    else:
        input_dim = int(sys_cfg["input_dimension"])
        if input_dim == 7:
            idx = list(range(16, 23))      # right arm motor only
        elif input_dim == 14:
            idx = list(range(9, 23))       # both arms
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


def prepare_kkt_inputs(
    states_seg : np.ndarray,   # (e+1, 75)
    inputs_seg : np.ndarray,   # (e,   31)
    M_traj     : np.ndarray,   # (N, 37, 37)
    CG_traj    : np.ndarray,   # (N, 37)
    ys_traj    : np.ndarray,   # (N, target_dim)
    start_idx  : int = 0,
) -> dict:
    """
    DataProcessor 출력과 raw traj 데이터를 KKTBuilder / IOCOptimizer 입력 형식으로 변환합니다.

    Returns:
        dict with keys: Xm_data, Um_data, ys_data, M_data, CG_data
    """
    e = states_seg.shape[0] - 1   # 세그먼트 스텝 수

    # M, CG: 세그먼트 구간 슬라이스
    k = start_idx
    M_seg  = M_traj [k : k + e]   # (e, 37, 37)
    CG_seg = CG_traj[k : k + e]   # (e, 37)
    ys_seg = ys_traj[k : k + e]   # (e, target_dim)

    # KKTBuilder 형식으로 변환
    M_data  = KKTBuilder.prepare_M_data([M_seg[i]  for i in range(e)])  # (37, 37*e)
    CG_data = KKTBuilder.prepare_CG_data([CG_seg[i] for i in range(e)]) # (37, e)

    Xm_data = states_seg.T    # (75, e+1)
    Um_data = inputs_seg.T    # (input_dim, e)
    ys_data = ys_seg.T        # (target_dim, e)

    print(f"\n[KKT 입력 준비 완료]  e={e} steps")
    print(f"  Xm_data : {Xm_data.shape}")
    print(f"  Um_data : {Um_data.shape}")
    print(f"  ys_data : {ys_data.shape}")
    print(f"  M_data  : {M_data.shape}")
    print(f"  CG_data : {CG_data.shape}")

    return {
        'Xm_data': Xm_data,
        'Um_data': Um_data,
        'ys_data': ys_data,
        'M_data' : M_data,
        'CG_data': CG_data,
        'e'      : e,
    }


# ====================================================================
# 메인 파이프라인
# ====================================================================

def run_pipeline(args):
    print("=" * 65)
    print("  IOC 학습 파이프라인 시작")
    print("=" * 65)

    # 상대경로 인자를 프로젝트 루트 기준으로 해석 (실행 위치와 무관하게 동작)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(PROJECT_ROOT, args.output_dir)

    # ── 1. 설정 파일 로드 ───────────────────────────────────────────
    config_dir = os.path.join(PROJECT_ROOT, "configs")

    system_params = load_yaml(os.path.join(config_dir, "system_params.yaml"))
    cost_params   = load_yaml(os.path.join(config_dir, "cost_params.yaml"))

    exp_params_path = os.path.join(config_dir, "experiment_params.yaml")
    if os.path.exists(exp_params_path):
        exp_params = load_yaml(exp_params_path) or {}   # ← None 방어
        processing_config  = exp_params.get('processing',               DEFAULT_PROCESSING_CONFIG)
        builder_config     = exp_params.get('constraint_builder',       DEFAULT_BUILDER_CONFIG)
        optimizer_config   = exp_params.get('optimizer',                DEFAULT_OPTIMIZER_CONFIG)
        identifier_config  = exp_params.get('constraint_identification', DEFAULT_IDENTIFIER_CONFIG)
    else:
        print("[경고] experiment_params.yaml 없음 → 기본값 사용")
        processing_config  = DEFAULT_PROCESSING_CONFIG
        builder_config     = DEFAULT_BUILDER_CONFIG
        optimizer_config   = DEFAULT_OPTIMIZER_CONFIG
        identifier_config  = DEFAULT_IDENTIFIER_CONFIG

    sys_cfg = system_params['system']
    print(f"\n[설정]")
    print(f"  state_dim  : {sys_cfg['state_dimension']}")
    print(f"  input_dim  : {sys_cfg['input_dimension']}")
    print(f"  time_step  : {sys_cfg['time_step']}")
    print(f"  target_idx : {cost_params['cost_function']['target_state_selection']['indices']}")

    # ── 2. 모델 초기화 ──────────────────────────────────────────────
    BASE_DIR = os.path.dirname(__file__)
    xml_path = os.path.join(BASE_DIR, "..", "system_models", "mujoco_models", "scene_ffw_sg2.xml")
    # xml_path = sys_cfg['model_xml_path']
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(PROJECT_ROOT, xml_path)

    raw = load_episode_data(args.data_dir)
    input_actuator_indices = resolve_input_actuator_indices(
        sys_cfg=sys_cfg,
        raw_input_dim=raw["input"].shape[1],
    )

    dynamics_model = DynamicsModel(
        dt             = sys_cfg['time_step'],
        model_xml_path = xml_path,
        input_dim      = sys_cfg['input_dimension'],
        actuator_indices=input_actuator_indices,
    )

    data_processor = DataProcessor(system_params, processing_config)

    stage_cost = ParametricStageCost(
        state_dim         = dynamics_model.state_dim,
        input_dim         = dynamics_model.input_dim,
        cost_params_config= cost_params['cost_function'],
    )

    # ── 3. 데이터 로드 및 전처리 ───────────────────────────────────
    raw['input'] = raw['input'][:, input_actuator_indices]
    print(f"  [입력 채널] actuator indices: {input_actuator_indices}")
    
    states_seg, inputs_seg = data_processor.process_demonstration(
        raw_qpos   = raw['qpos'],    # (N, 38)
        raw_qvel   = raw['qvel'],    # (N, 37)
        raw_inputs = raw['input'],   # (N, input_dim)
    )

    # segment_start_seconds → start_idx 계산
    start_sec = processing_config.get('segment_start_seconds', 0.0)
    start_idx = int(round(start_sec / sys_cfg['time_step']))

    kkt_inputs = prepare_kkt_inputs(
        states_seg = states_seg,
        inputs_seg = inputs_seg,
        M_traj     = raw['M'],
        CG_traj    = raw['CG'],
        ys_traj    = raw['ys'],
        start_idx  = start_idx,
    )
    e = kkt_inputs['e']

    # ── 4. 후보 제약 조건 생성 ─────────────────────────────────────
    # builder_config에 model_xml_path 주입 (ConstraintBuilder가 MuJoCo 로드에 사용)
    # 수정 - xml_path를 system_params에 주입
    system_params_with_xml = {
        'system': {
            **system_params['system'],
            'model_xml_path': xml_path   # ← 이미 계산된 절대경로 덮어쓰기
        }
    }
    constraint_builder = ConstraintBuilder(system_params_with_xml, builder_config)

    candidate_constraints = constraint_builder.build_candidate_constraints(
        processed_states = states_seg,   # (e+1, 75)
        processed_inputs = inputs_seg,   # (e,   31)
    )
    print(constraint_builder.get_constraint_summary(candidate_constraints))

    # ── 5. KKT 함수 빌드 ───────────────────────────────────────────
    kkt_builder = KKTBuilder(
        dynamics_model        = dynamics_model,
        stage_cost            = stage_cost,
        candidate_constraints = candidate_constraints,
        e                     = e,
    )

    # ── 6. IOC 최적화 ──────────────────────────────────────────────
    optimizer = IOCOptimizer(kkt_builder, stage_cost, optimizer_config)

    theta_star, nu_star, lambda_star, final_obj = optimizer.learn_parameters(
        Xm_data = kkt_inputs['Xm_data'],
        Um_data = kkt_inputs['Um_data'],
        ys_data = kkt_inputs['ys_data'],
        M_data  = kkt_inputs['M_data'],
        CG_data = kkt_inputs['CG_data'],
    )

    analysis = optimizer.analyze_result(theta_star, lambda_star)

    # ── 7. 활성 제약 조건 식별 ────────────────────────────────────
    identifier = ConstraintIdentifier(candidate_constraints, identifier_config)
    active_constraints = identifier.identify_active_constraints(lambda_star)

    print(f"\n[활성 제약 조건 식별 완료]")
    for c in active_constraints:
        print(f"  - {c.name}  (num_ineq={c.num_ineq})")

    # ── 8. 결과 저장 ───────────────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "theta_star.npy"),  theta_star)
    np.save(os.path.join(output_dir, "nu_star.npy"),     nu_star)
    np.save(os.path.join(output_dir, "lambda_star.npy"), lambda_star)
    np.save(os.path.join(output_dir, "Q_matrix.npy"),    analysis['Q'])
    np.save(os.path.join(output_dir, "R_matrix.npy"),    analysis['R'])

    # 활성 제약 조건 인덱스 저장
    active_indices = [
        i for i, c in enumerate(candidate_constraints)
        if c in active_constraints
    ]
    np.save(os.path.join(output_dir, "active_constraint_indices.npy"),
            np.array(active_indices))

    # 요약 yaml 저장
    summary = {
        'final_objective'       : float(final_obj),
        'R_trace'               : float(analysis['R_trace']),
        'n_active_constraints'  : len(active_constraints),
        'active_constraint_names': [c.name for c in active_constraints],
        'theta_dim'             : int(theta_star.shape[0]),
        'e_steps'               : int(e),
        'data_dir'              : args.data_dir,
    }
    with open(os.path.join(output_dir, "learning_summary.yaml"), 'w') as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)

    print(f"\n[결과 저장 완료] → {os.path.abspath(output_dir)}")
    print(f"  theta_star.npy  : {theta_star.shape}")
    print(f"  Q_matrix.npy    : {analysis['Q'].shape}")
    print(f"  R_matrix.npy    : {analysis['R'].shape}")
    print(f"  최종 objective  : {final_obj:.4e}")
    print(f"  R trace (≈1.0)  : {analysis['R_trace']:.6f}")

    print("\n" + "=" * 65)
    print("  IOC 학습 파이프라인 완료")
    print("=" * 65)

    return theta_star, active_constraints


# ====================================================================
# CLI 진입점
# ====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="IOC 학습 파이프라인 실행"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="에피소드 데이터 디렉토리 경로 (qpos_traj.npy 등이 있는 폴더)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "learned_params"),
        help="학습 결과 저장 디렉토리 (기본: configs/learned_params)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
