from turtle import rt
from xml.parsers.expat import model

import numpy as np
import mujoco
import casadi as ca
import pytest
import os 
from ffw2_ioc_mpc.system_models.dynamics import DynamicsModel # 실제 경로에 따라 수정

# 테스트를 위한 XML 파일 경로 설정 (프로젝트 루트 기준)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_MODEL_XML_PATH = os.path.join(BASE_DIR, "mujoco_models", "ffw_sg2.xml")

DT = 0.01

@pytest.fixture(scope="module")
def mujoco_model_data():
    """테스트를 위해 MuJoCo 모델과 데이터를 로드하는 픽스처."""
    try:
        model = mujoco.MjModel.from_xml_path(TEST_MODEL_XML_PATH)
        data = mujoco.MjData(model)
        return model, data
    except Exception as e:
        pytest.skip(f"MuJoCo 모델 로드 실패: {e}")

@pytest.fixture(scope="module")
def dynamics_model_instance(mujoco_model_data):
    """DynamicsModel 인스턴스를 생성하는 픽스처."""
    return DynamicsModel(dt=DT, model_xml_path=TEST_MODEL_XML_PATH)

def test_dimensions(dynamics_model_instance, mujoco_model_data):
    """상태 및 입력 차원 일치 여부 테스트."""
    model, _ = mujoco_model_data
    
    assert dynamics_model_instance.state_dim == model.nq + model.nv
    # input_dim은 XML에서 torque-controlled actuator 수를 정확히 세서 넣어야 함
    # 현재 코드에서는 14로 고정되어 있음. XML과 일치하는지 확인 필요.
    # 예시: assert dynamics_model_instance.input_dim == model.nu (이건 모든 액츄에이터 포함)
    # 실제로는 XML 파싱해서 motor class="robot_torque"의 개수를 세야 함.
    assert dynamics_model_instance.input_dim == 14 # XML에서 확인된 값

def test_casadi_func_signature(dynamics_model_instance):
    """CasADi 함수의 입력/출력 시그니처 테스트."""
    f_casadi = dynamics_model_instance.get_casadi_func()
    
    assert f_casadi.numel_in(0) == dynamics_model_instance.state_dim # x_sym
    assert f_casadi.numel_in(1) == dynamics_model_instance.input_dim # u_sym
    assert f_casadi.size_in(2) == (dynamics_model_instance.nv, dynamics_model_instance.nv)
    assert f_casadi.size_in(3) == (dynamics_model_instance.nv, 1)
    assert f_casadi.numel_out(0) == dynamics_model_instance.state_dim # x_next

def test_single_step_prediction(dynamics_model_instance, mujoco_model_data):
    """MuJoCo mj_step과 DynamicsModel.predict의 단일 스텝 예측 비교."""
    model, data = mujoco_model_data
    
    # 랜덤 초기 상태 및 입력 설정
    data.qpos[:] = np.random.uniform(-0.5, 0.5, model.nq)
    data.qvel[:] = np.random.uniform(-1.0, 1.0, model.nv)       
    # robot_torque 클래스 액츄에이터에 해당하는 입력만 설정 (XML 확인)
    data.ctrl = np.zeros(model.nu) 
    torque_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor_arm_l_joint{i}') for i in range(1,8)] + \
                     [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor_arm_r_joint{i}') for i in range(1,8)]
    
    input_torques = np.random.uniform(-10, 10, dynamics_model_instance.input_dim)
    for i, act_id in enumerate(torque_act_ids):
        data.ctrl[act_id] = input_torques[i]

    # MuJoCo의 초기 물리 계산 업데이트
    mujoco.mj_fwd(model, data)

    # -------------------------------------------------------------
    # 1. MuJoCo로 한 스텝 시뮬레이션 (Ground Truth)
    # 현재 상태 저장
    x_k_mujoco = np.concatenate([data.qpos, data.qvel])
    
    # MuJoCo 스텝
    mujoco.mj_step(model, data)
    
    # 다음 상태 추출
    x_k_plus_1_mujoco = np.concatenate([data.qpos, data.qvel])
    
    # -------------------------------------------------------------
    # 2. DynamicsModel.predict로 한 스텝 예측 (우리의 모델)
    # 예측을 위해 필요한 M, CG는 이전 상태(x_k_mujoco)에서 계산되어야 함.
    # mj_fwd(model, data)를 다시 호출하여 data.qpos, data.qvel이 x_k_mujoco 상태일 때의 M, CG를 얻음
    
    # MuJoCo data를 x_k_mujoco 상태로 되돌리고 mj_fwd로 물리량 업데이트
    temp_qpos = x_k_mujoco[:model.nq]
    temp_qvel = x_k_mujoco[model.nq:]
    
    temp_data = mujoco.MjData(model)
    temp_data.qpos[:] = temp_qpos
    temp_data.qvel[:] = temp_qvel
    mujoco.mj_fwd(model, temp_data) # M, CG 계산을 위해 mj_fwd 호출

    M_np = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_np, temp_data.qM)
    CG_np = temp_data.qfrc_bias.copy()

    x_k_plus_1_casadi = dynamics_model_instance.predict(x_k_mujoco, input_torques, model, temp_data)

    # -------------------------------------------------------------
    # 3. 결과 비교
    # 허용 오차 (Euler 적분기와 FreeJoint 처리 단순화로 인해 0이 아님)
    # 이 오차 임계값은 dt, 로봇의 동역학, 통합 방식에 따라 달라집니다.
    # 복잡한 동역학 모델과 Euler 적분기를 사용했으므로, 1e-3 ~ 1e-5 정도도 허용 가능
    tolerance = 1e-4 
    
    diff = np.abs(x_k_plus_1_mujoco - x_k_plus_1_casadi)
    max_diff = np.max(diff)

    print(f"\nMax difference between MuJoCo and CasADi prediction: {max_diff:.2e}")
    # 시각적 확인을 위해 일부 값 출력
    print(f"MuJoCo x_next (first 5): {x_k_plus_1_mujoco[:5]}")
    print(f"CasADi x_next (first 5): {x_k_plus_1_casadi[:5]}")

    assert max_diff < tolerance, f"Prediction mismatch too large: {max_diff:.2e}"

# 짧은 궤적 롤아웃 비교 테스트 (단일 스텝 통과 후 시도)
def test_short_trajectory_rollout(dynamics_model_instance, mujoco_model_data):
    model, data = mujoco_model_data
    
    N_STEPS = 10 # 롤아웃 스텝 수
    
    # 초기 상태 및 입력 시퀀스 설정
    data.qpos = np.random.uniform(model.jnt_range[:,0], model.jnt_range[:,1], model.nq)
    data.qvel = np.random.uniform(-0.1, 0.1, model.nv) # 작은 속도
    
    x_k = np.concatenate([data.qpos, data.qvel])
    
    # MuJoCo 롤아웃 궤적 저장
    mujoco_trajectory = [x_k.copy()]
    
    # CasADi 롤아웃 궤적 저장
    casadi_trajectory = [x_k.copy()]

    # 입력 시퀀스 (N_STEPS 만큼)
    input_sequence = np.random.uniform(-5, 5, (N_STEPS, dynamics_model_instance.input_dim))
    
    # MuJoCo 롤아웃
    temp_data_mujoco = mujoco.MjData(model)
    temp_data_mujoco.qpos[:] = data.qpos
    temp_data_mujoco.qvel[:] = data.qvel

    for i in range(N_STEPS):
        # MuJoCo 컨트롤 설정
        torque_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor_arm_l_joint{j}') for j in range(1,8)] + \
                         [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor_arm_r_joint{j}') for j in range(1,8)]
        for k, act_id in enumerate(torque_act_ids):
            temp_data_mujoco.ctrl[act_id] = input_sequence[i, k]
            
        mujoco.mj_step(model, temp_data_mujoco)
        mujoco_trajectory.append(np.concatenate([temp_data_mujoco.qpos, temp_data_mujoco.qvel]).copy())

    # CasADi 롤아웃
    x_current_casadi = x_k.copy()
    temp_data_casadi_M_CG = mujoco.MjData(model) # M, CG 계산용 데이터

    for i in range(N_STEPS):
        # M, CG는 현재 x_current_casadi에서 계산된 값
        temp_data_casadi_M_CG.qpos[:] = x_current_casadi[:model.nq]
        temp_data_casadi_M_CG.qvel[:] = x_current_casadi[model.nq:]
        mujoco.mj_fwd(model, temp_data_casadi_M_CG)
        
        M_np = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M_np, temp_data_casadi_M_CG.qM)
        CG_np = temp_data_casadi_M_CG.qfrc_bias.copy()

        x_current_casadi = dynamics_model_instance.predict(x_current_casadi, input_sequence[i], model, temp_data_casadi_M_CG)
        casadi_trajectory.append(x_current_casadi.copy())

    mujoco_trajectory = np.array(mujoco_trajectory)
    casadi_trajectory = np.array(casadi_trajectory)

    # 궤적 오차 계산 (RMS 또는 최대 오차)
    rms_diff = np.sqrt(np.mean(np.square(mujoco_trajectory - casadi_trajectory)))
    max_diff_traj = np.max(np.abs(mujoco_trajectory - casadi_trajectory))

    print(f"\nRMS difference over {N_STEPS} steps: {rms_diff:.2e}")
    print(f"Max difference over {N_STEPS} steps: {max_diff_traj:.2e}")
    
    # 롤아웃 오차는 단일 스텝 오차보다 클 수 있으나, 발산하지 않아야 합니다.
    # 허용 오차는 모델의 정확도와 dt에 따라 달라집니다.
    tolerance_rollout = 1e-2 
    assert max_diff_traj < tolerance_rollout, f"Rollout mismatch too large: {max_diff_traj:.2e}"

    # 선택적으로 궤적을 플롯하여 시각적으로 비교
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mujoco_trajectory[:,0], label='MuJoCo qpos[0]')
    plt.plot(casadi_trajectory[:,0], label='CasADi qpos[0]', linestyle='--')
    plt.legend()
    plt.title("Joint 0 Position Trajectory")
    plt.show()
