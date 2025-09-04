# Isaac Sim 완전 재설치 가이드

## 현재 문제
- `RuntimeError: Failed to acquire interface: omni::kit::IApp (pluginName: nullptr)`
- Isaac Sim이 실행되지 않음

## 완전 재설치 방법

### 1. 현재 환경 정리
```bash
# 현재 환경 비활성화
conda deactivate

# Isaac Sim 관련 패키지 완전 제거
conda remove isaacsim isaacsim-app isaacsim-asset isaacsim-benchmark isaacsim-code-editor isaacsim-core isaacsim-cortex isaacsim-example isaacsim-extscache-kit isaacsim-extscache-kit-sdk isaacsim-extscache-physics isaacsim-gui isaacsim-kernel isaacsim-replicator isaacsim-rl isaacsim-robot isaacsim-robot-motion isaacsim-robot-setup isaacsim-ros1 isaacsim-ros2 isaacsim-sensor isaacsim-storage isaacsim-template isaacsim-test isaacsim-utils

# Isaac Lab 관련 패키지 제거
conda remove isaaclab isaaclab-assets isaaclab-mimic isaaclab-rl isaaclab-tasks
```

### 2. Isaac Sim 재설치
```bash
# Isaac Sim 재설치
conda install -c conda-forge isaac-sim

# 또는 Isaac Lab 재설치
isaaclab -i
```

### 3. 환경 변수 설정
```bash
# Windows에서 환경 변수 설정
set ISAAC_SIM_PATH=C:\Users\%USERNAME%\AppData\Local\ov\pkg\isaac_sim-5.0.0
set OMNI_USER_DIR=C:\Users\%USERNAME%\AppData\Local\ov\pkg\isaac_sim-5.0.0\kit\kit\user
```

### 4. GPU 드라이버 확인
- NVIDIA 드라이버를 최신 버전으로 업데이트
- CUDA 버전 호환성 확인

## 대안: Isaac Sim 없이 작업 계속

현재 상태에서도 많은 작업을 계속할 수 있습니다:

### ✅ 완성된 작업들
1. **NVIDIA 데이터셋 변환**: GR00T-Teleop-G1 → IsaacLab HDF5 형식
2. **모방학습 모델 훈련**: 1,905개 샘플로 성공적으로 훈련
3. **데이터 분석**: 완전한 데이터셋 구조 및 통계 분석
4. **모델 추론**: 훈련된 모델로 액션 시퀀스 생성
5. **시각화**: 생성된 액션 시퀀스 시각화

### 🚀 다음 단계
- 실제 로봇에 모델 적용
- 모방학습 연구 계속 진행
- 데이터 분석 및 시각화 확장
