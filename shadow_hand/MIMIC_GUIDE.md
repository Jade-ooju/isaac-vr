# Isaac Lab Mimic 통합 가이드 (Windows 환경)

이 가이드는 Windows 환경에서 Isaac Sim 5.0과 VR 텔레오퍼레이션을 사용하여 Isaac Lab Mimic 워크플로우를 실행하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [시스템 요구사항](#시스템-요구사항)
3. [설치 및 설정](#설치-및-설정)
4. [사용법](#사용법)
5. [Windows 환경 특화 설정](#windows-환경-특화-설정)
6. [문제 해결](#문제-해결)

## 🎯 개요

Isaac Lab Mimic은 로봇 조작 태스크를 위한 모방 학습 프레임워크입니다. 이 통합 시스템은 다음 기능을 제공합니다:

- **VR 기반 데모 수집**: Quest 3/SteamVR를 사용한 자연스러운 데모 녹화
- **Isaac Lab Mimic 호환**: 표준 Isaac Lab Mimic 형식으로 데이터 저장
- **Robomimic 통합**: 수집된 데이터를 Robomimic으로 변환하여 정책 훈련
- **Windows 지원**: Linux 제약사항을 우회한 Windows 환경 지원

## 💻 시스템 요구사항

### 필수 요구사항
- **운영체제**: Windows 10/11
- **Isaac Sim**: 5.0 버전
- **Python**: 3.8+
- **VR 시스템**: Quest 3 또는 SteamVR 호환 헤드셋
- **GPU**: NVIDIA RTX 시리즈 (권장)

### 소프트웨어 요구사항
- Isaac Lab 환경 설정 완료
- VR 런타임 (SteamVR 또는 Quest Link)
- Python 패키지: `h5py`, `numpy`, `pyyaml`

## 🔧 설치 및 설정

### 1. Isaac Lab 환경 설정

```bash
# Isaac Lab 환경 활성화
conda activate isaaclab

# Isaac Lab 설치 확인
python -c "import isaaclab; print('Isaac Lab 설치 완료')"
```

### 2. VR 시스템 설정

#### Quest 3 설정
```bash
# Quest Link 또는 Air Link 활성화
# Meta Quest 앱에서 개발자 모드 활성화
# USB 디버깅 허용
```

#### SteamVR 설정
```bash
# SteamVR 설치 및 실행
# 헤드셋 및 컨트롤러 페어링 완료
# Room Setup 완료
```

### 3. Robomimic 설치 (선택사항)

```bash
# Robomimic 클론
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic

# 의존성 설치
pip install -e .

# Windows에서 추가 설정
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 사용법

### 1. 기본 텔레오퍼레이션

```bash
# VR 텔레오퍼레이션 실행
python run_teleop.py

# 특정 VR 시스템 지정
python run_teleop.py --vr-system quest3
```

### 2. Isaac Lab Mimic 데모 수집

```bash
# 전체 파이프라인 실행 (데모 수집 → 변환 → 훈련 → 평가)
python run_mimic_pipeline.py --action full --task "Isaac-Repose-Cube-Shadow-Direct-v0"

# 데모 수집만 실행
python run_mimic_pipeline.py --action collect --task "Isaac-Repose-Cube-Shadow-Direct-v0" --num-demos 10

# 특정 길이의 데모 수집
python run_mimic_pipeline.py --action collect --task "Isaac-Repose-Cube-Shadow-Direct-v0" --duration 15.0
```

### 3. 데이터 변환 및 훈련

```bash
# Isaac Lab Mimic 데이터를 Robomimic 형식으로 변환
python run_mimic_pipeline.py --action convert --dataset "./mimic_demos/task_dataset.h5"

# 정책 훈련
python run_mimic_pipeline.py --action train --dataset "./robomimic_format"

# 정책 평가
python run_mimic_pipeline.py --action evaluate --model "./trained_models" --dataset "./robomimic_format"
```

### 4. 개별 모듈 사용

```bash
# Isaac Lab Mimic 데모 녹화기 직접 사용
python mimic_demo_recorder.py --task "Isaac-Repose-Cube-Shadow-Direct-v0" --num-demos 5

# Robomimic 통합 직접 사용
python robomimic_integration.py --action convert --dataset "./mimic_demos/dataset.h5"
```

## 🪟 Windows 환경 특화 설정

### 1. VR 드라이버 설정

#### Quest 3
```yaml
# configs/teleop_config.yaml
vr_tracker:
  system_type: "quest3"
  hand_tracking:
    enable: true
    confidence_threshold: 0.7
    smoothing_factor: 0.3
```

#### SteamVR
```yaml
# configs/teleop_config.yaml
vr_tracker:
  system_type: "steamvr"
  hand_tracking:
    enable: true
    confidence_threshold: 0.8
    smoothing_factor: 0.2
```

### 2. Isaac Sim 5.0 통합

```python
# Isaac Sim 5.0 환경 설정
import isaaclab
from isaaclab.envs import ManagerBasedRLEnv

# 환경 생성
env = ManagerBasedRLEnv(cfg=YourEnvCfg())
```

### 3. Windows 경로 설정

```python
# Windows 경로 처리
import os
from pathlib import Path

# 절대 경로 사용 권장
save_path = Path("C:/IsaacLab/IsaacLab/custom_scripts/isaac-vr/shadow_hand/mimic_demos/")
save_path.mkdir(parents=True, exist_ok=True)
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. VR 연결 실패
```bash
# Quest 3 연결 확인
# - USB 케이블 연결 상태 확인
# - Quest Link 활성화 확인
# - 개발자 모드 활성화 확인

# SteamVR 연결 확인
# - SteamVR 실행 상태 확인
# - 헤드셋 및 컨트롤러 페어링 확인
```

#### 2. Isaac Sim 연결 실패
```bash
# Isaac Sim 실행 확인
# - Isaac Sim 5.0이 실행 중인지 확인
# - 환경 변수 설정 확인
# - Python 경로 설정 확인

# 환경 변수 설정
set ISAAC_SIM_PATH=C:\Program Files\NVIDIA\Isaac\isaac-sim-5.0.0
set PYTHONPATH=%ISAAC_SIM_PATH%\python;%PYTHONPATH%
```

#### 3. Robomimic 설치 문제
```bash
# Windows에서 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 버전 확인
nvidia-smi

# 의존성 충돌 해결
pip install --upgrade pip
pip install --no-deps robomimic
```

#### 4. 메모리 부족 문제
```yaml
# configs/teleop_config.yaml
mimic:
  default_duration: 5.0  # 데모 길이 단축
  success_criteria:
    min_duration: 1.0    # 최소 길이 단축
```

### 로그 확인

```bash
# 로그 파일 확인
type mimic_pipeline.log

# 상세 로그 활성화
python run_mimic_pipeline.py --action full --task "your_task" --config configs/teleop_config.yaml
```

## 📊 성능 최적화

### 1. VR 성능 최적화
- VR 헤드셋 해상도 조정
- 프레임레이트 최적화 (60fps → 72fps)
- 핸드 트래킹 신뢰도 임계값 조정

### 2. Isaac Sim 성능 최적화
- 렌더링 설정 최적화
- 물리 시뮬레이션 스텝 크기 조정
- GPU 메모리 사용량 모니터링

### 3. 데이터 수집 최적화
- 데모 길이 최적화 (5-15초)
- 서브태스크 주석 전략 개선
- 데이터 품질 필터링

## 📚 추가 자료

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab Mimic 문서](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html)
- [Robomimic GitHub](https://github.com/ARISE-Initiative/robomimic)
- [VR 개발 가이드](https://developer.oculus.com/)

## 🤝 지원

문제가 발생하면 다음을 확인하세요:

1. **로그 파일**: `mimic_pipeline.log` 확인
2. **VR 시스템**: 헤드셋 및 컨트롤러 상태 확인
3. **Isaac Sim**: 실행 상태 및 환경 설정 확인
4. **Python 환경**: 패키지 설치 및 버전 확인

추가 지원이 필요한 경우 GitHub Issues를 통해 문의하세요.

