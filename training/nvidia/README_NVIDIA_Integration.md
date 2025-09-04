# NVIDIA 데이터셋 IsaacLab 통합 가이드

이 디렉토리는 NVIDIA의 GR00T 관련 데이터셋을 IsaacLab에서 활용할 수 있도록 하는 통합 솔루션을 제공합니다.

## 📁 포함된 데이터셋

### 1. PhysicalAI-Robotics-GR00T-Teleop-G1
- **용도**: 로봇 원격조작(Teleoperation) 데이터
- **크기**: 123,713개 프레임, 187개 에피소드
- **내용**: 43차원 상태/액션 벡터, 시계열 로봇 조작 데이터
- **활용**: 모방학습(Imitation Learning) 훈련 데이터

### 2. PhysicalAI-Robotics-GR00T-Eval
- **용도**: 로봇 태스크 평가를 위한 텍스트 명령어
- **크기**: 126개 태스크
- **내용**: 다양한 일상생활 태스크 설명
- **활용**: 태스크 평가, 벤치마킹, 환경 설정

## 🛠️ 제공되는 도구

### 1. `nvidia_dataset_integration.py`
NVIDIA GR00T-Teleop 데이터셋을 IsaacLab HDF5 형식으로 변환합니다.

```bash
# 기본 변환
python nvidia_dataset_integration.py \
    --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --output_file ./datasets/gr00t_teleop.hdf5

# 테스트용 (5개 에피소드만)
python nvidia_dataset_integration.py \
    --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --output_file ./datasets/gr00t_teleop_demo.hdf5 \
    --max_episodes 5
```

### 2. `gr00t_eval_integration.py`
GR00T-Eval 데이터셋을 분석하고 태스크 리스트를 내보냅니다.

```bash
# 태스크 분석
python gr00t_eval_integration.py --analyze_tasks

# 태스크 리스트 내보내기
python gr00t_eval_integration.py \
    --export_task_list \
    --output_file ./datasets/gr00t_tasks.json

# 키워드로 필터링
python gr00t_eval_integration.py \
    --export_task_list \
    --filter_by_keywords pick place open close \
    --output_file ./datasets/gr00t_filtered_tasks.json
```

### 3. `example_usage.py`
통합 사용 예제를 제공합니다.

```bash
# 모든 데모 실행
python example_usage.py --demo all

# 개별 데모 실행
python example_usage.py --demo convert_teleop
python example_usage.py --demo analyze_eval
```

## 🚀 IsaacLab에서의 활용 방법

### 1. 변환된 데이터셋 재생
```bash
python scripts/tools/replay_demos.py \
    --dataset_file ./datasets/gr00t_teleop.hdf5 \
    --num_envs 1
```

### 2. 모방학습 훈련
```bash
# RoboMimic 사용
python scripts/imitation_learning/robomimic/train.py \
    --config ./configs/gr00t_config.yaml \
    --dataset ./datasets/gr00t_teleop.hdf5

# IsaacLab Mimic 사용
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --input_file ./datasets/gr00t_teleop.hdf5 \
    --output_file ./datasets/gr00t_generated.hdf5
```

### 3. 커스텀 환경에서 활용
```python
from isaaclab.utils.datasets import HDF5DatasetFileHandler

# 데이터셋 로드
dataset_handler = HDF5DatasetFileHandler()
dataset_handler.open("./datasets/gr00t_teleop.hdf5")

# 에피소드 재생
episode_names = dataset_handler.get_episode_names()
for episode_name in episode_names:
    episode_data = dataset_handler.load_episode(episode_name, device="cuda")
    # 에피소드 데이터 활용
```

## 📊 데이터 구조

### GR00T-Teleop 변환 후 구조
```
HDF5 Dataset
├── data/
│   ├── demo_0/
│   │   ├── initial_state
│   │   ├── actions
│   │   ├── obs/policy/state
│   │   └── obs/policy/img_state_delta
│   ├── demo_1/
│   └── ...
└── metadata
    ├── env_name
    └── conversion_info
```

### GR00T-Eval 태스크 카테고리
- **Pick and Place**: 물체 집기/놓기
- **Manipulation**: 조작 (잡기, 돌리기, 누르기)
- **Opening/Closing**: 열기/닫기
- **Cleaning**: 청소 (닦기, 지우기, 쓸기)
- **Cooking**: 요리 (섞기, 붓기, 자르기)
- **Music**: 음악 (연주, 흔들기, 치기)
- **Writing/Drawing**: 쓰기/그리기
- **Other**: 기타

## 🔧 요구사항

### 필수 패키지
```bash
pip install datasets
pip install torch
pip install h5py
pip install numpy
```

### IsaacLab 환경
- IsaacLab이 설치되어 있어야 함
- CUDA 지원 (권장)
- 충분한 디스크 공간 (데이터셋 크기에 따라)

## 📝 사용 예제

### 1. 전체 파이프라인 실행
```bash
# 1. 데이터셋 변환
python nvidia_dataset_integration.py \
    --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --output_file ./datasets/gr00t_teleop.hdf5

# 2. 태스크 분석
python gr00t_eval_integration.py \
    --analyze_tasks \
    --export_task_list \
    --output_file ./datasets/gr00t_tasks.json

# 3. 변환된 데이터 재생
python scripts/tools/replay_demos.py \
    --dataset_file ./datasets/gr00t_teleop.hdf5
```

### 2. 특정 태스크만 필터링
```bash
# 요리 관련 태스크만 추출
python gr00t_eval_integration.py \
    --export_task_list \
    --filter_by_keywords cook stir pour mix cut \
    --output_file ./datasets/cooking_tasks.json
```

### 3. 테스트용 소규모 변환
```bash
# 10개 에피소드만 변환하여 테스트
python nvidia_dataset_integration.py \
    --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    --output_file ./datasets/gr00t_teleop_test.hdf5 \
    --max_episodes 10
```

## 🎯 활용 시나리오

### 1. 모방학습 연구
- GR00T-Teleop 데이터를 사용하여 로봇 조작 정책 훈련
- IsaacLab의 다양한 모방학습 알고리즘과 비교 실험

### 2. 태스크 평가
- GR00T-Eval 태스크를 IsaacLab 환경에 구현
- 로봇 성능 벤치마킹 및 평가

### 3. 커스텀 환경 개발
- GR00T 태스크를 참고하여 새로운 IsaacLab 환경 개발
- 기존 태스크의 변형 및 확장

### 4. 데이터 증강
- 변환된 데이터셋을 기반으로 추가 데이터 생성
- 시뮬레이션 환경에서의 데이터 증강

## 🐛 문제 해결

### 일반적인 문제들

1. **메모리 부족**
   - `--max_episodes` 옵션으로 에피소드 수 제한
   - 배치 크기 조정

2. **CUDA 메모리 부족**
   - CPU 모드 사용: `device="cpu"`
   - 에피소드 수 제한

3. **데이터셋 다운로드 실패**
   - 인터넷 연결 확인
   - Hugging Face 토큰 설정 (필요시)

4. **IsaacLab 환경 오류**
   - IsaacLab 설치 확인
   - 환경 변수 설정 확인

## 📞 지원

문제가 발생하거나 추가 기능이 필요한 경우:
1. IsaacLab 공식 문서 참조
2. GitHub Issues에 문제 보고
3. 커뮤니티 포럼 활용

## 📄 라이선스

이 통합 솔루션은 IsaacLab과 동일한 라이선스를 따릅니다.
NVIDIA 데이터셋의 사용은 해당 데이터셋의 라이선스 조건을 준수해야 합니다.
