# Isaac-VR: Virtual Reality Teleoperation for Isaac Sim

VR 환경에서 Isaac Sim의 로봇을 제어하는 텔레오퍼레이션 시스템입니다.

## 🎯 **주요 기능**

- **VR 핸드 트래킹**: Quest 3/SteamVR를 통한 손 움직임 인식
- **실시간 제어**: VR 입력을 Isaac Sim의 Shadow Hand에 실시간 리타겟팅
- **모션 녹화**: 사용자 동작을 5초간 녹화하여 데이터셋 구축
- **외부 데이터 통합**: CMU MoCap 등 외부 모션 데이터셋 활용

## 🚀 **빠른 시작**

```bash
# 기본 텔레오퍼레이션 실행
python shadow_hand/run_teleop.py

# 모션 녹화 실행
python shadow_hand/record_motion.py

# 외부 데이터셋 테스트
python shadow_hand/test_external_data.py
```

## 📁 **프로젝트 구조**

```
isaac-vr/
├── README.md                    # 이 파일
├── shadow_hand/                 # Shadow Hand 텔레오퍼레이션
│   ├── __init__.py
│   ├── run_teleop.py           # 메인 텔레오퍼레이션 실행
│   ├── record_motion.py        # 모션 녹화
│   ├── test_external_data.py   # 외부 데이터셋 테스트
│   ├── core/                   # 핵심 기능
│   │   ├── __init__.py
│   │   ├── vr_tracker.py       # VR 핸드 트래킹
│   │   ├── hand_controller.py  # Shadow Hand 제어
│   │   └── motion_mapper.py    # 모션 리타겟팅
│   ├── configs/                # 설정 파일
│   │   └── teleop_config.yaml  # 텔레오퍼레이션 설정
│   └── utils/                  # 유틸리티
│       ├── __init__.py
│       └── coordinate_utils.py # 좌표계 변환
└── requirements.txt             # 필요한 패키지 목록
```

## 🔧 **설치 및 설정**

### 필수 요구사항
- Isaac Sim 2023.1.1+
- Python 3.8+
- SteamVR 또는 Quest 3
- IsaacLab 환경 설정 완료

### 패키지 설치
```bash
pip install -r requirements.txt
```

## 📖 **사용법**

### 1. 기본 텔레오퍼레이션
```python
from shadow_hand.core.vr_tracker import VRHandTracker
from shadow_hand.core.hand_controller import ShadowHandController

# VR 핸드 트래킹 시작
tracker = VRHandTracker()
controller = ShadowHandController()

# 실시간 제어 루프
while True:
    hand_data = tracker.get_hand_data()
    controller.update_hand(hand_data)
```

### 2. 모션 녹화
```python
from shadow_hand.record_motion import MotionRecorder

recorder = MotionRecorder(duration=5.0)
motion_data = recorder.record()
recorder.save("my_motion.h5")
```

## 🤝 **기여하기**

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 **라이센스**

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

## 🆘 **문제 해결**

### 일반적인 문제들
- **VR 연결 실패**: SteamVR/Quest 3 드라이버 확인
- **Isaac Sim 연결 실패**: 환경 변수 및 경로 설정 확인
- **성능 문제**: GPU 가속 및 설정 최적화

### 지원
- Issues: GitHub Issues 사용
- Discussions: GitHub Discussions 사용
