# 🚀 Isaac-VR 빠른 시작 가이드

## 📋 **사전 요구사항**

- ✅ Isaac Sim 2023.1.1+ 설치 및 설정
- ✅ IsaacLab 환경 설정 완료
- ✅ Python 3.8+
- ✅ SteamVR 또는 Quest 3 연결

## ⚡ **5분 내 실행하기**

### **1. 패키지 설치**
```bash
cd custom_scripts/isaac-vr
pip install -r requirements.txt
```

### **2. 기본 텔레오퍼레이션 실행**
```bash
# VR 없이 더미 데이터로 테스트
python shadow_hand/run_teleop.py

# SteamVR과 함께 실행
python shadow_hand/run_teleop.py --vr-system steamvr

# Quest 3와 함께 실행
python shadow_hand/run_teleop.py --vr-system quest3
```

### **3. 모션 녹화 실행**
```bash
# 5초간 기본 녹화
python shadow_hand/record_motion.py

# 10초간 제스처별 녹화
python shadow_hand/record_motion.py --duration 10 --gesture "pinch_grasp"
```

### **4. 외부 데이터셋 테스트**
```bash
python shadow_hand/test_external_data.py
```

## 🎯 **주요 기능 테스트**

### **A. VR 핸드 트래킹 테스트**
```python
from shadow_hand.core.vr_tracker import VRHandTracker

# VR 트래커 생성
tracker = VRHandTracker(vr_system="steamvr")

# 핸드 데이터 가져오기
hand_data = tracker.get_hand_data()
print(f"핸드 데이터: {hand_data.shape}")

# 트래킹 상태 확인
status = tracker.get_tracking_status()
print(f"연결 상태: {status['is_connected']}")
```

### **B. Shadow Hand 제어 테스트**
```python
from shadow_hand.core.hand_controller import ShadowHandController

# 컨트롤러 생성
controller = ShadowHandController()

# 핸드 상태 확인
joints = controller.get_joint_positions()
print(f"현재 관절 위치: {joints.shape}")

# 핸드 리셋
controller.reset_hand()
```

### **C. 모션 매핑 테스트**
```python
from shadow_hand.core.motion_mapper import MotionMapper
import numpy as np

# 모션 매퍼 생성
mapper = MotionMapper()

# 테스트 VR 데이터
vr_data = np.random.randn(21, 3) * 0.1

# Shadow Hand 관절 위치로 변환
joint_positions = mapper.map_vr_to_shadow_hand(vr_data, "steamvr")
print(f"변환된 관절 위치: {joint_positions.shape}")
```

## 🔧 **문제 해결**

### **VR 연결 실패**
```bash
# SteamVR 상태 확인
python shadow_hand/run_teleop.py --vr-system steamvr

# Quest 3 상태 확인
python shadow_hand/run_teleop.py --vr-system quest3

# 자동 감지
python shadow_hand/run_teleop.py --vr-system auto
```

### **Isaac Sim 연결 실패**
```bash
# 환경 변수 확인
echo $ISAAC_PATH
echo $PYTHONPATH

# IsaacLab 설치 확인
python -c "import isaaclab; print('✅ IsaacLab 설치됨')"
```

### **성능 문제**
```bash
# 로그 레벨 조정
python shadow_hand/run_teleop.py --config custom_config.yaml

# 설정 파일에서 업데이트 주파수 조정
# configs/teleop_config.yaml
vr:
  update_rate: 30  # 60 -> 30으로 낮춤
```

## 📁 **파일 구조 이해**

```
isaac-vr/
├── shadow_hand/
│   ├── run_teleop.py          # 🎮 메인 텔레오퍼레이션
│   ├── record_motion.py       # 🎬 모션 녹화
│   ├── test_external_data.py  # 🧪 외부 데이터 테스트
│   ├── core/                  # 🔧 핵심 기능
│   │   ├── vr_tracker.py      # VR 핸드 트래킹
│   │   ├── hand_controller.py # Shadow Hand 제어
│   │   └── motion_mapper.py   # 모션 리타겟팅
│   ├── configs/               # ⚙️ 설정 파일
│   └── utils/                 # 🛠️ 유틸리티
└── requirements.txt            # 📦 필요한 패키지
```

## 🎮 **실행 시나리오**

### **시나리오 1: VR 없이 테스트**
```bash
# 더미 데이터로 Shadow Hand 움직임 확인
python shadow_hand/run_teleop.py
# → 시간에 따라 움직이는 더미 핸드 데이터로 Shadow Hand 제어
```

### **시나리오 2: SteamVR 텔레오퍼레이션**
```bash
# SteamVR 실행 후
python shadow_hand/run_teleop.py --vr-system steamvr
# → 실제 VR 핸드 움직임으로 Shadow Hand 제어
```

### **시나리오 3: 모션 녹화**
```bash
# 5초간 핸드 움직임 녹화
python shadow_hand/record_motion.py --gesture "pinch"
# → recorded_motions/ 폴더에 H5 파일로 저장
```

### **시나리오 4: 외부 데이터 통합**
```bash
# 외부 모션 데이터 테스트
python shadow_hand/test_external_data.py
# → CMU MoCap 등 외부 데이터셋 로딩 방법 확인
```

## 📊 **성능 모니터링**

실행 중 다음 정보가 실시간으로 출력됩니다:
- **FPS**: 현재 프레임 속도
- **VR 상태**: 연결 상태 및 신뢰도
- **Shadow Hand**: 관절 위치 범위
- **진행률**: 모션 녹화 시 진행 상황

## 🚨 **주의사항**

1. **VR 시스템**: SteamVR 또는 Quest 3 중 하나만 연결
2. **Isaac Sim**: 반드시 IsaacLab 환경에서 실행
3. **성능**: GPU 가속 권장, 60fps 이상 유지
4. **저장**: 녹화된 데이터는 자동으로 `recorded_motions/` 폴더에 저장

## 🔗 **다음 단계**

- [ ] 기본 텔레오퍼레이션 동작 확인
- [ ] VR 시스템 연결 및 테스트
- [ ] 모션 녹화 및 저장 테스트
- [ ] 외부 데이터셋 통합
- [ ] 고급 기능 개발 (제스처 인식, AI 모델 등)

## 💡 **팁**

- **처음 실행**: VR 없이 더미 데이터로 테스트
- **디버깅**: 로그 레벨을 DEBUG로 설정하여 상세 정보 확인
- **성능**: 업데이트 주파수를 낮춰서 안정성 확보
- **저장**: H5 형식으로 저장하면 메타데이터와 함께 효율적으로 관리

---

**🎯 목표: 1주 내에 VR 핸드 트래킹으로 Shadow Hand를 실시간 제어하고, 모션을 녹화하여 데이터셋을 구축하는 것!**
