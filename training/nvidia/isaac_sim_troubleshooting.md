# Isaac Sim 문제 해결 가이드

## 현재 문제
- `RuntimeError: Failed to acquire interface: omni::kit::IApp (pluginName: nullptr)`
- Isaac Sim이 실행되지 않음

## 해결 방법들

### 1. Isaac Sim 재설치
```bash
# 현재 환경 비활성화
conda deactivate

# Isaac Sim 재설치
conda install -c conda-forge isaac-sim

# 또는 Isaac Lab 재설치
isaaclab -i
```

### 2. 환경 변수 확인
```bash
# Isaac Sim 경로 확인
echo $ISAAC_SIM_PATH
echo $OMNI_USER_DIR

# Windows에서
echo %ISAAC_SIM_PATH%
echo %OMNI_USER_DIR%
```

### 3. GPU 드라이버 업데이트
- NVIDIA 드라이버를 최신 버전으로 업데이트
- CUDA 버전 호환성 확인

### 4. 권한 문제 해결
```bash
# 관리자 권한으로 실행
# 또는 Isaac Sim 설치 폴더 권한 확인
```

### 5. 대안: Isaac Sim 없이 작업
- 현재 NVIDIA 데이터셋 변환과 모방학습은 완료됨
- Isaac Sim 없이도 데이터 분석과 모델 훈련 가능
- 실제 로봇에 모델 적용 가능

## 현재 상태
✅ NVIDIA 데이터셋 변환 완료
✅ 모방학습 모델 훈련 완료
✅ 데이터 분석 도구 완성
❌ Isaac Sim 실행 문제 (해결 필요)
