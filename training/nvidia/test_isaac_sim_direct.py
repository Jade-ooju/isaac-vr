#!/usr/bin/env python3
"""
Isaac Sim 직접 실행 테스트
SimulationApp을 직접 사용하여 문제를 진단합니다.
"""

import sys
import os

print("🔍 Isaac Sim 직접 실행 테스트")
print("=" * 40)

# 1. Isaac Sim 모듈 import 테스트
try:
    from isaacsim import SimulationApp
    print("✅ Isaac Sim SimulationApp import 성공")
except ImportError as e:
    print(f"❌ Isaac Sim SimulationApp import 실패: {e}")
    sys.exit(1)

# 2. SimulationApp 생성 테스트
try:
    print("🚀 SimulationApp 생성 시도...")
    sim_app = SimulationApp({"headless": True})
    print("✅ SimulationApp 생성 성공!")
    
    # 3. omni 모듈 테스트
    try:
        import omni.log
        print("✅ omni.log import 성공")
    except ImportError as e:
        print(f"❌ omni.log import 실패: {e}")
    
    try:
        import omni.kit.app
        print("✅ omni.kit.app import 성공")
    except ImportError as e:
        print(f"❌ omni.kit.app import 실패: {e}")
    
    # 4. Isaac Lab 모듈 테스트
    try:
        from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
        print("✅ Isaac Lab datasets import 성공")
    except ImportError as e:
        print(f"❌ Isaac Lab datasets import 실패: {e}")
    
    # 5. SimulationApp 종료
    sim_app.close()
    print("✅ SimulationApp 정상 종료")
    
except Exception as e:
    print(f"❌ SimulationApp 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
print("Isaac Sim이 정상적으로 작동합니다.")
