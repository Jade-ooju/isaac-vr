#!/usr/bin/env python3
"""
Isaac Lab과 Isaac Sim 통합 테스트 스크립트
omni 모듈 import 문제를 해결하기 위한 테스트
"""

import argparse
from isaaclab.app import AppLauncher

# AppLauncher 설정
parser = argparse.ArgumentParser(description="Test Isaac Lab integration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("✅ Isaac Sim이 성공적으로 실행되었습니다!")

# 이제 omni 모듈들을 안전하게 import할 수 있습니다
try:
    import omni.log
    print("✅ omni.log 모듈 import 성공")
except ImportError as e:
    print(f"❌ omni.log import 실패: {e}")

try:
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    print("✅ Isaac Lab datasets 모듈 import 성공")
except ImportError as e:
    print(f"❌ Isaac Lab datasets import 실패: {e}")

try:
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    print("✅ Isaac Lab tasks 모듈 import 성공")
except ImportError as e:
    print(f"❌ Isaac Lab tasks import 실패: {e}")

# HDF5 데이터셋 테스트
try:
    dataset_path = "./datasets/gr00t_teleop_test.hdf5"
    dataset_handler = HDF5DatasetFileHandler()
    dataset_handler.open(dataset_path)
    env_name = dataset_handler.get_env_name()
    episode_count = dataset_handler.get_num_episodes()
    
    print(f"✅ HDF5 데이터셋 로드 성공:")
    print(f"   - 환경 이름: {env_name}")
    print(f"   - 에피소드 수: {episode_count}")
    
except Exception as e:
    print(f"❌ HDF5 데이터셋 로드 실패: {e}")

# Isaac Sim 종료
simulation_app.close()
print("✅ Isaac Sim이 정상적으로 종료되었습니다!")
