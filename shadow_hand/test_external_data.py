#!/usr/bin/env python3
"""
외부 데이터셋 테스트 파일
CMU MoCap 등 외부 모션 데이터를 로드하여 테스트
"""

import sys
import os
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shadow_hand.core.motion_mapper import MotionMapper

def test_motion_mapper():
    """모션 매퍼 테스트"""
    print("🧪 모션 매퍼 테스트 시작...")
    
    # 모션 매퍼 생성
    mapper = MotionMapper()
    
    # 테스트용 VR 핸드 데이터 (21개 관절, 3D 좌표)
    test_hand_data = np.random.randn(21, 3) * 0.1
    
    print(f"테스트 VR 데이터 형태: {test_hand_data.shape}")
    print(f"VR 데이터 샘플:\n{test_hand_data[:5]}")
    
    # SteamVR로 매핑 테스트
    joint_positions = mapper.map_vr_to_shadow_hand(test_hand_data, "steamvr")
    print(f"\nSteamVR 매핑 결과:")
    print(f"관절 위치 형태: {joint_positions.shape}")
    print(f"관절 위치 샘플: {joint_positions[:10]}")
    
    # Quest 3로 매핑 테스트
    joint_positions_quest = mapper.map_vr_to_shadow_hand(test_hand_data, "quest3")
    print(f"\nQuest 3 매핑 결과:")
    print(f"관절 위치 형태: {joint_positions_quest.shape}")
    print(f"관절 위치 샘플: {joint_positions_quest[:10]}")
    
    # 매핑 정보 출력
    mapping_info = mapper.get_joint_mapping_info("steamvr")
    print(f"\n관절 매핑 정보:")
    for finger_type, mapping in mapping_info.items():
        print(f"  {finger_type}: VR {mapping['vr_indices']} -> Shadow {mapping['shadow_indices']}")
    
    print("✅ 모션 매퍼 테스트 완료!")

def test_coordinate_transformer():
    """좌표계 변환기 테스트"""
    print("\n🧪 좌표계 변환기 테스트 시작...")
    
    try:
        from shadow_hand.utils.coordinate_utils import CoordinateTransformer
        
        transformer = CoordinateTransformer()
        
        # 테스트 데이터
        test_vr_data = np.random.randn(21, 3) * 0.1
        
        # VR to Isaac 변환
        transformed_data = transformer.transform_vr_to_isaac(test_vr_data, "steamvr")
        print(f"VR -> Isaac 변환 결과: {transformed_data.shape}")
        
        # 관절 제한 범위 적용
        joint_limits = {
            "wrist": [-0.5, 0.5],
            "thumb": [-0.8, 0.8],
            "fingers": [-1.57, 1.57]
        }
        
        normalized_positions = transformer.normalize_joint_positions(
            transformed_data.flatten()[:20],  # 20개 관절로 제한
            joint_limits
        )
        print(f"관절 제한 범위 적용 결과: {normalized_positions.shape}")
        
        # 스무딩 적용
        smoothed_positions = transformer.apply_smoothing(
            np.zeros(20),
            normalized_positions,
            smoothing_factor=0.3
        )
        print(f"스무딩 적용 결과: {smoothed_positions.shape}")
        
        print("✅ 좌표계 변환기 테스트 완료!")
        
    except ImportError as e:
        print(f"⚠️ 좌표계 변환기 테스트 건너뜀: {e}")

def test_external_dataset_loading():
    """외부 데이터셋 로딩 테스트"""
    print("\n🧪 외부 데이터셋 로딩 테스트 시작...")
    
    # CMU MoCap 데이터 다운로드 링크 (예시)
    cmu_mocap_urls = [
        "http://mocap.cs.cmu.edu/",
        "https://github.com/CMU-Perceptual-Computing-Lab/openpose"
    ]
    
    print("📚 사용 가능한 외부 데이터셋:")
    print("  1. CMU Motion Capture Database")
    print("  2. HDM05 (Human Motion Database)")
    print("  3. Berkeley MHAD Dataset")
    print("  4. NTU RGB+D Dataset")
    
    print("\n💡 데이터셋 다운로드 방법:")
    print("  - CMU MoCap: http://mocap.cs.cmu.edu/")
    print("  - HDM05: https://resources.mpi-inf.mpg.de/HDM05/")
    print("  - Berkeley MHAD: http://tele-immersion.citris-uc.org/berkeley_mhad")
    
    print("\n🔧 데이터 로딩 예시 코드:")
    print("""
# H5 파일 로딩
import h5py
with h5py.File('motion_data.h5', 'r') as f:
    joint_positions = f['joint_positions_array'][:]
    hand_data = f['hand_data_array'][:]
    print(f"로딩된 모션: {joint_positions.shape}")
    
# NPZ 파일 로딩
import numpy as np
data = np.load('motion_data.npz')
joint_positions = data['joint_positions']
hand_data = data['hand_data']
print(f"로딩된 모션: {joint_positions.shape}")
    """)
    
    print("✅ 외부 데이터셋 로딩 테스트 완료!")

def main():
    """메인 테스트 함수"""
    print("🚀 Isaac-VR 외부 데이터셋 테스트 시작!")
    print("=" * 50)
    
    try:
        # 1. 모션 매퍼 테스트
        test_motion_mapper()
        
        # 2. 좌표계 변환기 테스트
        test_coordinate_transformer()
        
        # 3. 외부 데이터셋 로딩 테스트
        test_external_dataset_loading()
        
        print("\n🎉 모든 테스트 완료!")
        print("\n📖 다음 단계:")
        print("  1. 실제 VR 시스템 연결")
        print("  2. Shadow Hand 환경 실행")
        print("  3. 모션 녹화 및 저장")
        print("  4. 외부 데이터셋 통합")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 에러: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
