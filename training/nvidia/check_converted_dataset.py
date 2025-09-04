#!/usr/bin/env python3
"""
변환된 HDF5 데이터셋의 구조를 확인하는 스크립트
"""

import h5py
import numpy as np
import json
import os


def check_hdf5_dataset(file_path):
    """HDF5 데이터셋의 구조를 확인합니다."""
    print(f"Checking HDF5 dataset: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    with h5py.File(file_path, 'r') as f:
        print(f"📁 File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"📊 HDF5 structure:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  📄 {name}: {obj.shape} ({obj.dtype})")
            elif isinstance(obj, h5py.Group):
                print(f"  📁 {name}/")
        
        f.visititems(print_structure)
        
        # 메타데이터 확인
        if 'data' in f:
            print(f"\n📈 Dataset info:")
            print(f"  - Number of episodes: {len(f['data'])}")
            
            # 첫 번째 에피소드 상세 정보
            if len(f['data']) > 0:
                first_episode = list(f['data'].keys())[0]
                print(f"  - First episode: {first_episode}")
                
                episode_data = f['data'][first_episode]
                print(f"  - Episode structure:")
                for key in episode_data.keys():
                    if isinstance(episode_data[key], h5py.Dataset):
                        print(f"    - {key}: {episode_data[key].shape} ({episode_data[key].dtype})")
                    else:
                        print(f"    - {key}/: {len(episode_data[key])} items")
                
                # 액션과 관찰 데이터 샘플
                if 'actions' in episode_data:
                    actions = episode_data['actions']
                    print(f"\n🎮 Actions sample:")
                    print(f"  - Shape: {actions.shape}")
                    print(f"  - First action: {actions[0]}")
                    print(f"  - Action range: [{actions[:].min():.3f}, {actions[:].max():.3f}]")
                
                if 'obs/policy/state' in episode_data:
                    obs_state = episode_data['obs/policy/state']
                    print(f"\n👁️ Observation state sample:")
                    print(f"  - Shape: {obs_state.shape}")
                    print(f"  - First observation: {obs_state[0]}")
                    print(f"  - State range: [{obs_state[:].min():.3f}, {obs_state[:].max():.3f}]")


def check_conversion_info(file_path):
    """변환 정보 파일을 확인합니다."""
    info_file = os.path.splitext(file_path)[0] + "_conversion_info.json"
    
    if os.path.exists(info_file):
        print(f"\n📋 Conversion info:")
        with open(info_file, 'r') as f:
            info = json.load(f)
            for key, value in info.items():
                print(f"  - {key}: {value}")
    else:
        print(f"\n❌ Conversion info file not found: {info_file}")


def main():
    """메인 함수"""
    dataset_file = "./datasets/gr00t_teleop_test.hdf5"
    
    print("🔍 Checking converted NVIDIA dataset")
    print("=" * 60)
    
    check_hdf5_dataset(dataset_file)
    check_conversion_info(dataset_file)
    
    print(f"\n✅ Dataset check completed!")
    print(f"\nNext steps:")
    print(f"  1. Use the dataset with IsaacLab's imitation learning pipeline")
    print(f"  2. Convert more episodes if needed")
    print(f"  3. Integrate with your custom IsaacLab environments")


if __name__ == "__main__":
    main()
