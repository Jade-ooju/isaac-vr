#!/usr/bin/env python3
"""
변환된 HDF5 데이터셋의 환경 이름을 수정하는 스크립트
"""

import h5py
import shutil
import os


def fix_dataset_env_name(input_file, output_file, new_env_name):
    """데이터셋의 환경 이름을 수정합니다."""
    print(f"Fixing environment name in dataset...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  New env name: {new_env_name}")
    
    # 원본 파일을 복사
    shutil.copy2(input_file, output_file)
    
    # HDF5 파일 열기
    with h5py.File(output_file, 'r+') as f:
        # 환경 이름 변경
        if 'data' in f.attrs:
            f.attrs['env_name'] = new_env_name
            print(f"  ✅ Environment name updated to: {new_env_name}")
        else:
            print(f"  ⚠️  No env_name attribute found, adding new one")
            f.attrs['env_name'] = new_env_name
    
    print(f"  ✅ Dataset fixed successfully!")


def main():
    """메인 함수"""
    input_file = "./datasets/gr00t_teleop_test.hdf5"
    output_file = "./datasets/gr00t_teleop_test_fixed.hdf5"
    new_env_name = "Isaac-Cube-v0"  # 기존 IsaacLab 환경 이름 사용
    
    print("🔧 Fixing dataset environment name")
    print("=" * 50)
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    fix_dataset_env_name(input_file, output_file, new_env_name)
    
    print(f"\n✅ Fixed dataset saved to: {output_file}")
    print(f"\nNow you can try replaying with:")
    print(f"  python scripts/tools/replay_demos.py --dataset_file {output_file} --task {new_env_name}")


if __name__ == "__main__":
    main()
