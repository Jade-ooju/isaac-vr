import json
import os

# JSON 파일 경로
json_path = r"C:\IsaacLab\IsaacLab\custom_scripts\datasets\NVIDIA\PhysicalAI-Robotics-GR00T-Teleop-G1\nvidia___physical_ai-robotics-gr00_t-teleop-g1\default\0.0.0\0d7bdd06e67f3ca0868892d0ec8f03bcd3e49e40\dataset_info.json"

# JSON 파일 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== NVIDIA PhysicalAI-Robotics-GR00T-Teleop-G1 데이터셋 정보 ===\n")

# 기본 정보
print("📋 기본 정보:")
print(f"  - 데이터셋 이름: {data.get('dataset_name', 'N/A')}")
print(f"  - 버전: {data.get('version', {}).get('version_str', 'N/A')}")
print(f"  - 빌더: {data.get('builder_name', 'N/A')}")
print(f"  - 설명: {data.get('description', 'N/A')}")
print(f"  - 라이선스: {data.get('license', 'N/A')}")
print(f"  - 홈페이지: {data.get('homepage', 'N/A')}")
print(f"  - 인용: {data.get('citation', 'N/A')}")

# Features 정보
print("\n🔍 데이터 Features:")
features = data.get('features', {})
for feature_name, feature_info in features.items():
    print(f"  - {feature_name}:")
    if isinstance(feature_info, dict):
        if 'feature' in feature_info:
            # List 타입
            dtype = feature_info['feature'].get('dtype', 'unknown')
            print(f"    타입: List[{dtype}]")
        else:
            # 단일 값 타입
            dtype = feature_info.get('dtype', 'unknown')
            print(f"    타입: {dtype}")

# Splits 정보
print("\n📊 데이터 분할:")
splits = data.get('splits', {})
for split_name, split_info in splits.items():
    print(f"  - {split_name}:")
    print(f"    예제 수: {split_info.get('num_examples', 'N/A'):,}")
    print(f"    크기: {split_info.get('num_bytes', 0) / (1024*1024):.2f} MB")

# 다운로드 체크섬 정보 (요약)
print("\n🔐 다운로드 정보:")
download_checksums = data.get('download_checksums', {})
print(f"  - 체크섬 파일 수: {len(download_checksums)}")

print("\n" + "="*60)
print("데이터셋 분석 완료!")
