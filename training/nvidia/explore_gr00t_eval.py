from datasets import load_dataset

print("=== NVIDIA PhysicalAI-Robotics-GR00T-Eval 데이터셋 탐색 ===\n")

# 데이터셋 로드
dataset_name = "nvidia/PhysicalAI-Robotics-GR00T-Eval"
dataset = load_dataset(dataset_name, cache_dir=r"C:\IsaacLab\IsaacLab\custom_scripts\datasets\NVIDIA\PhysicalAI-Robotics-GR00T-Eval")

print("📊 데이터셋 기본 정보:")
print(f"  - 데이터셋 타입: {type(dataset)}")
print(f"  - 스플릿: {list(dataset.keys())}")

train_data = dataset['train']
print(f"  - 훈련 데이터 크기: {len(train_data):,}")

print("\n🔍 Features 정보:")
for feature_name, feature_info in train_data.features.items():
    print(f"  - {feature_name}: {feature_info}")

print("\n📋 샘플 태스크들:")
for i in range(min(10, len(train_data))):
    print(f"  {i+1}. {train_data[i]['text']}")

print(f"\n📈 전체 태스크 수: {len(train_data)}")
print("✅ 데이터 탐색 완료!")
