from datasets import load_dataset

print("=== NVIDIA PhysicalAI-Robotics-GR00T-Teleop-G1 데이터셋 탐색 ===\n")

# 데이터셋 로드 (캐시된 데이터 사용)
dataset_name = "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1"
dataset = load_dataset(dataset_name, cache_dir=r"C:\IsaacLab\IsaacLab\custom_scripts\datasets\NVIDIA\PhysicalAI-Robotics-GR00T-Teleop-G1")

print("📊 데이터셋 기본 정보:")
print(f"  - 데이터셋 타입: {type(dataset)}")
print(f"  - 스플릿: {list(dataset.keys())}")

train_data = dataset['train']
print(f"  - 훈련 데이터 크기: {len(train_data):,}")

print("\n🔍 Features 정보:")
for feature_name, feature_info in train_data.features.items():
    print(f"  - {feature_name}: {feature_info}")

print("\n📋 첫 번째 예제:")
first_example = train_data[0]
for key, value in first_example.items():
    if isinstance(value, list):
        print(f"  - {key}: List (길이: {len(value)})")
        if len(value) > 0:
            print(f"    첫 번째 값: {value[0]}")
            if len(value) > 3:
                print(f"    ... (총 {len(value)}개 값)")
    else:
        print(f"  - {key}: {value}")

print("\n📈 통계 정보:")
# 에피소드 통계
episode_indices = [train_data[i]['episode_index'] for i in range(len(train_data))]
unique_episodes = set(episode_indices)
print(f"  - 총 에피소드 수: {len(unique_episodes)}")
print(f"  - 에피소드 인덱스 범위: {min(episode_indices)} ~ {max(episode_indices)}")

# 태스크 통계
task_indices = [train_data[i]['task_index'] for i in range(len(train_data))]
unique_tasks = set(task_indices)
print(f"  - 총 태스크 수: {len(unique_tasks)}")
print(f"  - 태스크 인덱스 범위: {min(task_indices)} ~ {max(task_indices)}")

# 액션 차원 확인
action_dims = [len(train_data[i]['action']) for i in range(min(100, len(train_data)))]
print(f"  - 액션 차원 (샘플 100개): {set(action_dims)}")

# 상태 차원 확인
state_dims = [len(train_data[i]['observation.state']) for i in range(min(100, len(train_data)))]
print(f"  - 상태 차원 (샘플 100개): {set(state_dims)}")

print("\n🎬 특정 에피소드 분석:")
# 에피소드 0의 데이터 찾기
episode_0_indices = [i for i, ep_idx in enumerate(episode_indices) if ep_idx == 0]
if episode_0_indices:
    print(f"  - 에피소드 0의 프레임 수: {len(episode_0_indices)}")
    first_frame = train_data[episode_0_indices[0]]
    last_frame = train_data[episode_0_indices[-1]]
    print(f"  - 첫 프레임 timestamp: {first_frame['timestamp']}")
    print(f"  - 마지막 프레임 timestamp: {last_frame['timestamp']}")
    print(f"  - 에피소드 지속 시간: {last_frame['timestamp'] - first_frame['timestamp']:.2f}초")

print("\n✅ 데이터 탐색 완료!")
