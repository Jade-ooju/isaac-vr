from datasets import load_dataset
import numpy as np

# 데이터셋 로드
dataset_path = r"C:\IsaacLab\IsaacLab\custom_scripts\datasets\NVIDIA\PhysicalAI-Robotics-GR00T-Teleop-G1"
dataset = load_dataset("parquet", data_dir=dataset_path)

print("=== 데이터셋 샘플 탐색 ===\n")

# 첫 번째 예제 확인
first_example = dataset['train'][0]
print("🔍 첫 번째 예제:")
for key, value in first_example.items():
    if isinstance(value, list):
        print(f"  - {key}: List (길이: {len(value)})")
        if len(value) > 0:
            print(f"    첫 번째 값: {value[0]}")
            if len(value) > 1:
                print(f"    두 번째 값: {value[1]}")
    else:
        print(f"  - {key}: {type(value).__name__} = {value}")

print("\n" + "="*50)

# 여러 예제의 통계 정보
print("\n📊 데이터 통계:")
print(f"  - 총 예제 수: {len(dataset['train']):,}")

# episode_index 범위 확인
episode_indices = [dataset['train'][i]['episode_index'] for i in range(min(1000, len(dataset['train'])))]
print(f"  - 에피소드 인덱스 범위 (샘플 1000개): {min(episode_indices)} ~ {max(episode_indices)}")

# task_index 범위 확인
task_indices = [dataset['train'][i]['task_index'] for i in range(min(1000, len(dataset['train'])))]
print(f"  - 태스크 인덱스 범위 (샘플 1000개): {min(task_indices)} ~ {max(task_indices)}")

# action 차원 확인
action_dims = [len(dataset['train'][i]['action']) for i in range(min(10, len(dataset['train'])))]
print(f"  - 액션 차원 (샘플 10개): {action_dims}")

# observation.state 차원 확인
state_dims = [len(dataset['train'][i]['observation.state']) for i in range(min(10, len(dataset['train'])))]
print(f"  - 상태 차원 (샘플 10개): {state_dims}")

print("\n" + "="*50)

# 특정 에피소드의 데이터 확인
print("\n🎬 특정 에피소드 분석 (episode_index=0):")
episode_0_data = [dataset['train'][i] for i in range(len(dataset['train'])) if dataset['train'][i]['episode_index'] == 0]
print(f"  - 에피소드 0의 프레임 수: {len(episode_0_data)}")

if len(episode_0_data) > 0:
    print(f"  - 첫 번째 프레임의 timestamp: {episode_0_data[0]['timestamp']}")
    print(f"  - 마지막 프레임의 timestamp: {episode_0_data[-1]['timestamp']}")
    print(f"  - 에피소드 지속 시간: {episode_0_data[-1]['timestamp'] - episode_0_data[0]['timestamp']:.2f}초")

print("\n✅ 데이터 탐색 완료!")
