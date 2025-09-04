import pyarrow.parquet as pq
import pandas as pd
import os

# Arrow 파일 경로
arrow_file = r"C:\IsaacLab\IsaacLab\custom_scripts\datasets\NVIDIA\PhysicalAI-Robotics-GR00T-Teleop-G1\nvidia___physical_ai-robotics-gr00_t-teleop-g1\default\0.0.0\0d7bdd06e67f3ca0868892d0ec8f03bcd3e49e40\physical_ai-robotics-gr00_t-teleop-g1-train.arrow"

print("=== 데이터셋 샘플 탐색 ===\n")

try:
    # Parquet 파일로 읽기
    table = pq.read_table(arrow_file)
    df = table.to_pandas()
    
    print("📊 데이터셋 기본 정보:")
    print(f"  - 총 행 수: {len(df):,}")
    print(f"  - 총 열 수: {len(df.columns)}")
    print(f"  - 메모리 사용량: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    print("\n🔍 컬럼 정보:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        print(f"  - {col}: {dtype} (non-null: {non_null:,})")
    
    print("\n📋 첫 5개 행:")
    print(df.head())
    
    print("\n📈 통계 정보:")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print("\n🎬 에피소드 정보:")
    if 'episode_index' in df.columns:
        episode_counts = df['episode_index'].value_counts().sort_index()
        print(f"  - 총 에피소드 수: {len(episode_counts)}")
        print(f"  - 에피소드 0의 프레임 수: {episode_counts.iloc[0] if len(episode_counts) > 0 else 0}")
        print(f"  - 평균 프레임 수: {episode_counts.mean():.1f}")
    
    if 'task_index' in df.columns:
        task_counts = df['task_index'].value_counts().sort_index()
        print(f"  - 총 태스크 수: {len(task_counts)}")
        print(f"  - 태스크별 분포: {dict(task_counts.head())}")
    
    print("\n✅ 데이터 탐색 완료!")
    
except Exception as e:
    print(f"❌ 에러 발생: {e}")
    print("Arrow 파일을 직접 읽어보겠습니다...")
    
    try:
        # Arrow 파일 직접 읽기
        import pyarrow as pa
        with pa.ipc.open_file(arrow_file) as reader:
            table = reader.read_all()
            print(f"테이블 스키마: {table.schema}")
            print(f"총 행 수: {len(table)}")
    except Exception as e2:
        print(f"Arrow 파일 읽기 실패: {e2}")
