#!/usr/bin/env python3
"""
NVIDIA 데이터셋을 IsaacLab에서 활용하는 예제 스크립트

이 스크립트는 NVIDIA 데이터셋을 IsaacLab에서 어떻게 활용할 수 있는지
보여주는 예제들을 포함합니다.

사용법:
    python example_usage.py --demo all
    python example_usage.py --demo convert_teleop
    python example_usage.py --demo analyze_eval
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NVIDIA dataset usage examples for IsaacLab")
    parser.add_argument(
        "--demo", 
        type=str, 
        choices=["all", "convert_teleop", "analyze_eval", "replay_converted"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./datasets",
        help="Output directory for converted datasets"
    )
    return parser.parse_args()


def run_command(cmd: str, description: str):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Failed!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ Exception: {e}")


def demo_convert_teleop_dataset(args):
    """GR00T-Teleop 데이터셋 변환 데모"""
    print("\n🎬 Demo 1: Converting GR00T-Teleop dataset to IsaacLab HDF5 format")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 변환 명령어
    output_file = os.path.join(args.output_dir, "gr00t_teleop_demo.hdf5")
    cmd = f"""python custom_scripts/datasets/nvidia_dataset_integration.py \\
        --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \\
        --output_file {output_file} \\
        --max_episodes 5 \\
        --env_name GR00T-Teleop-Demo"""
    
    run_command(cmd, "Converting GR00T-Teleop dataset (limited to 5 episodes for demo)")
    
    # 변환된 파일 확인
    if os.path.exists(output_file):
        print(f"\n📁 Converted dataset saved to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    else:
        print(f"\n❌ Conversion failed - output file not found: {output_file}")


def demo_analyze_eval_dataset(args):
    """GR00T-Eval 데이터셋 분석 데모"""
    print("\n🎬 Demo 2: Analyzing GR00T-Eval dataset")
    
    # 분석 명령어
    cmd = """python custom_scripts/datasets/gr00t_eval_integration.py \\
        --analyze_tasks \\
        --export_task_list \\
        --output_file ./datasets/gr00t_tasks_demo.json"""
    
    run_command(cmd, "Analyzing GR00T-Eval dataset and exporting task list")
    
    # 내보낸 파일 확인
    task_file = "./datasets/gr00t_tasks_demo.json"
    if os.path.exists(task_file):
        print(f"\n📁 Task list exported to: {task_file}")
        print(f"   File size: {os.path.getsize(task_file) / 1024:.2f} KB")
        
        # JSON 파일 내용 일부 출력
        try:
            import json
            with open(task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   Total tasks: {data['dataset_info']['total_tasks']}")
            print(f"   Categories: {list(data['categories'].keys())}")
        except Exception as e:
            print(f"   Error reading task file: {e}")
    else:
        print(f"\n❌ Analysis failed - task file not found: {task_file}")


def demo_replay_converted_dataset(args):
    """변환된 데이터셋 재생 데모"""
    print("\n🎬 Demo 3: Replaying converted dataset with IsaacLab")
    
    # 변환된 데이터셋 파일 경로
    dataset_file = os.path.join(args.output_dir, "gr00t_teleop_demo.hdf5")
    
    if not os.path.exists(dataset_file):
        print(f"❌ Converted dataset not found: {dataset_file}")
        print("   Please run the convert_teleop demo first.")
        return
    
    # IsaacLab 재생 명령어 (headless 모드)
    cmd = f"""python scripts/tools/replay_demos.py \\
        --dataset_file {dataset_file} \\
        --headless \\
        --num_envs 1"""
    
    print(f"Note: This will launch IsaacLab simulator to replay the converted dataset.")
    print(f"Press Ctrl+C to stop the replay.")
    
    run_command(cmd, "Replaying converted dataset with IsaacLab")


def demo_all(args):
    """모든 데모 실행"""
    print("🚀 Running all NVIDIA dataset integration demos")
    
    # 1. 데이터셋 변환
    demo_convert_teleop_dataset(args)
    
    # 2. 평가 데이터셋 분석
    demo_analyze_eval_dataset(args)
    
    # 3. 변환된 데이터셋 재생 (선택사항)
    print(f"\n{'='*60}")
    print("Demo 3: Replay converted dataset")
    print("This requires IsaacLab simulator to be available.")
    print("Uncomment the line below to run this demo:")
    print("# demo_replay_converted_dataset(args)")
    print(f"{'='*60}")


def show_usage_guide():
    """사용 가이드를 출력합니다."""
    print(f"\n{'='*60}")
    print("📚 NVIDIA Dataset Integration Usage Guide")
    print(f"{'='*60}")
    
    print("""
1. 데이터셋 변환 (GR00T-Teleop):
   python custom_scripts/datasets/nvidia_dataset_integration.py \\
       --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \\
       --output_file ./datasets/gr00t_teleop.hdf5

2. 평가 데이터셋 분석 (GR00T-Eval):
   python custom_scripts/datasets/gr00t_eval_integration.py \\
       --analyze_tasks \\
       --export_task_list \\
       --output_file ./datasets/gr00t_tasks.json

3. 변환된 데이터셋 재생:
   python scripts/tools/replay_demos.py \\
       --dataset_file ./datasets/gr00t_teleop.hdf5

4. 모방학습 훈련 (예시):
   python scripts/imitation_learning/robomimic/train.py \\
       --config ./configs/gr00t_config.yaml \\
       --dataset ./datasets/gr00t_teleop.hdf5

5. IsaacLab 환경에서 활용:
   - 변환된 HDF5 파일을 IsaacLab의 기존 파이프라인에서 직접 사용
   - GR00T-Eval 태스크를 IsaacLab 환경 설정에 통합
   - 커스텀 태스크 생성 시 참고 데이터로 활용
""")


def main():
    """메인 함수"""
    args = parse_args()
    
    print("🎯 NVIDIA Dataset Integration Examples for IsaacLab")
    print("=" * 60)
    
    # 데모 실행
    if args.demo == "convert_teleop":
        demo_convert_teleop_dataset(args)
    elif args.demo == "analyze_eval":
        demo_analyze_eval_dataset(args)
    elif args.demo == "replay_converted":
        demo_replay_converted_dataset(args)
    elif args.demo == "all":
        demo_all(args)
    
    # 사용 가이드 출력
    show_usage_guide()
    
    print(f"\n✅ Demo completed!")
    print(f"\nNext steps:")
    print(f"  1. Check the output files in: {args.output_dir}")
    print(f"  2. Use the converted datasets with IsaacLab's imitation learning pipeline")
    print(f"  3. Integrate GR00T-Eval tasks into your IsaacLab environments")


if __name__ == "__main__":
    main()
