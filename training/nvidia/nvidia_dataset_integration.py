#!/usr/bin/env python3
"""
NVIDIA 데이터셋을 IsaacLab HDF5 형식으로 변환하는 스크립트

이 스크립트는 NVIDIA의 GR00T-Teleop-G1 데이터셋을 IsaacLab에서 사용할 수 있는
HDF5 형식으로 변환합니다. 변환된 데이터는 IsaacLab의 모방학습 파이프라인에서
직접 사용할 수 있습니다.

사용법:
    python nvidia_dataset_integration.py --input_dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 --output_file ./datasets/gr00t_teleop.hdf5
"""

import argparse
import os
import torch
import numpy as np
import h5py
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# IsaacLab imports
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert NVIDIA datasets to IsaacLab HDF5 format")
    parser.add_argument(
        "--input_dataset", 
        type=str, 
        default="nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
        help="Name of the NVIDIA dataset to convert"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="./datasets/gr00t_teleop.hdf5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Cache directory for datasets (optional)"
    )
    parser.add_argument(
        "--max_episodes", 
        type=int, 
        default=None,
        help="Maximum number of episodes to convert (for testing)"
    )
    parser.add_argument(
        "--env_name", 
        type=str, 
        default="GR00T-Teleop",
        help="Environment name for the converted dataset"
    )
    return parser.parse_args()


class NVIDIADatasetConverter:
    """NVIDIA 데이터셋을 IsaacLab HDF5 형식으로 변환하는 클래스"""
    
    def __init__(self, args):
        self.args = args
        # CUDA 사용 가능 여부 확인 및 오류 처리
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"CUDA initialization failed, using CPU: {e}")
            self.device = torch.device("cpu")
        
    def load_nvidia_dataset(self) -> Any:
        """NVIDIA 데이터셋을 로드합니다."""
        print(f"Loading NVIDIA dataset: {self.args.input_dataset}")
        
        cache_dir = self.args.cache_dir
        if cache_dir is None:
            # 기본 캐시 디렉토리 설정
            cache_dir = os.path.join(os.path.dirname(__file__), "NVIDIA")
        
        dataset = load_dataset(
            self.args.input_dataset, 
            cache_dir=cache_dir
        )
        
        print(f"Dataset loaded successfully!")
        print(f"  - Split: {list(dataset.keys())}")
        print(f"  - Train data size: {len(dataset['train']):,}")
        
        return dataset['train']
    
    def convert_episode_to_isaaclab_format(self, episode_frames: List[Dict], episode_idx: int) -> EpisodeData:
        """단일 에피소드를 IsaacLab 형식으로 변환합니다."""
        episode = EpisodeData()
        
        # 기본 메타데이터 설정
        episode.seed = episode_idx
        episode.success = True  # NVIDIA 데이터는 모두 성공한 데모로 가정
        
        if not episode_frames:
            raise ValueError(f"Episode {episode_idx} has no frames")
        
        # 초기 상태 설정 (첫 번째 프레임의 observation)
        initial_state = torch.tensor(
            episode_frames[0]['observation.state'], 
            dtype=torch.float32, 
            device=self.device
        )
        episode.add("initial_state", initial_state)
        
        # 액션과 관찰 데이터 추가
        for frame in episode_frames:
            # 액션 데이터
            action = torch.tensor(
                frame['action'], 
                dtype=torch.float32, 
                device=self.device
            )
            episode.add("actions", action)
            
            # 관찰 데이터 (policy용)
            obs_state = torch.tensor(
                frame['observation.state'], 
                dtype=torch.float32, 
                device=self.device
            )
            episode.add("obs/policy/state", obs_state)
            
            # 이미지 상태 변화 (있는 경우)
            if 'observation.img_state_delta' in frame:
                img_delta = torch.tensor(
                    frame['observation.img_state_delta'], 
                    dtype=torch.float32, 
                    device=self.device
                )
                episode.add("obs/policy/img_state_delta", img_delta)
        
        return episode
    
    def group_frames_by_episode(self, dataset) -> List[Dict]:
        """데이터셋의 프레임들을 에피소드별로 그룹화합니다."""
        print("Grouping frames by episode...")
        
        episodes = {}
        for frame in dataset:
            episode_idx = frame['episode_index']
            if episode_idx not in episodes:
                episodes[episode_idx] = []
            episodes[episode_idx].append(frame)
        
        # 에피소드를 인덱스 순으로 정렬
        sorted_episodes = []
        for episode_idx in sorted(episodes.keys()):
            episode_frames = episodes[episode_idx]
            # 프레임을 timestamp 순으로 정렬
            episode_frames.sort(key=lambda x: x['timestamp'])
            sorted_episodes.append({
                'episode_index': episode_idx,
                'frames': episode_frames
            })
        
        print(f"Found {len(sorted_episodes)} episodes")
        return sorted_episodes
    
    def convert_dataset(self):
        """전체 데이터셋을 변환합니다."""
        # NVIDIA 데이터셋 로드
        nvidia_dataset = self.load_nvidia_dataset()
        
        # 에피소드별로 그룹화
        episodes = self.group_frames_by_episode(nvidia_dataset)
        
        # 최대 에피소드 수 제한 (테스트용)
        if self.args.max_episodes:
            episodes = episodes[:self.args.max_episodes]
            print(f"Limited to {len(episodes)} episodes for testing")
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # HDF5 데이터셋 파일 생성
        print(f"Creating HDF5 dataset: {self.args.output_file}")
        dataset_handler = HDF5DatasetFileHandler()
        dataset_handler.create(
            os.path.splitext(self.args.output_file)[0],  # 확장자 제거
            env_name=self.args.env_name
        )
        
        # 각 에피소드를 변환하여 저장
        successful_episodes = 0
        failed_episodes = 0
        
        for i, episode_data in enumerate(episodes):
            try:
                print(f"Converting episode {i+1}/{len(episodes)} (episode_index: {episode_data['episode_index']})")
                
                # 에피소드 변환
                isaaclab_episode = self.convert_episode_to_isaaclab_format(
                    episode_data['frames'], 
                    episode_data['episode_index']
                )
                
                # HDF5 파일에 저장
                dataset_handler.write_episode(isaaclab_episode)
                successful_episodes += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1} episodes...")
                    
            except Exception as e:
                print(f"  Failed to convert episode {episode_data['episode_index']}: {e}")
                failed_episodes += 1
                continue
        
        # 데이터셋 파일 닫기
        dataset_handler.flush()
        dataset_handler.close()
        
        print(f"\nConversion completed!")
        print(f"  - Successful episodes: {successful_episodes}")
        print(f"  - Failed episodes: {failed_episodes}")
        print(f"  - Output file: {self.args.output_file}")
        
        # 변환 정보 저장
        conversion_info = {
            "source_dataset": self.args.input_dataset,
            "output_file": self.args.output_file,
            "env_name": self.args.env_name,
            "total_episodes": len(episodes),
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "conversion_timestamp": datetime.now().isoformat()
        }
        
        info_file = os.path.splitext(self.args.output_file)[0] + "_conversion_info.json"
        with open(info_file, 'w') as f:
            json.dump(conversion_info, f, indent=2)
        print(f"  - Conversion info saved to: {info_file}")


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("NVIDIA Dataset to IsaacLab HDF5 Converter")
    print("=" * 60)
    print(f"Input dataset: {args.input_dataset}")
    print(f"Output file: {args.output_file}")
    print(f"Environment name: {args.env_name}")
    if args.max_episodes:
        print(f"Max episodes: {args.max_episodes}")
    print("=" * 60)
    
    # 변환기 생성 및 실행
    converter = NVIDIADatasetConverter(args)
    converter.convert_dataset()
    
    print("\n✅ Conversion completed successfully!")
    print(f"\nYou can now use the converted dataset with IsaacLab:")
    print(f"  - Replay: python scripts/tools/replay_demos.py --dataset_file {args.output_file}")
    print(f"  - Train: Use with imitation learning scripts in scripts/imitation_learning/")


if __name__ == "__main__":
    main()
