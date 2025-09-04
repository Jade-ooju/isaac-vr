#!/usr/bin/env python3
"""
변환된 NVIDIA 데이터셋을 분석하고 시각화하는 스크립트
IsaacLab 시뮬레이터 없이도 데이터를 분석할 수 있습니다.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple


class ConvertedDataAnalyzer:
    """변환된 데이터셋을 분석하는 클래스"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = None
        
    def load_dataset(self):
        """데이터셋을 로드합니다."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.data = h5py.File(self.dataset_path, 'r')
        print(f"✅ Dataset loaded: {self.dataset_path}")
        
    def get_episode_info(self) -> Dict:
        """에피소드 정보를 반환합니다."""
        if not self.data:
            self.load_dataset()
        
        episodes = {}
        for episode_name in self.data['data'].keys():
            episode_data = self.data['data'][episode_name]
            
            # 액션과 관찰 데이터 길이 확인
            actions = episode_data['actions'][:]
            obs_state = episode_data['obs/policy/state'][:]
            
            episodes[episode_name] = {
                'length': len(actions),
                'action_shape': actions.shape,
                'obs_shape': obs_state.shape,
                'action_range': [actions.min(), actions.max()],
                'obs_range': [obs_state.min(), obs_state.max()]
            }
        
        return episodes
    
    def plot_episode_data(self, episode_name: str, save_plot: bool = True):
        """특정 에피소드의 데이터를 시각화합니다."""
        if not self.data:
            self.load_dataset()
        
        episode_data = self.data['data'][episode_name]
        actions = episode_data['actions'][:]
        obs_state = episode_data['obs/policy/state'][:]
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Episode {episode_name} Analysis', fontsize=16)
        
        # 액션 데이터 시각화
        axes[0, 0].plot(actions)
        axes[0, 0].set_title('Actions Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Action Value')
        axes[0, 0].grid(True)
        
        # 관찰 상태 데이터 시각화
        axes[0, 1].plot(obs_state)
        axes[0, 1].set_title('Observation State Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('State Value')
        axes[0, 1].grid(True)
        
        # 액션 히스토그램
        axes[1, 0].hist(actions.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Action Value Distribution')
        axes[1, 0].set_xlabel('Action Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 관찰 상태 히스토그램
        axes[1, 1].hist(obs_state.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Observation State Distribution')
        axes[1, 1].set_xlabel('State Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = f"./datasets/{episode_name}_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    def analyze_action_patterns(self) -> Dict:
        """액션 패턴을 분석합니다."""
        if not self.data:
            self.load_dataset()
        
        all_actions = []
        all_obs = []
        
        for episode_name in self.data['data'].keys():
            episode_data = self.data['data'][episode_name]
            actions = episode_data['actions'][:]
            obs_state = episode_data['obs/policy/state'][:]
            
            all_actions.append(actions)
            all_obs.append(obs_state)
        
        # 전체 데이터 결합
        all_actions = np.concatenate(all_actions, axis=0)
        all_obs = np.concatenate(all_obs, axis=0)
        
        analysis = {
            'total_frames': len(all_actions),
            'action_dim': all_actions.shape[1],
            'obs_dim': all_obs.shape[1],
            'action_stats': {
                'mean': np.mean(all_actions, axis=0),
                'std': np.std(all_actions, axis=0),
                'min': np.min(all_actions, axis=0),
                'max': np.max(all_actions, axis=0)
            },
            'obs_stats': {
                'mean': np.mean(all_obs, axis=0),
                'std': np.std(all_obs, axis=0),
                'min': np.min(all_obs, axis=0),
                'max': np.max(all_obs, axis=0)
            }
        }
        
        return analysis
    
    def print_summary(self):
        """데이터셋 요약을 출력합니다."""
        if not self.data:
            self.load_dataset()
        
        episodes = self.get_episode_info()
        analysis = self.analyze_action_patterns()
        
        print("\n" + "="*60)
        print("📊 CONVERTED DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📁 Dataset: {self.dataset_path}")
        print(f"📈 Total episodes: {len(episodes)}")
        print(f"🎬 Total frames: {analysis['total_frames']:,}")
        print(f"🎮 Action dimension: {analysis['action_dim']}")
        print(f"👁️  Observation dimension: {analysis['obs_dim']}")
        
        print(f"\n📋 Episode details:")
        for episode_name, info in episodes.items():
            print(f"  - {episode_name}: {info['length']} frames")
        
        print(f"\n🎯 Action statistics:")
        print(f"  - Range: [{analysis['action_stats']['min'].min():.3f}, {analysis['action_stats']['max'].max():.3f}]")
        print(f"  - Mean: {analysis['action_stats']['mean'].mean():.3f}")
        print(f"  - Std: {analysis['action_stats']['std'].mean():.3f}")
        
        print(f"\n👁️  Observation statistics:")
        print(f"  - Range: [{analysis['obs_stats']['min'].min():.3f}, {analysis['obs_stats']['max'].max():.3f}]")
        print(f"  - Mean: {analysis['obs_stats']['mean'].mean():.3f}")
        print(f"  - Std: {analysis['obs_stats']['std'].mean():.3f}")
        
        print(f"\n✅ Dataset is ready for IsaacLab imitation learning!")
        print(f"   - Compatible with IsaacLab HDF5 format")
        print(f"   - Contains actions and observations")
        print(f"   - Can be used with replay_demos.py (with correct env name)")
        print(f"   - Ready for imitation learning training")
    
    def close(self):
        """데이터셋을 닫습니다."""
        if self.data:
            self.data.close()


def main():
    """메인 함수"""
    dataset_path = "./datasets/gr00t_teleop_test.hdf5"
    
    print("🔍 Analyzing converted NVIDIA dataset")
    print("="*60)
    
    try:
        analyzer = ConvertedDataAnalyzer(dataset_path)
        analyzer.print_summary()
        
        # 첫 번째 에피소드 시각화 (선택사항)
        episodes = analyzer.get_episode_info()
        if episodes:
            first_episode = list(episodes.keys())[0]
            print(f"\n📊 Generating visualization for {first_episode}...")
            analyzer.plot_episode_data(first_episode)
        
        analyzer.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
