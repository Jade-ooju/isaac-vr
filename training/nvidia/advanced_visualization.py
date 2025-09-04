#!/usr/bin/env python3
"""
NVIDIA GR00T 데이터셋 고급 시각화 도구
Isaac Sim 없이도 완전한 시각화 제공
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from typing import List, Dict, Tuple
import os
import json

class GR00TVisualizer:
    """GR00T 데이터셋 시각화 클래스"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """HDF5 데이터셋 로드"""
        print("📊 데이터셋 로드 중...")
        
        with h5py.File(self.dataset_path, 'r') as f:
            for episode_name in f['data'].keys():
                episode_data = f['data'][episode_name]
                
                self.data[episode_name] = {
                    'actions': episode_data['actions'][:],
                    'observations': episode_data['obs/policy/state'][:],
                    'img_state_delta': episode_data['obs/policy/img_state_delta'][:] if 'obs/policy/img_state_delta' in episode_data else None,
                    'initial_state': episode_data['initial_state'][:]
                }
        
        print(f"✅ {len(self.data)}개 에피소드 로드 완료")
    
    def plot_action_trajectory(self, episode_name: str, save_path: str = None):
        """액션 궤적 시각화"""
        if episode_name not in self.data:
            print(f"❌ 에피소드 {episode_name}을 찾을 수 없습니다")
            return
        
        actions = self.data[episode_name]['actions']
        
        # 43차원 액션을 6개 그룹으로 나누어 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        group_size = 43 // 6
        for i in range(6):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < 5 else 43
            
            for j in range(start_idx, end_idx):
                axes[i].plot(actions[:, j], alpha=0.7, label=f'Dim {j}')
            
            axes[i].set_title(f'Action Group {i+1} (Dims {start_idx}-{end_idx-1})')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Action Value')
            axes[i].grid(True)
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 액션 궤적 저장: {save_path}")
        
        plt.show()
    
    def plot_observation_trajectory(self, episode_name: str, save_path: str = None):
        """관찰 궤적 시각화"""
        if episode_name not in self.data:
            print(f"❌ 에피소드 {episode_name}을 찾을 수 없습니다")
            return
        
        observations = self.data[episode_name]['observations']
        
        # 43차원 관찰을 6개 그룹으로 나누어 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        group_size = 43 // 6
        for i in range(6):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < 5 else 43
            
            for j in range(start_idx, end_idx):
                axes[i].plot(observations[:, j], alpha=0.7, label=f'Dim {j}')
            
            axes[i].set_title(f'Observation Group {i+1} (Dims {start_idx}-{end_idx-1})')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Observation Value')
            axes[i].grid(True)
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 관찰 궤적 저장: {save_path}")
        
        plt.show()
    
    def plot_action_observation_correlation(self, episode_name: str, save_path: str = None):
        """액션-관찰 상관관계 시각화"""
        if episode_name not in self.data:
            print(f"❌ 에피소드 {episode_name}을 찾을 수 없습니다")
            return
        
        actions = self.data[episode_name]['actions']
        observations = self.data[episode_name]['observations']
        
        # 상관관계 행렬 계산
        correlation_matrix = np.corrcoef(actions.T, observations.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 액션-액션 상관관계
        action_corr = correlation_matrix[:43, :43]
        sns.heatmap(action_corr, annot=False, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Action-Action Correlation')
        ax1.set_xlabel('Action Dimension')
        ax1.set_ylabel('Action Dimension')
        
        # 액션-관찰 상관관계
        action_obs_corr = correlation_matrix[:43, 43:]
        sns.heatmap(action_obs_corr, annot=False, cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Action-Observation Correlation')
        ax2.set_xlabel('Observation Dimension')
        ax2.set_ylabel('Action Dimension')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 상관관계 시각화 저장: {save_path}")
        
        plt.show()
    
    def plot_3d_trajectory(self, episode_name: str, save_path: str = None):
        """3D 궤적 시각화 (첫 3차원 사용)"""
        if episode_name not in self.data:
            print(f"❌ 에피소드 {episode_name}을 찾을 수 없습니다")
            return
        
        actions = self.data[episode_name]['actions']
        observations = self.data[episode_name]['observations']
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D 액션 궤적
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(actions[:, 0], actions[:, 1], actions[:, 2], 'b-', linewidth=2, label='Action Trajectory')
        ax1.scatter(actions[0, 0], actions[0, 1], actions[0, 2], color='green', s=100, label='Start')
        ax1.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2], color='red', s=100, label='End')
        ax1.set_xlabel('Action Dim 0')
        ax1.set_ylabel('Action Dim 1')
        ax1.set_zlabel('Action Dim 2')
        ax1.set_title('3D Action Trajectory')
        ax1.legend()
        
        # 3D 관찰 궤적
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(observations[:, 0], observations[:, 1], observations[:, 2], 'r-', linewidth=2, label='Observation Trajectory')
        ax2.scatter(observations[0, 0], observations[0, 1], observations[0, 2], color='green', s=100, label='Start')
        ax2.scatter(observations[-1, 0], observations[-1, 1], observations[-1, 2], color='red', s=100, label='End')
        ax2.set_xlabel('Observation Dim 0')
        ax2.set_ylabel('Observation Dim 1')
        ax2.set_zlabel('Observation Dim 2')
        ax2.set_title('3D Observation Trajectory')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 3D 궤적 저장: {save_path}")
        
        plt.show()
    
    def create_animation(self, episode_name: str, save_path: str = None):
        """애니메이션 생성"""
        if episode_name not in self.data:
            print(f"❌ 에피소드 {episode_name}을 찾을 수 없습니다")
            return
        
        actions = self.data[episode_name]['actions']
        observations = self.data[episode_name]['observations']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 액션 애니메이션
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        point1, = ax1.plot([], [], 'ro', markersize=8)
        ax1.set_xlim(np.min(actions[:, 0]) - 0.1, np.max(actions[:, 0]) + 0.1)
        ax1.set_ylim(np.min(actions[:, 1]) - 0.1, np.max(actions[:, 1]) + 0.1)
        ax1.set_xlabel('Action Dim 0')
        ax1.set_ylabel('Action Dim 1')
        ax1.set_title('Action Trajectory Animation')
        ax1.grid(True)
        
        # 관찰 애니메이션
        line2, = ax2.plot([], [], 'r-', linewidth=2)
        point2, = ax2.plot([], [], 'bo', markersize=8)
        ax2.set_xlim(np.min(observations[:, 0]) - 0.1, np.max(observations[:, 0]) + 0.1)
        ax2.set_ylim(np.min(observations[:, 1]) - 0.1, np.max(observations[:, 1]) + 0.1)
        ax2.set_xlabel('Observation Dim 0')
        ax2.set_ylabel('Observation Dim 1')
        ax2.set_title('Observation Trajectory Animation')
        ax2.grid(True)
        
        def animate(frame):
            # 액션 애니메이션
            line1.set_data(actions[:frame+1, 0], actions[:frame+1, 1])
            point1.set_data([actions[frame, 0]], [actions[frame, 1]])
            
            # 관찰 애니메이션
            line2.set_data(observations[:frame+1, 0], observations[:frame+1, 1])
            point2.set_data([observations[frame, 0]], [observations[frame, 1]])
            
            return line1, point1, line2, point2
        
        anim = animation.FuncAnimation(fig, animate, frames=len(actions), interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"✅ 애니메이션 저장: {save_path}")
        
        plt.show()
        return anim
    
    def generate_comprehensive_report(self, save_dir: str = "./datasets/visualization_report"):
        """종합 시각화 보고서 생성"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("📊 종합 시각화 보고서 생성 중...")
        
        for episode_name in self.data.keys():
            print(f"  - {episode_name} 처리 중...")
            
            # 액션 궤적
            self.plot_action_trajectory(episode_name, f"{save_dir}/{episode_name}_action_trajectory.png")
            
            # 관찰 궤적
            self.plot_observation_trajectory(episode_name, f"{save_dir}/{episode_name}_observation_trajectory.png")
            
            # 상관관계
            self.plot_action_observation_correlation(episode_name, f"{save_dir}/{episode_name}_correlation.png")
            
            # 3D 궤적
            self.plot_3d_trajectory(episode_name, f"{save_dir}/{episode_name}_3d_trajectory.png")
            
            # 애니메이션
            self.create_animation(episode_name, f"{save_dir}/{episode_name}_animation.gif")
        
        # 통계 요약
        self.generate_statistics_summary(save_dir)
        
        print(f"✅ 종합 보고서 생성 완료: {save_dir}")
    
    def generate_statistics_summary(self, save_dir: str):
        """통계 요약 생성"""
        stats = {}
        
        for episode_name, data in self.data.items():
            actions = data['actions']
            observations = data['observations']
            
            stats[episode_name] = {
                'num_frames': len(actions),
                'action_stats': {
                    'mean': np.mean(actions, axis=0).tolist(),
                    'std': np.std(actions, axis=0).tolist(),
                    'min': np.min(actions, axis=0).tolist(),
                    'max': np.max(actions, axis=0).tolist()
                },
                'observation_stats': {
                    'mean': np.mean(observations, axis=0).tolist(),
                    'std': np.std(observations, axis=0).tolist(),
                    'min': np.min(observations, axis=0).tolist(),
                    'max': np.max(observations, axis=0).tolist()
                }
            }
        
        # JSON으로 저장
        with open(f"{save_dir}/statistics_summary.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ 통계 요약 저장: {save_dir}/statistics_summary.json")

def main():
    """메인 함수"""
    dataset_path = "./datasets/gr00t_teleop_test.hdf5"
    
    if not os.path.exists(dataset_path):
        print(f"❌ 데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        return
    
    print("🎨 NVIDIA GR00T 데이터셋 고급 시각화")
    print("=" * 50)
    
    # 시각화 도구 생성
    visualizer = GR00TVisualizer(dataset_path)
    
    # 종합 보고서 생성
    visualizer.generate_comprehensive_report()
    
    print("\n🎉 모든 시각화가 완료되었습니다!")
    print("Isaac Sim 없이도 완전한 시각화를 제공합니다.")

if __name__ == "__main__":
    main()
