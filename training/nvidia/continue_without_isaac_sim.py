#!/usr/bin/env python3
"""
Isaac Sim 없이도 계속할 수 있는 작업들
NVIDIA 데이터셋 활용 및 모방학습 연구
"""

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict
import os

def analyze_converted_dataset():
    """변환된 데이터셋 분석"""
    dataset_path = "./datasets/gr00t_teleop_test.hdf5"
    
    if not os.path.exists(dataset_path):
        print(f"❌ 데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        return
    
    print("📊 변환된 NVIDIA 데이터셋 분석")
    print("=" * 50)
    
    with h5py.File(dataset_path, 'r') as f:
        # 데이터셋 구조 분석
        print("📁 데이터셋 구조:")
        def print_structure(name, obj):
            indent = "  " * (name.count('/') + 1)
            if isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name.split('/')[-1]}/")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {name.split('/')[-1]}: {obj.shape} ({obj.dtype})")
        f.visititems(print_structure)
        
        # 에피소드 정보
        episodes = list(f['data'].keys())
        print(f"\n📈 에피소드 정보:")
        print(f"  - 총 에피소드 수: {len(episodes)}")
        
        total_frames = 0
        for episode_name in episodes:
            episode_data = f['data'][episode_name]
            actions = episode_data['actions'][:]
            total_frames += len(actions)
            print(f"  - {episode_name}: {len(actions)} 프레임")
        
        print(f"  - 총 프레임 수: {total_frames}")
        
        # 액션/관찰 통계
        if episodes:
            first_episode = f['data'][episodes[0]]
            actions = first_episode['actions'][:]
            observations = first_episode['obs/policy/state'][:]
            
            print(f"\n🎮 액션 통계:")
            print(f"  - 차원: {actions.shape[1]}")
            print(f"  - 범위: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
            print(f"  - 평균: {np.mean(actions):.3f}")
            print(f"  - 표준편차: {np.std(actions):.3f}")
            
            print(f"\n👁️ 관찰 통계:")
            print(f"  - 차원: {observations.shape[1]}")
            print(f"  - 범위: [{np.min(observations):.3f}, {np.max(observations):.3f}]")
            print(f"  - 평균: {np.mean(observations):.3f}")
            print(f"  - 표준편차: {np.std(observations):.3f}")

def load_trained_model():
    """훈련된 모델 로드"""
    model_path = "./datasets/gr00t_imitation_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 훈련된 모델을 찾을 수 없습니다: {model_path}")
        return None
    
    print("🤖 훈련된 모델 로드")
    print("=" * 30)
    
    # 모델 구조 재구성
    class ImitationLearningModel(torch.nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # 모델 로드
    model = ImitationLearningModel(43, 43)  # GR00T 데이터셋 차원
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("✅ 모델이 성공적으로 로드되었습니다!")
    print(f"  - 입력 차원: 43")
    print(f"  - 출력 차원: 43")
    print(f"  - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_model_inference(model):
    """모델 추론 테스트"""
    print("\n🧪 모델 추론 테스트")
    print("=" * 30)
    
    # 랜덤 입력으로 테스트
    test_input = torch.randn(1, 43)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ 추론 성공!")
    print(f"  - 입력: {test_input.shape}")
    print(f"  - 출력: {output.shape}")
    print(f"  - 출력 범위: [{torch.min(output):.3f}, {torch.max(output):.3f}]")

def generate_action_sequence(model, num_steps: int = 100):
    """액션 시퀀스 생성"""
    print(f"\n🎬 {num_steps}단계 액션 시퀀스 생성")
    print("=" * 40)
    
    # 랜덤 관찰 시퀀스 생성
    observations = torch.randn(num_steps, 43)
    actions = []
    
    with torch.no_grad():
        for obs in observations:
            action = model(obs.unsqueeze(0))
            actions.append(action.squeeze(0))
    
    actions = torch.stack(actions)
    
    print(f"✅ 액션 시퀀스 생성 완료!")
    print(f"  - 시퀀스 길이: {len(actions)}")
    print(f"  - 액션 차원: {actions.shape[1]}")
    print(f"  - 액션 범위: [{torch.min(actions):.3f}, {torch.max(actions):.3f}]")
    
    return actions

def visualize_action_sequence(actions):
    """액션 시퀀스 시각화"""
    print("\n📊 액션 시퀀스 시각화")
    print("=" * 30)
    
    actions_np = actions.numpy()
    
    # 첫 10개 액션 차원만 시각화
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(10, actions_np.shape[1])):
        axes[i].plot(actions_np[:, i])
        axes[i].set_title(f'Action Dim {i}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('./datasets/generated_action_sequence.png', dpi=300, bbox_inches='tight')
    print("✅ 시각화 완료: ./datasets/generated_action_sequence.png")

def main():
    """메인 함수"""
    print("🚀 Isaac Sim 없이 계속할 수 있는 작업들")
    print("=" * 50)
    
    # 1. 데이터셋 분석
    analyze_converted_dataset()
    
    # 2. 모델 로드
    model = load_trained_model()
    if model is None:
        return
    
    # 3. 모델 추론 테스트
    test_model_inference(model)
    
    # 4. 액션 시퀀스 생성
    actions = generate_action_sequence(model, 100)
    
    # 5. 시각화
    visualize_action_sequence(actions)
    
    print("\n🎉 모든 작업이 완료되었습니다!")
    print("Isaac Sim 없이도 NVIDIA 데이터셋을 완전히 활용할 수 있습니다.")

if __name__ == "__main__":
    main()
