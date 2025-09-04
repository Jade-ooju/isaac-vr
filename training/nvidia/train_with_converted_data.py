#!/usr/bin/env python3
"""
변환된 NVIDIA 데이터셋으로 모방학습 훈련을 수행하는 스크립트
IsaacLab 시뮬레이터 없이도 데이터를 활용할 수 있습니다.
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


class GR00TDataset(Dataset):
    """GR00T 데이터셋을 PyTorch Dataset으로 변환"""
    
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.episodes = []
        self.load_data()
    
    def load_data(self):
        """HDF5 파일에서 데이터를 로드합니다."""
        with h5py.File(self.hdf5_path, 'r') as f:
            episode_count = 0
            for episode_name in f['data'].keys():
                episode_data = f['data'][episode_name]
                
                # 액션과 관찰 데이터 추출
                actions = torch.tensor(episode_data['actions'][:], dtype=torch.float32)
                observations = torch.tensor(episode_data['obs/policy/state'][:], dtype=torch.float32)
                
                # 시퀀스 데이터로 저장
                for i in range(len(actions)):
                    self.episodes.append({
                        'observation': observations[i],
                        'action': actions[i]
                    })
                
                episode_count += 1
        
        print(f"Loaded {len(self.episodes)} training samples from {episode_count} episodes")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        return self.episodes[idx]


class ImitationLearningModel(nn.Module):
    """간단한 모방학습 모델"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(dataset_path: str, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
    """모방학습 모델을 훈련합니다."""
    
    # 데이터셋 로드
    dataset = GR00TDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 생성
    obs_dim = dataset.episodes[0]['observation'].shape[0]
    action_dim = dataset.episodes[0]['action'].shape[0]
    model = ImitationLearningModel(obs_dim, action_dim)
    
    # 옵티마이저와 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 훈련 루프
    losses = []
    model.train()
    
    print(f"Training model with {len(dataset)} samples...")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            observations = batch['observation']
            actions = batch['action']
            
            # 순전파
            predicted_actions = model(observations)
            loss = criterion(predicted_actions, actions)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses


def evaluate_model(model: nn.Module, dataset_path: str, num_samples: int = 100):
    """모델을 평가합니다."""
    dataset = GR00TDataset(dataset_path)
    model.eval()
    
    # 랜덤 샘플 선택
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    total_error = 0.0
    errors = []
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            observation = sample['observation'].unsqueeze(0)
            true_action = sample['action']
            
            predicted_action = model(observation).squeeze(0)
            error = torch.norm(predicted_action - true_action).item()
            errors.append(error)
            total_error += error
    
    avg_error = total_error / len(errors)
    print(f"Average prediction error: {avg_error:.6f}")
    print(f"Error std: {np.std(errors):.6f}")
    
    return errors


def plot_training_results(losses: List[float], errors: List[float], save_path: str = None):
    """훈련 결과를 시각화합니다."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 훈련 손실
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)
    
    # 예측 오차 분포
    ax2.hist(errors, bins=30, alpha=0.7)
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('L2 Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results saved to: {save_path}")
    
    plt.show()


def main():
    """메인 함수"""
    dataset_path = "./datasets/gr00t_teleop_test.hdf5"
    
    print("🤖 Training Imitation Learning Model with GR00T Data")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # 모델 훈련
    model, losses = train_model(dataset_path, epochs=50, batch_size=64)
    
    # 모델 평가
    errors = evaluate_model(model, dataset_path)
    
    # 결과 시각화
    plot_training_results(losses, errors, "./datasets/training_results.png")
    
    # 모델 저장
    model_path = "./datasets/gr00t_imitation_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    print(f"\n🎉 Training completed!")
    print(f"   - Final loss: {losses[-1]:.6f}")
    print(f"   - Average prediction error: {np.mean(errors):.6f}")
    print(f"   - Model ready for deployment")


if __name__ == "__main__":
    main()
