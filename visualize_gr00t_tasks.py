#!/usr/bin/env python3
"""
Script to visualize PhysicalAI-GR00T-Tuned-Tasks dataset in Isaac Lab
"""

import argparse
import h5py
import numpy as np
import torch
from datasets import load_dataset
from pathlib import Path
import os

def convert_gr00t_to_hdf5(dataset_name="nvidia/PhysicalAI-GR00T-Tuned-Tasks", 
                          output_file="datasets/gr00t_tuned_tasks.hdf5",
                          max_episodes=10):
    """Convert GR00T Tuned Tasks dataset to Isaac Lab HDF5 format"""
    
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)
    
    print(f"Dataset loaded: {dataset['train'].num_rows:,} samples")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Save metadata
        f.attrs['env_name'] = 'Isaac-GR00T-Tuned-Tasks-v0'
        f.attrs['num_episodes'] = min(max_episodes, len(set(dataset['train']['episode_index'])))
        
        # Create data group
        data_group = f.create_group('data')
        data_group.attrs['env_args'] = {
            'task': 'Isaac-GR00T-Tuned-Tasks-v0',
            'num_envs': 1,
            'device': 'cuda:0'
        }
        
        # Store data by episode
        episode_data = {}
        for i, (obs, action, episode_idx, task_idx, task_desc) in enumerate(zip(
            dataset['train']['observation.state'],
            dataset['train']['action'],
            dataset['train']['episode_index'],
            dataset['train']['task_index'],
            dataset['train']['annotation.human.action.task_description']
        )):
            if episode_idx not in episode_data:
                episode_data[episode_idx] = {
                    'observations': [],
                    'actions': [],
                    'task_index': task_idx,
                    'task_description': task_desc
                }
            
            episode_data[episode_idx]['observations'].append(obs)
            episode_data[episode_idx]['actions'].append(action)
        
        # Save to HDF5
        episode_count = 0
        for episode_idx in sorted(episode_data.keys())[:max_episodes]:
            demo_group = data_group.create_group(f'demo_{episode_count}')
            
            # Convert data to numpy arrays
            obs_array = np.array(episode_data[episode_idx]['observations'])
            action_array = np.array(episode_data[episode_idx]['actions'])
            
            # Create datasets
            demo_group.create_dataset('obs/state', data=obs_array, compression='gzip')
            demo_group.create_dataset('actions', data=action_array, compression='gzip')
            
            # Save metadata
            demo_group.attrs['num_samples'] = len(obs_array)
            demo_group.attrs['task_index'] = episode_data[episode_idx]['task_index']
            demo_group.attrs['task_description'] = episode_data[episode_idx]['task_description']
            
            episode_count += 1
            print(f"Episode {episode_count}/{min(max_episodes, len(episode_data))} saved")
    
    print(f"Conversion completed: {output_file}")
    return output_file

def create_visualization_script():
    """Create script to visualize data in Isaac Lab"""
    
    script_content = '''#!/usr/bin/env python3
"""
Visualize GR00T Tuned Tasks data in Isaac Lab
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import seaborn as sns

class GR00TTasksVisualizer:
    """GR00T Tuned Tasks data visualization class"""
    
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """Load HDF5 data"""
        print("Loading data...")
        
        with h5py.File(self.hdf5_file, 'r') as f:
            for demo_name in f['data'].keys():
                demo_data = f['data'][demo_name]
                
                self.data[demo_name] = {
                    'observations': demo_data['obs/state'][:],
                    'actions': demo_data['actions'][:],
                    'task_index': demo_data.attrs.get('task_index', 0),
                    'task_description': demo_data.attrs.get('task_description', 'Unknown')
                }
        
        print(f"{len(self.data)} demos loaded")
    
    def plot_trajectory_2d(self, demo_name, save_path=None):
        """2D trajectory visualization"""
        if demo_name not in self.data:
            print(f"Demo {demo_name} not found")
            return
        
        data = self.data[demo_name]
        obs = data['observations']
        actions = data['actions']
        
        # Extract joint positions (assuming first 7 joints are arm joints)
        joint_positions = obs[:, :7]  # First 7 joints
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'GR00T Tuned Tasks - {demo_name}\\nTask: {data["task_description"]}', fontsize=16)
        
        # Joint position time series
        axes[0, 0].plot(joint_positions)
        axes[0, 0].set_title('Joint Positions Over Time')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Joint Position (rad)')
        axes[0, 0].legend([f'Joint {i}' for i in range(7)])
        axes[0, 0].grid(True)
        
        # Action time series
        axes[0, 1].plot(actions)
        axes[0, 1].set_title('Actions Over Time')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Action Value')
        axes[0, 1].legend([f'Action {i}' for i in range(actions.shape[1])])
        axes[0, 1].grid(True)
        
        # Joint position distribution
        axes[1, 0].hist(joint_positions.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Joint Position Distribution')
        axes[1, 0].set_xlabel('Joint Position (rad)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Action distribution
        axes[1, 1].hist(actions.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].set_xlabel('Action Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved: {save_path}")
        
        plt.show()
    
    def create_animation(self, demo_name, save_path=None):
        """Create animation"""
        if demo_name not in self.data:
            print(f"Demo {demo_name} not found")
            return
        
        data = self.data[demo_name]
        obs = data['observations']
        
        # Extract joint positions
        joint_positions = obs[:, :7]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Robot arm visualization (simple 2D representation)
        def animate(frame):
            ax.clear()
            
            # Convert joint positions to 2D coordinates (assuming simple link lengths)
            link_lengths = [0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]
            x, y = 0, 0
            angles = np.cumsum(joint_positions[frame])
            
            for i, (angle, length) in enumerate(zip(angles, link_lengths)):
                next_x = x + length * np.cos(angle)
                next_y = y + length * np.sin(angle)
                
                ax.plot([x, next_x], [y, next_y], 'b-', linewidth=3)
                ax.plot(x, y, 'ro', markersize=8)
                
                x, y = next_x, next_y
            
            ax.plot(x, y, 'go', markersize=10)  # End effector
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.set_title(f'GR00T Robot Arm - Frame {frame}/{len(joint_positions)-1}')
            ax.grid(True)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(joint_positions), 
                                     interval=50, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, save_dir="visualization_report"):
        """Generate comprehensive report"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive report...")
        
        # Statistics for all demos
        all_obs = []
        all_actions = []
        task_counts = {}
        
        for demo_name, data in self.data.items():
            all_obs.append(data['observations'])
            all_actions.append(data['actions'])
            
            task_desc = data['task_description']
            task_counts[task_desc] = task_counts.get(task_desc, 0) + 1
        
        all_obs = np.concatenate(all_obs)
        all_actions = np.concatenate(all_actions)
        
        # Statistical summary
        stats = {
            'total_demos': len(self.data),
            'total_samples': len(all_obs),
            'avg_episode_length': len(all_obs) / len(self.data),
            'task_distribution': task_counts,
            'obs_mean': np.mean(all_obs, axis=0).tolist(),
            'obs_std': np.std(all_obs, axis=0).tolist(),
            'action_mean': np.mean(all_actions, axis=0).tolist(),
            'action_std': np.std(all_actions, axis=0).tolist()
        }
        
        # Save as JSON
        import json
        with open(f"{save_dir}/statistics_summary.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Report saved: {save_dir}/")
        return stats

def main():
    parser = argparse.ArgumentParser(description="GR00T Tuned Tasks data visualization")
    parser.add_argument("--hdf5_file", type=str, default="datasets/gr00t_tuned_tasks.hdf5",
                       help="HDF5 file path")
    parser.add_argument("--demo_name", type=str, default="demo_0",
                       help="Demo name to visualize")
    parser.add_argument("--save_dir", type=str, default="visualization_report",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_file):
        print(f"HDF5 file not found: {args.hdf5_file}")
        print("Please run convert_gr00t_to_hdf5() first to convert the data.")
        return
    
    # Create visualization tool
    visualizer = GR00TTasksVisualizer(args.hdf5_file)
    
    # Visualize individual demo
    visualizer.plot_trajectory_2d(args.demo_name, f"{args.save_dir}/{args.demo_name}_trajectory.png")
    visualizer.create_animation(args.demo_name, f"{args.save_dir}/{args.demo_name}_animation.gif")
    
    # Generate comprehensive report
    visualizer.generate_summary_report(args.save_dir)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()
'''
    
    with open("visualize_gr00t_tasks.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("Visualization script created: visualize_gr00t_tasks.py")

def main():
    parser = argparse.ArgumentParser(description="GR00T Tuned Tasks dataset Isaac Lab conversion")
    parser.add_argument("--dataset_name", type=str, default="nvidia/PhysicalAI-GR00T-Tuned-Tasks",
                       help="Hugging Face dataset name")
    parser.add_argument("--output_file", type=str, default="datasets/gr00t_tuned_tasks.hdf5",
                       help="Output HDF5 file path")
    parser.add_argument("--max_episodes", type=int, default=10,
                       help="Maximum number of episodes")
    parser.add_argument("--create_visualizer", action="store_true",
                       help="Also create visualization script")
    
    args = parser.parse_args()
    
    print("Starting GR00T Tuned Tasks dataset Isaac Lab conversion")
    print("=" * 60)
    
    # Convert dataset
    hdf5_file = convert_gr00t_to_hdf5(
        dataset_name=args.dataset_name,
        output_file=args.output_file,
        max_episodes=args.max_episodes
    )
    
    # Create visualization script
    if args.create_visualizer:
        create_visualization_script()
    
    print("\nConversion completed!")
    print(f"HDF5 file: {hdf5_file}")
    print(f"Visualization methods:")
    print(f"   1. python visualize_gr00t_tasks.py --hdf5_file {hdf5_file}")
    print(f"   2. Replay in Isaac Lab: ./isaaclab.sh -p scripts/tools/replay_demos.py --dataset_file {hdf5_file}")

if __name__ == "__main__":
    main()
