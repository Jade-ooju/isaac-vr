"""
Isaac Lab Mimic Standalone Demo Runner
Standalone execution script that can run demos even when Isaac Lab environment is not configured
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add project root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class MimicDemoMetadata:
    """Isaac Lab Mimic demo metadata"""
    task_name: str
    demo_id: int
    timestamp: str
    duration: float
    total_steps: int
    success: bool
    vr_system: str
    hand_tracking_confidence: float
    subtask_annotations: Optional[Dict[str, Any]] = None

class StandaloneMimicDemoRecorder:
    """독립 실행 Isaac Lab Mimic 데모 녹화기"""
    
    def __init__(self, config_path: str = None):
        """
        데모 녹화기 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 데모 데이터 저장
        self.demo_data = []
        self.save_path = Path("./mimic_demos/")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Standalone Isaac Lab Mimic demo recorder initialized successfully!")
    
    def _load_config(self, config_path: str = None) -> dict:
        """설정 파일 로드"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "teleop_config.yaml"
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.warning(f"설정 파일 로드 실패: {e}, 기본 설정 사용")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            "mimic": {
                "save_path": "./mimic_demos/",
                "default_duration": 10.0,
                "subtask_annotation": True,
                "success_criteria": {
                    "min_confidence": 0.7,
                    "min_duration": 2.0
                }
            }
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def generate_demo_data(self, task_name: str, num_demos: int = 5, 
                          duration: float = 10.0) -> List[Dict]:
        """
        Generate dummy demo data
        
        Args:
            task_name: Task name
            num_demos: Number of demos to generate
            duration: Length of each demo (seconds)
            
        Returns:
            List of generated demo data
        """
        self.logger.info(f"Starting dummy demo data generation: {task_name}")
        self.logger.info(f"Target number of demos: {num_demos}")
        self.logger.info(f"Demo duration: {duration} seconds")
        
        demo_data_list = []
        
        for demo_idx in range(num_demos):
            self.logger.info(f"\n=== Demo {demo_idx + 1}/{num_demos} ===")
            
            # Generate dummy demo data
            demo_data = self._generate_single_demo(task_name, demo_idx, duration)
            demo_data_list.append(demo_data)
            
            self.logger.info(f"Demo {demo_idx + 1} generation completed!")
        
        self.logger.info(f"All demos generation completed! Total {len(demo_data_list)} demos")
        return demo_data_list
    
    def _generate_single_demo(self, task_name: str, demo_id: int, duration: float) -> Dict:
        """Generate single dummy demo"""
        start_time = time.time()
        step_count = int(duration * 60)  # Assuming 60fps
        
        # Generate dummy data
        observations = np.random.rand(step_count, 20)  # 20-dimensional observation
        actions = np.random.rand(step_count, 24)  # 24-dimensional action (20 joints + 4 additional)
        rewards = np.random.rand(step_count) * 0.1  # Small reward
        dones = np.zeros(step_count, dtype=bool)
        dones[-1] = True  # End at last step
        timestamps = np.linspace(0, duration, step_count)
        hand_data = np.random.rand(step_count, 21, 3)  # 21 joints, 3D coordinates
        joint_positions = np.random.rand(step_count, 20)  # 20 joint positions
        
        # Generate metadata
        metadata = MimicDemoMetadata(
            task_name=task_name,
            demo_id=demo_id,
            timestamp=datetime.now().isoformat(),
            duration=duration,
            total_steps=step_count,
            success=True,
            vr_system="demo",
            hand_tracking_confidence=0.8 + np.random.rand() * 0.2,  # 0.8-1.0
            subtask_annotations=None
        )
        
        # 데모 데이터 구성
        demo_data = {
            "metadata": metadata,
            "data": {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "timestamps": timestamps,
                "hand_data": hand_data,
                "joint_positions": joint_positions
            }
        }
        
        return demo_data
    
    def save_mimic_dataset(self, demo_data_list: List[Dict], task_name: str):
        """Isaac Lab Mimic 형식으로 데이터셋 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = self.save_path / f"{task_name}_mimic_dataset_{timestamp}.h5"
        
        with h5py.File(dataset_path, 'w') as f:
            # 메타데이터 저장
            f.attrs['task_name'] = task_name
            f.attrs['num_demos'] = len(demo_data_list)
            f.attrs['timestamp'] = timestamp
            f.attrs['vr_system'] = "demo"
            
            # 각 데모 저장
            for i, demo in enumerate(demo_data_list):
                demo_group = f.create_group(f'demo_{i}')
                
                # 메타데이터 저장
                metadata = demo["metadata"]
                demo_group.attrs['demo_id'] = metadata.demo_id
                demo_group.attrs['timestamp'] = metadata.timestamp
                demo_group.attrs['duration'] = metadata.duration
                demo_group.attrs['total_steps'] = metadata.total_steps
                demo_group.attrs['success'] = metadata.success
                demo_group.attrs['vr_system'] = metadata.vr_system
                demo_group.attrs['hand_tracking_confidence'] = metadata.hand_tracking_confidence
                
                # 데이터 저장
                data = demo["data"]
                demo_group.create_dataset('observations', data=data['observations'])
                demo_group.create_dataset('actions', data=data['actions'])
                demo_group.create_dataset('rewards', data=data['rewards'])
                demo_group.create_dataset('dones', data=data['dones'])
                demo_group.create_dataset('timestamps', data=data['timestamps'])
                demo_group.create_dataset('hand_data', data=data['hand_data'])
                demo_group.create_dataset('joint_positions', data=data['joint_positions'])
        
        # JSON 메타데이터도 저장
        metadata_path = self.save_path / f"{task_name}_metadata_{timestamp}.json"
        metadata_dict = {
            "task_name": task_name,
            "num_demos": len(demo_data_list),
            "timestamp": timestamp,
            "vr_system": "demo",
            "demos": [asdict(demo["metadata"]) for demo in demo_data_list]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Isaac Lab Mimic 데이터셋 저장 완료: {dataset_path}")
        self.logger.info(f"📄 메타데이터 저장 완료: {metadata_path}")
        
        return str(dataset_path)
    
    def convert_to_robomimic_format(self, mimic_dataset_path: str, output_path: str = None):
        """Isaac Lab Mimic 데이터를 Robomimic 형식으로 변환"""
        if output_path is None:
            output_path = os.path.join(os.path.dirname(mimic_dataset_path), "robomimic_format")
        
        os.makedirs(output_path, exist_ok=True)
        
        self.logger.info(f"🔄 Robomimic 형식으로 변환 중...")
        
        # Isaac Lab Mimic 데이터 로드
        with h5py.File(mimic_dataset_path, 'r') as f:
            num_demos = f.attrs['num_demos']
            
            # Robomimic 형식으로 변환
            for demo_idx in range(num_demos):
                demo_group = f[f'demo_{demo_idx}']
                
                # 관찰 데이터 변환
                observations = demo_group['observations'][:]
                actions = demo_group['actions'][:]
                rewards = demo_group['rewards'][:]
                dones = demo_group['dones'][:]
                
                # Robomimic 형식의 관찰 생성
                robomimic_obs = {
                    "robot0_eef_pos": observations[:, :3],  # 엔드 이펙터 위치
                    "robot0_eef_quat": observations[:, 3:7],  # 엔드 이펙터 쿼터니언
                    "robot0_gripper_qpos": observations[:, 7:9],  # 그리퍼 위치
                    "object": observations[:, 9:12],  # 오브젝트 위치
                }
                
                # Robomimic 형식의 액션 생성
                robomimic_actions = {
                    "robot0_eef_pos": actions[:, :3],
                    "robot0_eef_quat": actions[:, 3:7],
                    "robot0_gripper_qpos": actions[:, 7:9],
                }
                
                # HDF5 파일로 저장
                demo_output_path = os.path.join(output_path, f"demo_{demo_idx}.hdf5")
                with h5py.File(demo_output_path, 'w') as out_f:
                    # 관찰 데이터 저장
                    obs_group = out_f.create_group('observations')
                    for key, value in robomimic_obs.items():
                        obs_group.create_dataset(key, data=value)
                    
                    # 액션 데이터 저장
                    action_group = out_f.create_group('actions')
                    for key, value in robomimic_actions.items():
                        action_group.create_dataset(key, data=value)
                    
                    # 기타 데이터 저장
                    out_f.create_dataset('rewards', data=rewards)
                    out_f.create_dataset('dones', data=dones)
                    out_f.attrs['demo_id'] = demo_idx
        
        # 데이터셋 메타데이터 생성
        metadata = {
            "env": "shadow_hand_manipulation",
            "type": 1,  # HDF5 형식
            "version": "1.0.0",
            "n_demos": num_demos,
            "n_episodes": num_demos,
            "n_steps": 0,  # 실제 스텝 수로 업데이트 필요
            "env_kwargs": {
                "env_name": "shadow_hand_manipulation",
                "render": False,
                "render_camera": "frontview",
                "ignore_done": True,
                "use_image_obs": False,
                "reward_shaping": True,
            },
            "env_meta": {
                "env_name": "shadow_hand_manipulation",
                "type": "Manipulation",
                "env_version": "1.0.0",
                "codebase": "isaac_lab_mimic",
            },
            "data": {
                "action_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "obs_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
                "n_actions": 8,
                "n_obs": 12,
            }
        }
        
        # 메타데이터 저장 (numpy 타입을 Python 타입으로 변환)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        metadata_converted = convert_numpy_types(metadata)
        metadata_path = os.path.join(output_path, "dataset_info.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_converted, f, indent=2)
        
        self.logger.info(f"✅ Robomimic 형식 변환 완료: {output_path}")
        return output_path

def main():
    """메인 실행 함수"""
    import time
    
    parser = argparse.ArgumentParser(
        description="Isaac Lab Mimic 독립 실행 데모",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 더미 데모 데이터 생성
  python run_mimic_standalone.py --action generate --task "Isaac-Repose-Cube-Shadow-Direct-v0" --num-demos 5
  
  # 데이터 변환
  python run_mimic_standalone.py --action convert --dataset "./mimic_demos/dataset.h5"
        """
    )
    
    parser.add_argument("--action", 
                       choices=["generate", "convert"],
                       required=True,
                       help="실행할 작업")
    parser.add_argument("--task", type=str, help="태스크 이름")
    parser.add_argument("--dataset", type=str, help="데이터셋 경로")
    parser.add_argument("--output", type=str, help="출력 경로")
    parser.add_argument("--num-demos", type=int, default=5, help="생성할 데모 수")
    parser.add_argument("--duration", type=float, default=10.0, help="각 데모의 길이 (초)")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    
    args = parser.parse_args()
    
    try:
        # 데모 녹화기 생성
        recorder = StandaloneMimicDemoRecorder(config_path=args.config)
        
        if args.action == "generate":
            # 더미 데모 데이터 생성
            if not args.task:
                raise ValueError("데모 생성 시 --task가 필요합니다")
            
            demo_data_list = recorder.generate_demo_data(
                task_name=args.task,
                num_demos=args.num_demos,
                duration=args.duration
            )
            
            # 데이터셋 저장
            dataset_path = recorder.save_mimic_dataset(demo_data_list, args.task)
            print(f"\n🎉 데모 생성 완료!")
            print(f"📁 데이터셋: {dataset_path}")
            
        elif args.action == "convert":
            # 데이터 변환
            if not args.dataset:
                raise ValueError("데이터 변환 시 --dataset이 필요합니다")
            
            output_path = recorder.convert_to_robomimic_format(args.dataset, args.output)
            print(f"✅ 변환 완료: {output_path}")
        
    except Exception as e:
        logging.error(f"작업 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
