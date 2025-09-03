"""
Isaac Lab Mimic Compatible Demo Recording System
Collects demo data in Isaac Lab Mimic format using VR teleoperation
"""

import os
import time
import json
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Import existing VR system with error handling
try:
    from core.vr_tracker import VRHandTracker
    from core.hand_controller import ShadowHandController
    from core.motion_mapper import MotionMapper
    VR_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VR system not available: {e}")
    print("Running in demo mode without VR integration")
    VR_SYSTEM_AVAILABLE = False
    # Create dummy classes for demo mode
    class VRHandTracker:
        def __init__(self, *args, **kwargs):
            self.vr_system = "demo"
        def get_hand_data(self):
            return np.random.rand(21, 3)  # Dummy hand data
        def update(self):
            pass
        def disconnect(self):
            pass
    
    class ShadowHandController:
        def __init__(self, *args, **kwargs):
            self.is_initialized = False
        def update_hand(self, *args, **kwargs):
            return True
        def step_environment(self):
            pass
        def get_joint_positions(self):
            return np.zeros(20)
        def get_joint_limits(self):
            return {"wrist": [-0.5, 0.5], "thumb": [-0.8, 0.8], "fingers": [-1.57, 1.57]}
        def close(self):
            pass
    
    class MotionMapper:
        def __init__(self, *args, **kwargs):
            pass
        def map_vr_to_shadow_hand(self, hand_data, vr_system):
            return np.random.rand(20)  # Dummy joint positions

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

@dataclass
class SubTaskAnnotation:
    """Subtask annotation"""
    name: str
    start_step: int
    end_step: int
    object_ref: Optional[str] = None
    success: bool = True

class MimicDemoRecorder:
    """Isaac Lab Mimic compatible demo recorder"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize demo recorder
        
        Args:
            config_path: Configuration file path
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # VR system initialization
        self.vr_tracker = None
        self.hand_controller = None
        self.motion_mapper = None
        
        # Recording state
        self.is_recording = False
        self.current_demo = None
        self.demo_data = []
        self.subtask_annotations = []
        
        # Isaac Lab Mimic configuration
        self.mimic_config = self.config.get("mimic", {})
        self.save_path = Path(self.mimic_config.get("save_path", "./mimic_demos/"))
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Component initialization
        self._initialize_components()
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration file"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "teleop_config.yaml"
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}, using default config")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "vr_tracker": {
                "system_type": "auto",
                "update_rate": 60,
                "hand_tracking": {
                    "enable": True,
                    "confidence_threshold": 0.7,
                    "smoothing_factor": 0.3
                }
            },
            "shadow_hand": {
                "environment_name": "Isaac-Repose-Cube-Shadow-Direct-v0",
                "joint_limits": {
                    "wrist": [-0.5, 0.5],
                    "thumb": [-0.8, 0.8],
                    "fingers": [-1.57, 1.57]
                }
            },
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
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """Initialize VR components"""
        try:
            # Initialize VR hand tracker
            vr_config = self.config.get("vr_tracker", {})
            self.vr_tracker = VRHandTracker(
                vr_system=vr_config.get("system_type", "auto"),
                config=vr_config
            )
            
            # Initialize Shadow Hand controller
            shadow_config = self.config.get("shadow_hand", {})
            self.hand_controller = ShadowHandController(
                environment_name=shadow_config.get("environment_name", "Isaac-Repose-Cube-Shadow-Direct-v0")
            )
            
            # Initialize motion mapper
            self.motion_mapper = MotionMapper(config=self.config, is_dummy_mode=not VR_SYSTEM_AVAILABLE)
            
            self.logger.info("Isaac Lab Mimic demo recorder initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def start_demo_collection(self, task_name: str, num_demos: int = 5, 
                            duration: float = None, enable_subtask_annotation: bool = True):
        """
        Start demo collection
        
        Args:
            task_name: Task name
            num_demos: Number of demos to collect
            duration: Length of each demo (seconds)
            enable_subtask_annotation: Enable subtask annotation
        """
        if duration is None:
            duration = self.mimic_config.get("default_duration", 10.0)
        
        self.logger.info(f"Starting Isaac Lab Mimic demo collection")
        self.logger.info(f"Task: {task_name}")
        self.logger.info(f"Target number of demos: {num_demos}")
        self.logger.info(f"Demo duration: {duration} seconds")
        
        for demo_idx in range(num_demos):
            self.logger.info(f"\n=== Demo {demo_idx + 1}/{num_demos} ===")
            
            # Collect single demo
            demo_metadata = self._collect_single_demo(
                task_name=task_name,
                demo_id=demo_idx,
                duration=duration,
                enable_subtask_annotation=enable_subtask_annotation
            )
            
            if demo_metadata.success:
                self.logger.info(f"Demo {demo_idx + 1} collected successfully!")
            else:
                self.logger.warning(f"Demo {demo_idx + 1} collection failed")
        
        # Save all demos
        self._save_mimic_dataset(task_name)
        
        self.logger.info(f"Demo collection completed! Total {len(self.demo_data)} demos")
    
    def _collect_single_demo(self, task_name: str, demo_id: int, 
                           duration: float, enable_subtask_annotation: bool) -> MimicDemoMetadata:
        """Collect single demo"""
        start_time = time.time()
        step_count = 0
        max_steps = int(duration * 60)  # Assuming 60fps
        
        # Initialize demo data
        demo_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "timestamps": [],
            "hand_data": [],
            "joint_positions": []
        }
        
        # Initialize subtask annotation
        current_subtask = None
        subtask_start_step = 0
        
        self.logger.info(f"Starting demo {demo_id + 1} recording...")
        self.logger.info("Please move your hand in VR...")
        self.logger.info("Keyboard shortcuts:")
        self.logger.info("  'S' - Start/End subtask")
        self.logger.info("  'Q' - End demo")
        
        try:
            while step_count < max_steps:
                frame_start = time.time()
                
                # Get VR hand data
                hand_data = self.vr_tracker.get_hand_data()
                
                if hand_data is not None:
                    # Convert VR data to Shadow Hand joint positions
                    joint_positions = self.motion_mapper.map_vr_to_shadow_hand(
                        hand_data, 
                        self.vr_tracker.vr_system
                    )
                    
                    # Create Isaac Lab Mimic format action
                    action = self._create_mimic_action(joint_positions, hand_data)
                    
                    # Generate observation data (from environment)
                    observation = self._get_environment_observation()
                    
                    # Generate reward and info
                    reward, done, info = self._evaluate_step(observation, action, hand_data)
                    
                    # Save data
                    demo_data["observations"].append(observation.copy())
                    demo_data["actions"].append(action.copy())
                    demo_data["rewards"].append(reward)
                    demo_data["dones"].append(done)
                    demo_data["infos"].append(info.copy())
                    demo_data["timestamps"].append(time.time() - start_time)
                    demo_data["hand_data"].append(hand_data.copy())
                    demo_data["joint_positions"].append(joint_positions.copy())
                    
                    # Shadow Hand 업데이트
                    if self.hand_controller and self.hand_controller.is_initialized:
                        self.hand_controller.update_hand(joint_positions)
                        self.hand_controller.step_environment()
                    
                    step_count += 1
                    
                    # Handle subtask annotation
                    if enable_subtask_annotation:
                        self._handle_subtask_annotation(
                            step_count, current_subtask, subtask_start_step
                        )
                    
                    # Print progress
                    if step_count % 60 == 0:  # Every 1 second
                        progress = (time.time() - start_time) / duration * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({step_count}/{max_steps})")
                
                # Update VR tracker
                self.vr_tracker.update()
                
                # Maintain 60fps
                elapsed = time.time() - frame_start
                if elapsed < 1.0/60:
                    time.sleep(1.0/60 - elapsed)
            
            # Demo completed
            actual_duration = time.time() - start_time
            success = self._evaluate_demo_success(demo_data)
            
            # Generate metadata
            metadata = MimicDemoMetadata(
                task_name=task_name,
                demo_id=demo_id,
                timestamp=datetime.now().isoformat(),
                duration=actual_duration,
                total_steps=step_count,
                success=success,
                vr_system=self.vr_tracker.vr_system,
                hand_tracking_confidence=self._calculate_average_confidence(demo_data),
                subtask_annotations=self.subtask_annotations.copy() if enable_subtask_annotation else None
            )
            
            # Save demo data
            self.demo_data.append({
                "metadata": metadata,
                "data": demo_data
            })
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error during demo collection: {e}")
            return MimicDemoMetadata(
                task_name=task_name,
                demo_id=demo_id,
                timestamp=datetime.now().isoformat(),
                duration=time.time() - start_time,
                total_steps=step_count,
                success=False,
                vr_system=self.vr_tracker.vr_system,
                hand_tracking_confidence=0.0
            )
    
    def _create_mimic_action(self, joint_positions: np.ndarray, hand_data: np.ndarray) -> np.ndarray:
        """Create Isaac Lab Mimic format action"""
        # Convert Shadow Hand joint positions to action
        # In actual implementation, adjust according to environment action space
        action = joint_positions.copy()
        
        # Include additional information (gripper state, etc.)
        if len(action) < 24:  # Shadow Hand has 20 joints + additional info
            # Add gripper state
            gripper_state = self._extract_gripper_state(hand_data)
            action = np.concatenate([action, gripper_state])
        
        return action
    
    def _extract_gripper_state(self, hand_data: np.ndarray) -> np.ndarray:
        """Extract gripper state from hand data"""
        # Calculate gripper state based on distance between thumb and index finger
        thumb_tip = hand_data[4]  # Thumb tip
        index_tip = hand_data[8]  # Index finger tip
        
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        # Convert distance to gripper state in 0-1 range
        gripper_state = np.clip(1.0 - distance / 0.1, 0.0, 1.0)  # Based on 10cm
        
        return np.array([gripper_state])
    
    def _get_environment_observation(self) -> np.ndarray:
        """Get observation data from environment"""
        if self.hand_controller and self.hand_controller.is_initialized:
            # Use current state of Shadow Hand as observation
            joint_positions = self.hand_controller.get_joint_positions()
            
            # Additional observation information (wrist position, object position, etc.)
            # In actual implementation, adjust according to environment observation space
            observation = joint_positions.copy()
            
            return observation
        else:
            # Dummy observation data
            return np.zeros(20)
    
    def _evaluate_step(self, observation: np.ndarray, action: np.ndarray, 
                      hand_data: np.ndarray) -> Tuple[float, bool, Dict]:
        """Evaluate step (reward, termination condition, info)"""
        reward = 0.0
        done = False
        info = {}
        
        # Reward based on hand tracking confidence
        confidence = self._calculate_hand_confidence(hand_data)
        reward += confidence * 0.1
        
        # Joint limit violation penalty
        if self._check_joint_limits_violation(action):
            reward -= 0.5
            info["joint_limit_violation"] = True
        
        # Stability reward (joint velocity not too high)
        if hasattr(self, '_prev_action'):
            joint_velocity = np.linalg.norm(action - self._prev_action)
            if joint_velocity < 0.1:  # Stable movement
                reward += 0.05
        
        self._prev_action = action.copy()
        
        return reward, done, info
    
    def _calculate_hand_confidence(self, hand_data: np.ndarray) -> float:
        """Calculate hand tracking confidence"""
        # Simple confidence calculation (actually provided by VR system)
        # Calculate based on consistency of finger tip points
        finger_tips = [hand_data[4], hand_data[8], hand_data[12], hand_data[16], hand_data[20]]
        
        # Check if finger tips are at reasonable distances
        palm_center = np.mean([hand_data[0], hand_data[5], hand_data[9], hand_data[13], hand_data[17]], axis=0)
        
        distances = [np.linalg.norm(tip - palm_center) for tip in finger_tips]
        avg_distance = np.mean(distances)
        
        # High confidence if distance is in reasonable range (5-15cm)
        if 0.05 <= avg_distance <= 0.15:
            confidence = 1.0
        else:
            confidence = max(0.0, 1.0 - abs(avg_distance - 0.1) / 0.1)
        
        return confidence
    
    def _check_joint_limits_violation(self, action: np.ndarray) -> bool:
        """Check joint limit violations"""
        joint_limits = self.hand_controller.get_joint_limits()
        
        # Check wrist joints
        wrist_limits = joint_limits["wrist"]
        if np.any(action[17:19] < wrist_limits[0]) or np.any(action[17:19] > wrist_limits[1]):
            return True
        
        # Check thumb joints
        thumb_limits = joint_limits["thumb"]
        if np.any(action[0:5] < thumb_limits[0]) or np.any(action[0:5] > thumb_limits[1]):
            return True
        
        # Check finger joints
        finger_limits = joint_limits["fingers"]
        if np.any(action[5:17] < finger_limits[0]) or np.any(action[5:17] > finger_limits[1]):
            return True
        
        return False
    
    def _evaluate_demo_success(self, demo_data: Dict) -> bool:
        """Evaluate demo success"""
        success_criteria = self.mimic_config.get("success_criteria", {})
        
        # Check minimum confidence
        min_confidence = success_criteria.get("min_confidence", 0.7)
        avg_confidence = self._calculate_average_confidence(demo_data)
        if avg_confidence < min_confidence:
            return False
        
        # Check minimum duration
        min_duration = success_criteria.get("min_duration", 2.0)
        if demo_data["timestamps"][-1] < min_duration:
            return False
        
        # Check joint limit violations
        for action in demo_data["actions"]:
            if self._check_joint_limits_violation(action):
                return False
        
        return True
    
    def _calculate_average_confidence(self, demo_data: Dict) -> float:
        """Calculate average confidence of demo"""
        confidences = []
        for hand_data in demo_data["hand_data"]:
            confidence = self._calculate_hand_confidence(hand_data)
            confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _handle_subtask_annotation(self, step_count: int, current_subtask: Optional[str], 
                                 subtask_start_step: int):
        """Handle subtask annotation"""
        # Check keyboard input (in actual implementation, handle keyboard events)
        # Currently dummy implementation
        pass
    
    def _save_mimic_dataset(self, task_name: str):
        """Save dataset in Isaac Lab Mimic format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = self.save_path / f"{task_name}_mimic_dataset_{timestamp}.h5"
        
        with h5py.File(dataset_path, 'w') as f:
            # Save metadata
            f.attrs['task_name'] = task_name
            f.attrs['num_demos'] = len(self.demo_data)
            f.attrs['timestamp'] = timestamp
            f.attrs['vr_system'] = self.vr_tracker.vr_system
            
            # Save each demo
            for i, demo in enumerate(self.demo_data):
                demo_group = f.create_group(f'demo_{i}')
                
                # Save metadata
                metadata = demo["metadata"]
                demo_group.attrs['demo_id'] = metadata.demo_id
                demo_group.attrs['timestamp'] = metadata.timestamp
                demo_group.attrs['duration'] = metadata.duration
                demo_group.attrs['total_steps'] = metadata.total_steps
                demo_group.attrs['success'] = metadata.success
                demo_group.attrs['vr_system'] = metadata.vr_system
                demo_group.attrs['hand_tracking_confidence'] = metadata.hand_tracking_confidence
                
                # Save data
                data = demo["data"]
                demo_group.create_dataset('observations', data=np.array(data['observations']))
                demo_group.create_dataset('actions', data=np.array(data['actions']))
                demo_group.create_dataset('rewards', data=np.array(data['rewards']))
                demo_group.create_dataset('dones', data=np.array(data['dones']))
                demo_group.create_dataset('timestamps', data=np.array(data['timestamps']))
                demo_group.create_dataset('hand_data', data=np.array(data['hand_data']))
                demo_group.create_dataset('joint_positions', data=np.array(data['joint_positions']))
                
                # Save subtask annotation
                if metadata.subtask_annotations:
                    subtask_group = demo_group.create_group('subtasks')
                    for j, subtask in enumerate(metadata.subtask_annotations):
                        subtask_group.attrs[f'subtask_{j}_name'] = subtask.name
                        subtask_group.attrs[f'subtask_{j}_start'] = subtask.start_step
                        subtask_group.attrs[f'subtask_{j}_end'] = subtask.end_step
                        if subtask.object_ref:
                            subtask_group.attrs[f'subtask_{j}_object_ref'] = subtask.object_ref
        
        # Also save JSON metadata
        metadata_path = self.save_path / f"{task_name}_metadata_{timestamp}.json"
        metadata_dict = {
            "task_name": task_name,
            "num_demos": len(self.demo_data),
            "timestamp": timestamp,
            "vr_system": self.vr_tracker.vr_system,
            "demos": [asdict(demo["metadata"]) for demo in self.demo_data]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Isaac Lab Mimic dataset saved successfully: {dataset_path}")
        self.logger.info(f"Metadata saved successfully: {metadata_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.vr_tracker:
                self.vr_tracker.disconnect()
            
            if self.hand_controller:
                self.hand_controller.close()
            
            self.logger.info("Isaac Lab Mimic demo recorder cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Isaac Lab Mimic demo collection")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--num-demos", type=int, default=5, help="Number of demos to collect")
    parser.add_argument("--duration", type=float, default=10.0, help="Length of each demo (seconds)")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--no-subtask", action="store_true", help="Disable subtask annotation")
    
    args = parser.parse_args()
    
    try:
        # Create demo recorder
        recorder = MimicDemoRecorder(config_path=args.config)
        
        # Start demo collection
        recorder.start_demo_collection(
            task_name=args.task,
            num_demos=args.num_demos,
            duration=args.duration,
            enable_subtask_annotation=not args.no_subtask
        )
        
    except Exception as e:
        logging.error(f"Demo collection failed: {e}")
        return 1
    
    finally:
        recorder.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())
