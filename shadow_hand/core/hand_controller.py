"""
Shadow Hand Control Class
Directly control Shadow Hand in Isaac Sim environment
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
import logging

# Isaac Lab imports with error handling
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import update_dict
    ISAAC_LAB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Isaac Lab not available: {e}")
    print("Running in standalone mode without Isaac Lab integration")
    ISAAC_LAB_AVAILABLE = False
    # Create dummy classes for standalone mode
    class ManagerBasedRLEnv:
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            return None, {}
        def step(self, action):
            return None, 0, False, {}
        def close(self):
            pass

class ShadowHandController:
    """Shadow Hand control class"""
    
    def __init__(self, environment_name: str = "Isaac-Repose-Cube-Shadow-Direct-v0"):
        """
        Initialize Shadow Hand controller
        
        Args:
            environment_name: IsaacLab environment name
        """
        self.environment_name = environment_name
        self.env = None
        self.robot = None
        self.is_initialized = False
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Joint limit ranges (radians)
        self.joint_limits = {
            "wrist": [-0.5, 0.5],      # WRJ0, WRJ1
            "thumb": [-0.8, 0.8],      # THJ0-4
            "fingers": [-1.57, 1.57]   # FFJ, MFJ, RFJ, LFJ
        }
        
        # Current joint state
        self.current_joint_positions = np.zeros(20)
        self.target_joint_positions = np.zeros(20)
        
        # Control settings
        self.control_config = {
            "enable_ik": True,
            "damping_factor": 0.1,
            "stiffness_factor": 1.0,
            "max_velocity": 2.0,  # rad/s
            "position_tolerance": 0.01  # rad
        }
        
        # Environment initialization
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize IsaacLab environment"""
        try:
            self.logger.info(f"Initializing environment: {self.environment_name}")
            
            if ISAAC_LAB_AVAILABLE:
                # Use Isaac Lab's ManagerBasedRLEnv
                try:
                    # Import Isaac Lab environment configuration
                    import isaaclab_tasks
                    self.logger.info("IsaacLab environment registration completed!")
                    
                    # Create environment using Isaac Lab's method
                    if "Isaac-Repose-Cube-Shadow-Direct-v0" in self.environment_name:
                        # Shadow Hand environment
                        from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
                        cfg = ShadowHandEnvCfg()
                        self.env = ManagerBasedRLEnv(cfg=cfg)
                        self.logger.info("Isaac Lab Shadow Hand environment created successfully!")
                    else:
                        # Other Isaac Lab environments
                        self.env = ManagerBasedRLEnv(cfg=None)
                        self.logger.info("Isaac Lab environment created successfully!")
                    
                    # Reset environment to load into Isaac Sim window
                    obs, info = self.env.reset()
                    self.logger.info("Environment loaded into Isaac Sim window!")
                    
                    # Find Shadow Hand object
                    self.robot = self._find_shadow_hand()
                    
                    if self.robot is not None:
                        self.is_initialized = True
                        self.logger.info("Shadow Hand object found successfully!")
                        
                        # Get initial joint state
                        self._update_current_state()
                        
                    else:
                        self.logger.error("Shadow Hand object not found")
                        self.is_initialized = False
                        
                except Exception as e:
                    self.logger.error(f"Isaac Lab environment creation failed: {e}")
                    self.is_initialized = False
            else:
                # Fallback to standalone mode
                self.logger.warning("Isaac Lab not available, running in standalone mode")
                self.is_initialized = False
                
        except Exception as e:
            self.logger.error(f"Environment initialization failed: {e}")
            self.is_initialized = False
    
    def _find_shadow_hand(self):
        """Find Shadow Hand object in environment"""
        if self.env is None:
            return None
        
        # Try multiple methods to find Shadow Hand object
        robot_candidates = []
        
        # 1. Check Isaac Lab ManagerBasedRLEnv structure
        if hasattr(self.env, 'scene'):
            if hasattr(self.env.scene, 'robot'):
                robot_candidates.append(('env.scene.robot', self.env.scene.robot))
        
        # 2. Check direct robot attribute
        if hasattr(self.env, 'robot'):
            robot_candidates.append(('env.robot', self.env.robot))
        
        # 3. Check _env attribute
        if hasattr(self.env, '_env'):
            if hasattr(self.env._env, 'robot'):
                robot_candidates.append(('env._env.robot', self.env._env.robot))
        
        # 4. Check Isaac Lab specific attributes
        if hasattr(self.env, 'managers'):
            if hasattr(self.env.managers, 'robot'):
                robot_candidates.append(('env.managers.robot', self.env.managers.robot))
        
        # 5. Print environment structure (for debugging)
        if not robot_candidates:
            self.logger.info("Environment structure analysis:")
            self._print_environment_structure(self.env)
        
        # Return first candidate
        if robot_candidates:
            name, robot = robot_candidates[0]
            self.logger.info(f"Shadow Hand object found: {name}")
            return robot
        
        return None
    
    def _print_environment_structure(self, env, max_depth=3, current_depth=0):
        """Print environment structure (for debugging)"""
        if current_depth >= max_depth:
            return
        
        indent = "  " * current_depth
        for attr in dir(env):
            if not attr.startswith('_'):
                try:
                    value = getattr(env, attr)
                    if hasattr(value, '__class__'):
                        self.logger.info(f"{indent}{attr}: {value.__class__.__name__}")
                    else:
                        self.logger.info(f"{indent}{attr}: {type(value)}")
                except:
                    pass
    
    def _update_current_state(self):
        """Update current joint state"""
        if self.robot is None:
            return
        
        try:
            # Get current joint positions
            if hasattr(self.robot, 'data') and hasattr(self.robot.data, 'joint_pos'):
                self.current_joint_positions = self.robot.data.joint_pos.clone().cpu().numpy()
            elif hasattr(self.robot, 'get_joint_positions'):
                self.current_joint_positions = self.robot.get_joint_positions()
            else:
                self.logger.warning("Cannot get joint positions")
                
        except Exception as e:
            self.logger.error(f"Current state update failed: {e}")
    
    def update_hand(self, hand_data: np.ndarray, smoothing: bool = True) -> bool:
        """
        Update Shadow Hand with VR hand data
        
        Args:
            hand_data: VR hand data (21 joints, 3D coordinates)
            smoothing: Whether to apply smoothing
            
        Returns:
            Update success status
        """
        if not self.is_initialized:
            self.logger.error("Controller not initialized")
            return False
        
        try:
            # Convert VR data to Shadow Hand joint positions
            joint_positions = self._convert_hand_data_to_joints(hand_data)
            
            # Apply joint limits
            joint_positions = self._apply_joint_limits(joint_positions)
            
            # Apply smoothing
            if smoothing:
                joint_positions = self._apply_smoothing(joint_positions)
            
            # Update Shadow Hand
            success = self._set_joint_positions(joint_positions)
            
            if success:
                self.target_joint_positions = joint_positions.copy()
                self.logger.debug(f"Hand update successful: {joint_positions[:5]}...")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Hand update failed: {e}")
            return False
    
    def _convert_hand_data_to_joints(self, hand_data: np.ndarray) -> np.ndarray:
        """
        Convert VR hand data to Shadow Hand joint positions
        
        Args:
            hand_data: VR hand data (21 joints, 3D coordinates)
            
        Returns:
            Shadow Hand joint positions (20 joints)
        """
        # Simple conversion: Convert 3D positions to joint angles
        # More sophisticated conversion logic needed in actual implementation
        
        joint_positions = np.zeros(20)
        
        # Wrist joints (WRJ0, WRJ1)
        joint_positions[17:19] = hand_data[0:2, 0] * 0.1  # X coordinates to angles
        
        # Thumb joints (THJ0-4)
        joint_positions[0:5] = hand_data[1:6, 1] * 0.1  # Y coordinates to angles
        
        # Index finger joints (FFJ1-3)
        joint_positions[5:8] = hand_data[5:8, 2] * 0.1  # Z coordinates to angles
        
        # Middle finger joints (MFJ1-3)
        joint_positions[8:11] = hand_data[9:12, 1] * 0.1
        
        # Ring finger joints (RFJ1-3)
        joint_positions[11:14] = hand_data[13:16, 2] * 0.1
        
        # Little finger joints (LFJ1-3)
        joint_positions[14:17] = hand_data[17:20, 0] * 0.1
        
        return joint_positions
    
    def _apply_joint_limits(self, joint_positions: np.ndarray) -> np.ndarray:
        """Apply joint limit ranges"""
        limited_positions = joint_positions.copy()
        
        # Wrist joint limits
        limited_positions[17:19] = np.clip(
            limited_positions[17:19], 
            self.joint_limits["wrist"][0], 
            self.joint_limits["wrist"][1]
        )
        
        # Thumb joint limits
        limited_positions[0:5] = np.clip(
            limited_positions[0:5], 
            self.joint_limits["thumb"][0], 
            self.joint_limits["thumb"][1]
        )
        
        # Finger joint limits
        limited_positions[5:17] = np.clip(
            limited_positions[5:17], 
            self.joint_limits["fingers"][0], 
            self.joint_limits["fingers"][1]
        )
        
        return limited_positions
    
    def _apply_smoothing(self, target_positions: np.ndarray) -> np.ndarray:
        """Apply smoothing to joint positions"""
        smoothing_factor = 0.3
        
        smoothed_positions = (
            self.current_joint_positions * (1 - smoothing_factor) + 
            target_positions * smoothing_factor
        )
        
        return smoothed_positions
    
    def _set_joint_positions(self, joint_positions: np.ndarray) -> bool:
        """Set Shadow Hand joint positions"""
        if self.robot is None:
            return False
        
        try:
            # Set joint positions using Isaac Lab methods
            if hasattr(self.robot, 'set_joint_position_target'):
                self.robot.set_joint_position_target(joint_positions)
            elif hasattr(self.robot, 'set_joint_positions'):
                self.robot.set_joint_positions(joint_positions)
            elif hasattr(self.robot, 'write_joint_positions'):
                self.robot.write_joint_positions(joint_positions)
            else:
                self.logger.warning("Joint position setting method not found")
                return False
            
            # Write data to simulation
            if hasattr(self.robot, 'write_data_to_sim'):
                self.robot.write_data_to_sim()
            elif hasattr(self.robot, 'write'):
                self.robot.write()
            
            # Update current state
            self.current_joint_positions = joint_positions.copy()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Joint position setting failed: {e}")
            return False
    
    def get_joint_positions(self) -> np.ndarray:
        """Return current joint positions"""
        return self.current_joint_positions.copy()
    
    def get_joint_limits(self) -> Dict[str, List[float]]:
        """Return joint limit ranges"""
        return self.joint_limits.copy()
    
    def reset_hand(self):
        """Reset Shadow Hand to initial position"""
        if not self.is_initialized:
            return False
        
        try:
            # Initial joint positions (all joints at 0 degrees)
            initial_positions = np.zeros(20)
            
            # Reset
            success = self._set_joint_positions(initial_positions)
            
            if success:
                self.logger.info("Shadow Hand reset completed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Shadow Hand reset failed: {e}")
            return False
    
    def step_environment(self):
        """Execute environment step"""
        if self.env is None:
            return False
        
        try:
            # Execute environment step using Isaac Lab methods
            if hasattr(self.env, 'step'):
                # For Isaac Lab ManagerBasedRLEnv, we need to provide an action
                # Create a dummy action (zeros) for the step
                dummy_action = np.zeros(20)  # 20 joint actions for Shadow Hand
                obs, reward, terminated, truncated, info = self.env.step(dummy_action)
                return True
            else:
                self.logger.warning("Environment step method not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Environment step execution failed: {e}")
            return False
    
    def close(self):
        """Cleanup controller"""
        try:
            if self.env:
                self.env.close()
                self.env = None
            
            self.robot = None
            self.is_initialized = False
            
            self.logger.info("Shadow Hand controller cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Controller cleanup failed: {e}")
    
    def __del__(self):
        """Destructor"""
        self.close()
