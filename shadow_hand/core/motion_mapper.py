"""
Motion Retargeting Class
Convert VR hand data to Shadow Hand joint positions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class MotionMapper:
    """Motion retargeting class"""
    
    def __init__(self, config: Optional[Dict] = None, is_dummy_mode: bool = False):
        """
        Initialize motion mapper
        
        Args:
            config: Configuration dictionary
            is_dummy_mode: Whether running in dummy mode (suppress warnings)
        """
        self.config = config or {}
        self.is_dummy_mode = is_dummy_mode
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # VR system joint mappings
        self.joint_mappings = self._initialize_joint_mappings()
        
        # Warning tracking
        self._unsupported_warning_logged = False
        
        # Conversion parameters
        self.scale_factors = {
            "wrist": 0.1,      # Wrist joint scale
            "thumb": 0.1,      # Thumb joint scale
            "fingers": 0.1     # Finger joint scale
        }
        
        # Smoothing settings
        self.smoothing_config = {
            "enabled": True,
            "factor": 0.3,
            "buffer_size": 5
        }
        
        # Smoothing buffer
        self.smoothing_buffer = []
    
    def _initialize_joint_mappings(self) -> Dict:
        """Initialize joint mappings"""
        return {
            "steamvr": {
                "wrist": {
                    "vr_indices": [0, 1],      # VR wrist joints
                    "shadow_indices": [17, 18]  # Shadow Hand WRJ0, WRJ1
                },
                "thumb": {
                    "vr_indices": [2, 3, 4, 5, 6],  # VR thumb joints
                    "shadow_indices": [0, 1, 2, 3, 4]  # Shadow Hand THJ0-4
                },
                "index": {
                    "vr_indices": [7, 8, 9, 10],  # VR index finger joints
                    "shadow_indices": [5, 6, 7]  # Shadow Hand FFJ1-3
                },
                "middle": {
                    "vr_indices": [11, 12, 13, 14],  # VR middle finger joints
                    "shadow_indices": [8, 9, 10]  # Shadow Hand MFJ1-3
                },
                "ring": {
                    "vr_indices": [15, 16, 17, 18],  # VR ring finger joints
                    "shadow_indices": [11, 12, 13]  # Shadow Hand RFJ1-3
                },
                "little": {
                    "vr_indices": [19, 20],  # VR little finger joints
                    "shadow_indices": [14, 15, 16]  # Shadow Hand LFJ1-3
                }
            },
            "quest3": {
                # Quest 3 uses same mapping as SteamVR
                "wrist": {
                    "vr_indices": [0, 1],
                    "shadow_indices": [17, 18]
                },
                "thumb": {
                    "vr_indices": [2, 3, 4, 5, 6],
                    "shadow_indices": [0, 1, 2, 3, 4]
                },
                "index": {
                    "vr_indices": [7, 8, 9, 10],
                    "shadow_indices": [5, 6, 7]
                },
                "middle": {
                    "vr_indices": [11, 12, 13, 14],
                    "shadow_indices": [8, 9, 10]
                },
                "ring": {
                    "vr_indices": [15, 16, 17, 18],
                    "shadow_indices": [11, 12, 13]
                },
                "little": {
                    "vr_indices": [19, 20],
                    "shadow_indices": [14, 15, 16]
                }
            }
        }
    
    def map_vr_to_shadow_hand(self, 
                              vr_data: np.ndarray, 
                              vr_system: str = "steamvr") -> np.ndarray:
        """
        Map VR data to Shadow Hand joint positions
        
        Args:
            vr_data: VR hand data (21 joints, 3D coordinates)
            vr_system: VR system type
            
        Returns:
            Shadow Hand joint positions (20 joints)
        """
        if vr_system not in self.joint_mappings:
            # Handle "auto" case by defaulting to steamvr
            if vr_system == "auto":
                vr_system = "steamvr"
            else:
                # Only log warning once and not in dummy mode
                if not self.is_dummy_mode and not self._unsupported_warning_logged:
                    self.logger.warning(f"Unsupported VR system: {vr_system}, using SteamVR")
                    self._unsupported_warning_logged = True
                vr_system = "steamvr"
        
        mapping = self.joint_mappings[vr_system]
        joint_positions = np.zeros(20)
        
        try:
            # Map each finger type
            for finger_type, finger_mapping in mapping.items():
                vr_indices = finger_mapping["vr_indices"]
                shadow_indices = finger_mapping["shadow_indices"]
                
                # Extract joint positions from VR data
                finger_positions = vr_data[vr_indices]
                
                # Convert 3D positions to joint angles
                finger_angles = self._convert_positions_to_angles(
                    finger_positions, 
                    finger_type
                )
                
                # Assign to Shadow Hand indices
                for i, shadow_idx in enumerate(shadow_indices):
                    if i < len(finger_angles):
                        joint_positions[shadow_idx] = finger_angles[i]
            
            # Apply smoothing
            if self.smoothing_config["enabled"]:
                joint_positions = self._apply_smoothing(joint_positions)
            
            return joint_positions
            
        except Exception as e:
            self.logger.error(f"VR to Shadow Hand mapping failed: {e}")
            return np.zeros(20)
    
    def _convert_positions_to_angles(self, 
                                   positions: np.ndarray, 
                                   finger_type: str) -> np.ndarray:
        """
        Convert 3D positions to joint angles
        
        Args:
            positions: 3D position array
            finger_type: Finger type
            
        Returns:
            Joint angle array
        """
        angles = np.zeros(len(positions))
        
        try:
            if finger_type == "wrist":
                # Wrist: Convert X, Y coordinates to roll, pitch
                angles[0] = positions[0, 0] * self.scale_factors["wrist"]  # X -> roll
                angles[1] = positions[0, 1] * self.scale_factors["wrist"]  # Y -> pitch
                
            elif finger_type == "thumb":
                # Thumb: Use different axes for each joint
                for i in range(len(positions)):
                    if i == 0:  # THJ0: Y-axis
                        angles[i] = positions[i, 1] * self.scale_factors["thumb"]
                    elif i == 1:  # THJ1: Z-axis
                        angles[i] = positions[i, 2] * self.scale_factors["thumb"]
                    else:  # THJ2-4: X-axis
                        angles[i] = positions[i, 0] * self.scale_factors["thumb"]
                        
            else:  # Fingers
                # Fingers: Use different axes for each joint
                for i in range(len(positions)):
                    if i == 0:  # MCP: Y-axis
                        angles[i] = positions[i, 1] * self.scale_factors["fingers"]
                    elif i == 1:  # PIP: Z-axis
                        angles[i] = positions[i, 2] * self.scale_factors["fingers"]
                    elif i == 2:  # DIP: X-axis
                        angles[i] = positions[i, 0] * self.scale_factors["fingers"]
                    elif i == 3:  # Tip: Y-axis
                        angles[i] = positions[i, 1] * self.scale_factors["fingers"]
            
            return angles
            
        except Exception as e:
            self.logger.error(f"Position to angle conversion failed: {e}")
            return np.zeros(len(positions))
    
    def _apply_smoothing(self, joint_positions: np.ndarray) -> np.ndarray:
        """Apply smoothing to joint positions"""
        if not self.smoothing_config["enabled"]:
            return joint_positions
        
        # Add to smoothing buffer
        self.smoothing_buffer.append(joint_positions.copy())
        
        # Limit buffer size
        if len(self.smoothing_buffer) > self.smoothing_config["buffer_size"]:
            self.smoothing_buffer.pop(0)
        
        # Smooth with weighted average
        if len(self.smoothing_buffer) > 1:
            weights = np.exp(np.linspace(
                -self.smoothing_config["factor"], 
                0, 
                len(self.smoothing_buffer)
            ))
            weights = weights / np.sum(weights)
            
            smoothed_positions = np.zeros_like(joint_positions)
            for i, positions in enumerate(self.smoothing_buffer):
                smoothed_positions += weights[i] * positions
            
            return smoothed_positions
        
        return joint_positions
    
    def apply_joint_limits(self, 
                          joint_positions: np.ndarray, 
                          joint_limits: Dict) -> np.ndarray:
        """
        Apply joint limit ranges
        
        Args:
            joint_positions: Joint position array
            joint_limits: Joint-specific limit ranges
            
        Returns:
            Joint positions with limits applied
        """
        limited_positions = joint_positions.copy()
        
        try:
            # Apply limit ranges for each joint type
            for joint_type, limits in joint_limits.items():
                if joint_type == "wrist":
                    # Wrist joints (WRJ0, WRJ1)
                    indices = [17, 18]
                    limited_positions[indices] = np.clip(
                        limited_positions[indices], 
                        limits[0], 
                        limits[1]
                    )
                
                elif joint_type == "thumb":
                    # Thumb joints (THJ0-4)
                    indices = [0, 1, 2, 3, 4]
                    limited_positions[indices] = np.clip(
                        limited_positions[indices], 
                        limits[0], 
                        limits[1]
                    )
                
                elif joint_type == "fingers":
                    # Finger joints (FFJ, MFJ, RFJ, LFJ)
                    indices = list(range(5, 17))
                    limited_positions[indices] = np.clip(
                        limited_positions[indices], 
                        limits[0], 
                        limits[1]
                    )
            
            return limited_positions
            
        except Exception as e:
            self.logger.error(f"Joint limit application failed: {e}")
            return joint_positions
    
    def get_joint_mapping_info(self, vr_system: str = "steamvr") -> Dict:
        """Return joint mapping information"""
        if vr_system not in self.joint_mappings:
            return {}
        
        return self.joint_mappings[vr_system]
    
    def update_scale_factors(self, new_factors: Dict):
        """Update scale factors"""
        self.scale_factors.update(new_factors)
        self.logger.info(f"Scale factors updated: {new_factors}")
    
    def update_smoothing_config(self, new_config: Dict):
        """Update smoothing configuration"""
        self.smoothing_config.update(new_config)
        self.logger.info(f"Smoothing configuration updated: {new_config}")
    
    def reset_smoothing_buffer(self):
        """Reset smoothing buffer"""
        self.smoothing_buffer.clear()
        self.logger.info("Smoothing buffer reset completed")
