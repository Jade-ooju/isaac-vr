# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different dexterous hands.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/hands.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.shapes import shapes_cfg
from isaaclab.sim.spawners.shapes import shapes

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG  # isort:skip
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # isort:skip


class VRHandMotionSequence:
    """Manages sequences of Quest 3 VR hand motion data from multiple JSON files."""
    
    def __init__(self, dataset_path: str, sequence_type: str = "Pick", target_duration: float = 3.0):
        """Initialize the VR hand motion sequence manager.
        
        Args:
            dataset_path: Path to the Quest dataset directory
            sequence_type: Type of sequence to play ("Hold", "Place", "Pick", or "All")
            target_duration: Target duration in seconds for each animation (default: 3.0)
        """
        self.dataset_path = Path(dataset_path)
        self.sequence_type = sequence_type
        self.target_duration = target_duration
        self.motion_groups = {
            'Hold': [],
            'Place': [],
            'Pick': []
        }
        self.current_group = sequence_type
        self.current_file_index = 0
        self.current_frame_index = 0
        self.motion_data = None
        self.frame_skip = 1  # Will be calculated based on target duration
        
        # Hand joint names in order
        self.joint_names = [
            "Wrist", "ForearmWrist", "Palm",
            "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
            "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
            "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
            "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
            "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyTip"
        ]
        
        self._scan_and_categorize_files()
        self._load_current_motion()
        
        total_files = sum(len(files) for files in self.motion_groups.values())
        print(f"[INFO]: Found {total_files} VR motion files total")
        print(f"[INFO]: Hold: {len(self.motion_groups['Hold'])}, Place: {len(self.motion_groups['Place'])}, Pick: {len(self.motion_groups['Pick'])}")
        
    def _scan_and_categorize_files(self):
        """Scan all JSON files and categorize them by type."""
        if not self.dataset_path.exists():
            print(f"[ERROR]: Dataset path does not exist: {self.dataset_path}")
            return
        
        for file_path in self.dataset_path.glob("*.json"):
            filename = file_path.name
            if filename.startswith('Hold_'):
                self.motion_groups['Hold'].append(filename)
            elif filename.startswith('Place_'):
                self.motion_groups['Place'].append(filename)
            elif filename.startswith('Pick_'):
                self.motion_groups['Pick'].append(filename)
        
        # Sort each group
        for group in self.motion_groups:
            self.motion_groups[group].sort()
    
    def _load_current_motion(self) -> bool:
        """Load the current motion file."""
        current_group_files = self.motion_groups[self.current_group]
        if not current_group_files or self.current_file_index >= len(current_group_files):
            print(f"[ERROR]: No files available in group {self.current_group}")
            return False
        
        motion_file = current_group_files[self.current_file_index]
        file_path = self.dataset_path / motion_file
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                if len(content) == 0:
                    print(f"[ERROR]: File is empty: {file_path}")
                    return False
                
                self.motion_data = json.loads(content)
                total_frames = len(self.motion_data.get('frames', []))
                
                # Calculate frame skip to achieve target duration
                if total_frames > 0:
                    # Assuming 30 FPS for VR data (typical Quest frame rate)
                    original_duration = total_frames / 30.0
                    self.frame_skip = max(1, int(original_duration / self.target_duration))
                    actual_duration = (total_frames / self.frame_skip) / 30.0
                    print(f"[INFO]: Original duration: {original_duration:.2f}s, Target: {self.target_duration}s, Frame skip: {self.frame_skip}, Actual duration: {actual_duration:.2f}s")
                else:
                    self.frame_skip = 1
            
            self.current_frame_index = 0
            motion_info = self.get_motion_info()
            print(f"[INFO]: Loaded VR motion: {motion_info}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR]: Failed to load motion file {motion_file}: {e}")
            return False
    
    def next_motion(self) -> bool:
        """Advance to the next motion in the sequence."""
        current_group_files = self.motion_groups[self.current_group]
        
        if self.current_file_index < len(current_group_files) - 1:
            # Same group, next file
            self.current_file_index += 1
        else:
            # Move to next group
            if self.current_group == 'Hold':
                self.current_group = 'Place'
            elif self.current_group == 'Place':
                self.current_group = 'Pick'
            else:  # Pick
                self.current_group = 'Hold'
            
            self.current_file_index = 0
        
        print(f"[INFO]: Switching to group: {self.current_group}, file: {self.current_file_index + 1}")
        return self._load_current_motion()
    
    def has_current_frame(self) -> bool:
        """Check if there's a current frame available."""
        return (self.motion_data is not None and 
                'frames' in self.motion_data and 
                self.current_frame_index < len(self.motion_data['frames']))
    
    def get_current_joint_positions(self) -> torch.Tensor:
        """Get joint positions for the current frame."""
        frame_data = self.get_current_frame()
        return self.get_joint_positions(frame_data)
    
    def get_current_frame(self) -> dict | None:
        """Get the current frame data.
        
        Returns:
            Dictionary containing joint positions and metadata, or None if no data.
        """
        if not self.has_current_frame():
            return None
            
        if self.motion_data and "frames" in self.motion_data:
            return self.motion_data["frames"][self.current_frame_index]
        return None
    
    def get_joint_positions(self, frame_data: dict | None = None) -> torch.Tensor:
        """Extract joint positions from frame data.
        
        Args:
            frame_data: Frame data dictionary from the motion file. If None, uses current frame.
            
        Returns:
            Tensor of shape (num_joints, 3) containing joint positions.
        """
        if frame_data is None:
            frame_data = self.get_current_frame()
        
        if not frame_data or "joints" not in frame_data:
            return torch.zeros(len(self.joint_names), 3)
            
        positions = []
        joint_dict = {joint["jointName"]: joint for joint in frame_data["joints"]}
        
        for joint_name in self.joint_names:
            if joint_name in joint_dict:
                pos = joint_dict[joint_name]["position"]
                # Convert from Unity coordinate system to Isaac Sim coordinate system
                # Unity: X=Right, Y=Up, Z=Forward (Left-handed)
                # Isaac Sim: X=Forward, Y=Left, Z=Up (Right-handed)
                # Conversion: Unity(x,y,z) -> Isaac Sim(z,-x,y)
                isaac_pos = [pos["z"], -pos["x"], pos["y"]]
                positions.append(isaac_pos)
            else:
                # If joint not found, use zero position
                positions.append([0.0, 0.0, 0.0])
                
        return torch.tensor(positions, dtype=torch.float32)
    
    def next_frame(self) -> bool:
        """Move to the next frame with frame skipping.
        
        Returns:
            True if there are more frames, False if at the end.
        """
        if not self.has_current_frame():
            return False
            
        # Skip frames to achieve target duration
        self.current_frame_index += self.frame_skip
        if self.motion_data and "frames" in self.motion_data:
            return self.current_frame_index < len(self.motion_data["frames"])
        return False
    
    def reset_frame(self):
        """Reset to the first frame."""
        self.current_frame_index = 0
    
    def get_motion_info(self) -> str:
        """Get information about the current motion.
        
        Returns:
            String with motion information.
        """
        if not self.motion_data:
            return "No motion loaded"
        
        current_group_files = self.motion_groups[self.current_group]
        if not current_group_files or self.current_file_index >= len(current_group_files):
            return "Invalid motion state"
        
        motion_file = current_group_files[self.current_file_index]
        num_frames = len(self.motion_data.get("frames", []))
        interaction_target = self.motion_data.get("interactionTargetObject", "Unknown")
        effective_frames = num_frames // self.frame_skip if self.frame_skip > 0 else num_frames
        
        return f"Group: {self.current_group} | Motion: {motion_file} | Frames: {num_frames} (skip: {self.frame_skip}) | Target: {interaction_target} | Frame: {self.current_frame_index + 1}/{effective_frames}"


class VRToShadowHandIK:
    """Inverse Kinematics solver for mapping VR hand data to Shadow Hand joint angles."""
    
    def __init__(self):
        """Initialize the IK solver with joint mappings and limits."""
        # Shadow Hand joint names (24 joints - includes additional joints)
        self.shadow_joint_names = [
            "robot0_WRJ1", "robot0_WRJ0",  # Wrist (2)
            "robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1",  # Index (3)
            "robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1",  # Middle (3)
            "robot0_RFJ3", "robot0_RFJ2", "robot0_RFJ1",  # Ring (3)
            "robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1",  # Little (4)
            "robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0",  # Thumb (5)
            "robot0_FFJ0", "robot0_MFJ0", "robot0_RFJ0", "robot0_LFJ0"  # Additional joints (4)
        ]
        
        # Joint limits (radians) - conservative limits for safety
        self.joint_limits = torch.tensor([
            [-0.5, 0.5], [-0.5, 0.5],  # Wrist
            [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57],  # Index
            [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57],  # Middle
            [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57],  # Ring
            [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57],  # Little
            [-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8], [-0.8, 0.8],  # Thumb
            [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57], [-1.57, 1.57]  # Additional joints
        ])
        
        # VR joint to Shadow Hand finger mapping
        self.finger_mappings = {
            "thumb": {
                "vr_joints": ["ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip"],
                "shadow_joints": [16, 17, 18, 19, 20],  # THJ0-4 indices
                "base_joint": "Palm"
            },
            "index": {
                "vr_joints": ["IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip"],
                "shadow_joints": [5, 6, 7],  # FFJ1-3 indices
                "base_joint": "Palm"
            },
            "middle": {
                "vr_joints": ["MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip"],
                "shadow_joints": [8, 9, 10],  # MFJ1-3 indices
                "base_joint": "Palm"
            },
            "ring": {
                "vr_joints": ["RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip"],
                "shadow_joints": [11, 12, 13],  # RFJ1-3 indices
                "base_joint": "Palm"
            },
            "little": {
                "vr_joints": ["PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyTip"],
                "shadow_joints": [14, 15, 16, 17],  # LFJ1-4 indices
                "base_joint": "Palm"
            }
        }
        
        # Previous joint positions for smoothing
        self.prev_joint_positions = torch.zeros(24)
        self.smoothing_factor = 0.3
        
        # VR wrist position for direct mapping
        self.vr_wrist_position = None
    
    def solve_ik(self, vr_joint_positions: torch.Tensor, vr_joint_names: list) -> torch.Tensor:
        """Solve IK to convert VR joint positions to Shadow Hand joint angles.
        Focus on wrist movement first, then simple finger mapping.
        
        Args:
            vr_joint_positions: Tensor of shape (num_joints, 3) with VR joint positions
            vr_joint_names: List of VR joint names corresponding to positions
            
        Returns:
            Tensor of shape (24,) with Shadow Hand joint angles
        """
        # Create joint position dictionary
        joint_dict = {name: pos for name, pos in zip(vr_joint_names, vr_joint_positions)}
        
        # Initialize result
        shadow_joint_angles = torch.zeros(24)
        
        # Step 1: Map wrist position directly to robot0_wrist
        wrist_success = self._map_wrist_position_direct(joint_dict)
        if not wrist_success:
            # Fallback to angle-based mapping if direct mapping fails
            wrist_angles = self._map_wrist_movement(joint_dict)
            shadow_joint_angles[0] = wrist_angles[0]  # WRJ1
            shadow_joint_angles[1] = wrist_angles[1]  # WRJ0
        
        # Step 2: Simple finger mapping based on VR joint positions
        finger_angles = self._map_fingers_simple(joint_dict)
        shadow_joint_angles[2:22] = finger_angles  # Finger joints (20 joints)
        
        # Step 3: Set additional joints to zero for now
        shadow_joint_angles[22:24] = 0.0  # Additional joints
        
        # Apply joint limits
        shadow_joint_angles = self._apply_joint_limits(shadow_joint_angles)
        
        # Apply smoothing
        shadow_joint_angles = self._apply_smoothing(shadow_joint_angles)
        
        # Update previous positions
        self.prev_joint_positions = shadow_joint_angles.clone()
        
        return shadow_joint_angles
    
    def _map_wrist_position_direct(self, joint_dict: dict) -> bool:
        """Map VR ForearmWrist position directly to robot0_wrist position.
        
        Args:
            joint_dict: Dictionary mapping VR joint names to positions
            
        Returns:
            bool: True if mapping was successful, False otherwise
        """
        # Look for ForearmWrist in VR data
        forearm_wrist_pos = None
        forearm_wrist_candidates = ["ForearmWrist", "forearmWrist", "Forearm", "forearm"]
        
        for candidate in forearm_wrist_candidates:
            if candidate in joint_dict:
                forearm_wrist_pos = joint_dict[candidate]
                break
        
        if forearm_wrist_pos is None:
            return False
        
        # Apply position directly to robot0_wrist
        # This would require access to the robot's wrist prim
        # For now, we'll store the position for later use
        self.vr_wrist_position = forearm_wrist_pos.clone()
        
        return True
    
    def _map_wrist_movement(self, joint_dict: dict) -> torch.Tensor:
        """Map VR wrist movement to Shadow Hand wrist joints.
        
        Args:
            joint_dict: Dictionary mapping VR joint names to positions
            
        Returns:
            Tensor of shape (2,) with wrist joint angles [WRJ1, WRJ0]
        """
        # Try to find wrist and palm positions with different possible names
        wrist_pos = None
        palm_pos = None
        
        # Look for wrist position with various possible names
        wrist_candidates = ["Wrist", "wrist", "Hand", "hand", "HandCenter", "handCenter"]
        for candidate in wrist_candidates:
            if candidate in joint_dict:
                wrist_pos = joint_dict[candidate]
                break
        
        # Look for palm position with various possible names  
        palm_candidates = ["Palm", "palm", "HandCenter", "handCenter", "HandPalm", "handPalm"]
        for candidate in palm_candidates:
            if candidate in joint_dict:
                palm_pos = joint_dict[candidate]
                break
        
        # If we can't find wrist or palm, use first available joint as reference
        if wrist_pos is None or palm_pos is None:
            available_joints = list(joint_dict.keys())
            if available_joints:
                ref_joint = joint_dict[available_joints[0]]
                if wrist_pos is None:
                    wrist_pos = ref_joint
                if palm_pos is None:
                    palm_pos = ref_joint
        
        # Fallback to zero if still no positions found
        if wrist_pos is None:
            wrist_pos = torch.zeros(3)
        if palm_pos is None:
            palm_pos = torch.zeros(3)
        
        # Calculate wrist orientation relative to palm
        # WRJ1: Pitch (up/down movement)
        # WRJ0: Yaw (left/right movement)
        
        wrist_angles = torch.zeros(2)
        
        # Map Z difference to pitch (WRJ1)
        z_diff = wrist_pos[2] - palm_pos[2]
        wrist_angles[0] = torch.clamp(z_diff * 5.0, -0.5, 0.5)  # WRJ1 - increased sensitivity
        
        # Map Y difference to yaw (WRJ0)  
        y_diff = wrist_pos[1] - palm_pos[1]
        wrist_angles[1] = torch.clamp(y_diff * 5.0, -0.5, 0.5)  # WRJ0 - increased sensitivity
        
        return wrist_angles
    
    def _map_fingers_simple(self, joint_dict: dict) -> torch.Tensor:
        """Simple finger mapping based on VR joint positions.
        
        Args:
            joint_dict: Dictionary mapping VR joint names to positions
            
        Returns:
            Tensor of shape (20,) with finger joint angles
        """
        finger_angles = torch.zeros(20)
        
        # Get palm position as reference
        palm_pos = joint_dict.get("Palm", torch.zeros(3))
        
        # Map each finger based on tip position relative to palm
        finger_mappings = {
            "IndexTip": [2, 3, 4],      # FFJ1, FFJ2, FFJ3
            "MiddleTip": [5, 6, 7],     # MFJ1, MFJ2, MFJ3  
            "RingTip": [8, 9, 10],      # RFJ1, RFJ2, RFJ3
            "PinkyTip": [11, 12, 13, 14], # LFJ1, LFJ2, LFJ3, LFJ4
            "ThumbTip": [15, 16, 17, 18, 19]  # THJ0, THJ1, THJ2, THJ3, THJ4
        }
        
        for tip_name, joint_indices in finger_mappings.items():
            if tip_name in joint_dict:
                tip_pos = joint_dict[tip_name]
                
                # Calculate distance from palm
                distance = torch.norm(tip_pos - palm_pos)
                
                # Map distance to finger curl (simple linear mapping)
                # Closer to palm = more curled (positive angle)
                # Further from palm = less curled (negative angle)
                curl_angle = torch.clamp((0.1 - distance) * 10.0, -1.57, 1.57)
                
                # Apply to all joints of this finger
                for joint_idx in joint_indices:
                    if joint_idx < 20:
                        finger_angles[joint_idx] = curl_angle
        
        return finger_angles
    
    def _solve_finger_ik(self, joint_dict: dict, mapping: dict, finger_name: str) -> torch.Tensor:
        """Solve IK for a single finger.
        
        Args:
            joint_dict: Dictionary mapping VR joint names to positions
            mapping: Finger mapping configuration
            
        Returns:
            Tensor with finger joint angles
        """
        vr_joints = mapping["vr_joints"]
        base_joint = mapping["base_joint"]
        
        # Get base position
        if base_joint not in joint_dict:
            return torch.zeros(len(mapping["shadow_joints"]))
        
        base_pos = joint_dict[base_joint]
        finger_angles = []
        
        # Calculate angles for each joint segment
        for i in range(len(vr_joints) - 1):
            if vr_joints[i] in joint_dict and vr_joints[i + 1] in joint_dict:
                # Get joint positions
                joint1_pos = joint_dict[vr_joints[i]]
                joint2_pos = joint_dict[vr_joints[i + 1]]
                
                # Calculate direction vector
                direction = joint2_pos - joint1_pos
                
                # Convert to angle (simplified 2D projection)
                if finger_name == "thumb":
                    # Thumb has different angle calculation
                    angle = torch.atan2(direction[1], direction[0])
                else:
                    # Other fingers: use Z component for flexion
                    angle = torch.atan2(direction[2], torch.norm(direction[:2]))
                
                finger_angles.append(angle.item())
            else:
                finger_angles.append(0.0)
        
        return torch.tensor(finger_angles)
    
    def _apply_joint_limits(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Apply joint limits to prevent unsafe positions."""
        for i in range(len(joint_angles)):
            joint_angles[i] = torch.clamp(joint_angles[i], 
                                        self.joint_limits[i][0], 
                                        self.joint_limits[i][1])
        return joint_angles
    
    def _apply_smoothing(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to prevent jerky movements."""
        return (1 - self.smoothing_factor) * joint_angles + self.smoothing_factor * self.prev_joint_positions


class VRHandVisualizer:
    """Visualizer for VR hand motion data using spheres with gradient materials."""
    
    def __init__(self, prim_path: str = "/World/VRHand"):
        """Initialize the VR hand visualizer.
        
        Args:
            prim_path: Base prim path for the hand visualization.
        """
        self.prim_path = prim_path
        self.joint_spheres = {}
        self.joint_materials = {}
        self.joint_names = [
            "Wrist", "ForearmWrist", "Palm",
            "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
            "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
            "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
            "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
            "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyTip"
        ]
        
        # Define joint hierarchy for gradient calculation (0=base, higher=tip)
        self.joint_hierarchy = {
            "Wrist": 0, "ForearmWrist": 0, "Palm": 0,
            "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbDistal": 3, "ThumbTip": 4,
            "IndexMetacarpal": 1, "IndexProximal": 2, "IndexIntermediate": 3, "IndexDistal": 4, "IndexTip": 5,
            "MiddleMetacarpal": 1, "MiddleProximal": 2, "MiddleIntermediate": 3, "MiddleDistal": 4, "MiddleTip": 5,
            "RingMetacarpal": 1, "RingProximal": 2, "RingIntermediate": 3, "RingDistal": 4, "RingTip": 5,
            "PinkyMetacarpal": 1, "PinkyProximal": 2, "PinkyIntermediate": 3, "PinkyDistal": 4, "PinkyTip": 5
        }
        
        self._create_joint_spheres()
    
    def _get_joint_gradient_color(self, joint_name: str) -> tuple:
        """Get gradient blue color for a joint based on its hierarchy.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            RGB color tuple (r, g, b) with gradient blue
        """
        hierarchy_level = self.joint_hierarchy.get(joint_name, 0)
        max_level = 5  # Maximum hierarchy level (fingertips)
        
        # Calculate gradient: darker blue at base (0) to lighter blue at tips (max_level)
        # Base color: dark blue (0.0, 0.0, 0.8)
        # Tip color: light blue (0.7, 0.9, 1.0)
        if hierarchy_level == 0:
            # Base joints (Wrist, Palm) - darkest blue
            return (0.0, 0.0, 0.8)
        else:
            # Finger joints - gradient from dark to light blue
            t = hierarchy_level / max_level  # 0 to 1
            r = 0.0 + (0.7 - 0.0) * t  # 0.0 to 0.7
            g = 0.0 + (0.9 - 0.0) * t  # 0.0 to 0.9
            b = 0.8 + (1.0 - 0.8) * t  # 0.8 to 1.0
            return (r, g, b)
    
    def _create_joint_spheres(self):
        """Create sphere primitives for each joint with gradient materials."""
        # Create parent group
        prim_utils.create_prim(f"{self.prim_path}", "Xform")
        
        for joint_name in self.joint_names:
            sphere_path = f"{self.prim_path}/{joint_name}"
            
            # Create sphere using Isaac Lab spawner (larger radius for better visibility)
            sphere_cfg = shapes_cfg.SphereCfg(radius=0.0067)
            shapes.spawn_sphere(sphere_path, sphere_cfg)
            
            # Create material with gradient blue color
            self._create_material_for_joint(joint_name, sphere_path)
            
            self.joint_spheres[joint_name] = sphere_path
    
    def _create_material_for_joint(self, joint_name: str, sphere_path: str):
        """Create a material with gradient blue color for a joint.
        
        Args:
            joint_name: Name of the joint
            sphere_path: USD path to the sphere primitive
        """
        from pxr import UsdShade, Sdf
        
        # Get gradient color for this joint
        color = self._get_joint_gradient_color(joint_name)
        
        # Create material path
        material_path = f"{self.prim_path}/Materials/{joint_name}_Material"
        
        # Create material
        material = UsdShade.Material.Define(stage_utils.get_current_stage(), material_path)
        
        # Create shader
        shader = UsdShade.Shader.Define(stage_utils.get_current_stage(), f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # Set shader parameters
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((color[0], color[1], color[2]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        
        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        
        # Bind material to sphere
        sphere_prim = prim_utils.get_prim_at_path(sphere_path)
        if sphere_prim:
            UsdShade.MaterialBindingAPI.Apply(sphere_prim).Bind(material)
        
        # Store material reference
        self.joint_materials[joint_name] = material_path
    
    def update_joint_positions(self, joint_positions: torch.Tensor, offset: torch.Tensor | None = None):
        """Update the positions of all joint spheres.
        
        Args:
            joint_positions: Tensor of shape (num_joints, 3) containing joint positions.
            offset: Optional offset to apply to all positions.
        """
        if joint_positions is None or len(joint_positions) != len(self.joint_names):
            return
            
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in self.joint_spheres:
                position = joint_positions[i].tolist()
                if offset is not None:
                    position = [p + o for p, o in zip(position, offset.tolist())]
                
                # Update sphere position using USD API
                from pxr import UsdGeom, Gf
                sphere_prim = prim_utils.get_prim_at_path(self.joint_spheres[joint_name])
                if sphere_prim:
                    xform = UsdGeom.Xformable(sphere_prim)
                    xform.ClearXformOpOrder()
                    # Convert position to GfVec3d
                    gf_position = Gf.Vec3d(position[0], position[1], position[2])
                    xform.AddTranslateOp().Set(gf_position)


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]], VRHandMotionSequence, VRHandVisualizer, VRToShadowHandIK]:
    """Designs the scene."""
    # Ground plane removed for better VR hand visibility
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Origin1: Shadow Hand, Origin2: VR Hand Visualization
    # Use custom positions based on captured image values
    origins = [
        [0.13587, 0.11907, -1.07026],  # Origin1: Shadow Hand position
        [0.0, 0.0, 0.0]                # Origin2: VR Hand at origin
    ]

    # Origin 1 with Shadow Hand
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    shadow_hand_cfg = SHADOW_HAND_CFG.__class__(**SHADOW_HAND_CFG.__dict__)
    shadow_hand_cfg.prim_path = "/World/Origin1/Robot"
    shadow_hand = Articulation(shadow_hand_cfg)

    # Origin 2 with VR Hand Visualization
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    
    # Initialize VR hand motion loader and visualizer
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "Quest")
    vr_motion_sequence = VRHandMotionSequence(dataset_path, sequence_type="Pick", target_duration=3.0)
    vr_hand_visualizer = VRHandVisualizer("/World/Origin2/VRHand")
    ik_solver = VRToShadowHandIK()

    # return the scene information
    scene_entities = {
        "shadow_hand": shadow_hand,
    }
    return scene_entities, origins, vr_motion_sequence, vr_hand_visualizer, ik_solver


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, 
                  vr_motion_sequence: VRHandMotionSequence, vr_hand_visualizer: VRHandVisualizer, ik_solver: VRToShadowHandIK):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Start with hand open
    grasp_mode = 0
    # VR motion control
    vr_frame_count = 0
    vr_motion_index = 0
    
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            vr_frame_count = 0
            # reset robots
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # joint state
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # reset the internal state
                robot.reset()
            # Reset VR motion
            vr_motion_sequence.reset_frame()
            print("[INFO]: Resetting robots state...")
        
        # Toggle grasp mode for robot hands
        if count % 100 == 0:
            grasp_mode = 1 - grasp_mode
        
        # Update VR hand visualization every few frames
        if count % 5 == 0 and vr_motion_sequence.has_current_frame():
            joint_positions = vr_motion_sequence.get_current_joint_positions()
            
            # Apply offset to position the VR hand next to the robot hand
            vr_offset = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
            vr_hand_visualizer.update_joint_positions(joint_positions, vr_offset)
            
            # Move to next frame
            if not vr_motion_sequence.next_frame():
                # If we've reached the end, move to next motion in sequence
                print(f"[INFO]: Motion completed, switching to next motion")
                if not vr_motion_sequence.next_motion():
                    print("[WARNING]: Failed to load next motion")
        
        # Apply VR motion to robot hands using IK
        for robot in entities.values():
            if vr_motion_sequence.has_current_frame():
                # Get VR joint positions and names
                vr_joint_positions = vr_motion_sequence.get_current_joint_positions()
                vr_joint_names = vr_motion_sequence.joint_names
                
                # Solve IK to get Shadow Hand joint angles
                shadow_joint_angles = ik_solver.solve_ik(vr_joint_positions, vr_joint_names)
                
                # Debug: Print IK results (less frequently)
                if count % 100 == 0:  # Print every 100 steps
                    print(f"[DEBUG] Frame {count}: VR data loaded, Shadow angles: {shadow_joint_angles[0]:.3f}, {shadow_joint_angles[1]:.3f}")
                
                # Apply to robot
                robot.set_joint_position_target(shadow_joint_angles)
                
                # Apply direct wrist position mapping if available
                if hasattr(ik_solver, 'vr_wrist_position') and ik_solver.vr_wrist_position is not None:
                    _update_robot_wrist_position(robot, ik_solver.vr_wrist_position)
            else:
                # Fallback to default grasp mode if no VR data
                joint_pos_target = robot.data.soft_joint_pos_limits[..., grasp_mode]
                robot.set_joint_position_target(joint_pos_target)
            
            # write data to sim
            robot.write_data_to_sim()
        
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1


def _update_robot_wrist_position(robot, vr_wrist_position):
    """Update robot's wrist position to match VR wrist position.
    
    Args:
        robot: The robot articulation object
        vr_wrist_position: VR wrist position as torch.Tensor
    """
    try:
        # Get the robot's wrist prim path
        # Assuming the robot is named "Robot" and wrist is "robot0_wrist"
        wrist_prim_path = "/World/Origin1/Robot/robot0_wrist"
        
        # Get the prim from the stage
        from isaacsim.core.utils import stage_utils
        stage = stage_utils.get_current_stage()
        wrist_prim = stage.GetPrimAtPath(wrist_prim_path)
        
        if wrist_prim.IsValid():
            # Convert VR position to Isaac Sim coordinates if needed
            # VR position is already in Isaac Sim coordinates from get_joint_positions
            isaac_position = vr_wrist_position
            
            # Update the wrist position
            xform = UsdGeom.Xformable(wrist_prim)
            if xform:
                # Clear existing transform
                xform.ClearXformOpOrder()
                # Set new translation
                xform.AddTranslateOp().Set(Gf.Vec3d(isaac_position[0], isaac_position[1], isaac_position[2]))
            else:
                pass  # Silent failure
        else:
            pass  # Silent failure
                
        except Exception as e:
            pass  # Silent failure


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=(0.0, -0.5, 1.5), target=(0.0, -0.2, 0.5))
    # design scene
    scene_entities, scene_origins, vr_motion_sequence, vr_hand_visualizer, ik_solver = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, vr_motion_sequence, vr_hand_visualizer, ik_solver)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
