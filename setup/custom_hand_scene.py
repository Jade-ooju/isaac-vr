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
            print(f"[DEBUG]: Loading motion: {motion_file}")
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                if len(content) == 0:
                    print(f"[ERROR]: File is empty: {file_path}")
                    return False
                
                self.motion_data = json.loads(content)
                total_frames = len(self.motion_data.get('frames', []))
                print(f"[DEBUG]: Successfully loaded JSON with {total_frames} frames")
                
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
        
        print(f"[DEBUG]: Found {len(joint_dict)} joints in frame data")
        print(f"[DEBUG]: Available joint names: {list(joint_dict.keys())}")
        
        for joint_name in self.joint_names:
            if joint_name in joint_dict:
                pos = joint_dict[joint_name]["position"]
                positions.append([pos["x"], pos["y"], pos["z"]])
                if joint_name == "Wrist":  # Debug first joint
                    print(f"[DEBUG]: {joint_name} position: x={pos['x']}, y={pos['y']}, z={pos['z']}")
            else:
                # If joint not found, use zero position
                positions.append([0.0, 0.0, 0.0])
                print(f"[DEBUG]: Joint {joint_name} not found, using zero position")
                
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


class VRHandVisualizer:
    """Visualizer for VR hand motion data using spheres."""
    
    def __init__(self, prim_path: str = "/World/VRHand"):
        """Initialize the VR hand visualizer.
        
        Args:
            prim_path: Base prim path for the hand visualization.
        """
        self.prim_path = prim_path
        self.joint_spheres = {}
        self.joint_names = [
            "Wrist", "ForearmWrist", "Palm",
            "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
            "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
            "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
            "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
            "PinkyMetacarpal", "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyTip"
        ]
        
        # Different colors for different finger groups
        self.joint_colors = {
            "Wrist": (1.0, 1.0, 1.0),  # White
            "ForearmWrist": (0.8, 0.8, 0.8),  # Light gray
            "Palm": (0.6, 0.6, 0.6),  # Gray
            "Thumb": (1.0, 0.0, 0.0),  # Red
            "Index": (0.0, 1.0, 0.0),  # Green
            "Middle": (0.0, 0.0, 1.0),  # Blue
            "Ring": (1.0, 1.0, 0.0),  # Yellow
            "Pinky": (1.0, 0.0, 1.0),  # Magenta
        }
        
        self._create_joint_spheres()
    
    def _get_joint_color(self, joint_name: str) -> tuple:
        """Get color for a joint based on its name.
        
        Args:
            joint_name: Name of the joint.
            
        Returns:
            RGB color tuple.
        """
        if "Thumb" in joint_name:
            return self.joint_colors["Thumb"]
        elif "Index" in joint_name:
            return self.joint_colors["Index"]
        elif "Middle" in joint_name:
            return self.joint_colors["Middle"]
        elif "Ring" in joint_name:
            return self.joint_colors["Ring"]
        elif "Pinky" in joint_name:
            return self.joint_colors["Pinky"]
        elif joint_name in ["Wrist", "ForearmWrist", "Palm"]:
            return self.joint_colors[joint_name]
        else:
            return (0.5, 0.5, 0.5)  # Default gray
    
    def _create_joint_spheres(self):
        """Create sphere primitives for each joint."""
        # Create parent group
        prim_utils.create_prim(f"{self.prim_path}", "Xform")
        
        for joint_name in self.joint_names:
            sphere_path = f"{self.prim_path}/{joint_name}"
            
            # Create sphere using Isaac Lab spawner (larger radius for better visibility)
            sphere_cfg = shapes_cfg.SphereCfg(radius=0.01)
            shapes.spawn_sphere(sphere_path, sphere_cfg)
            
            # For now, just create the spheres without materials
            # We can add colors later if needed
            self.joint_spheres[joint_name] = sphere_path
    
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


def design_scene() -> tuple[dict, list[list[float]], VRHandMotionSequence, VRHandVisualizer]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Origin1: Shadow Hand, Origin2: VR Hand Visualization
    origins = define_origins(num_origins=2, spacing=0.8)

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

    # return the scene information
    scene_entities = {
        "shadow_hand": shadow_hand,
    }
    return scene_entities, origins, vr_motion_sequence, vr_hand_visualizer


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, 
                  vr_motion_sequence: VRHandMotionSequence, vr_hand_visualizer: VRHandVisualizer):
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
            print(f"[DEBUG]: Processing frame {vr_motion_sequence.current_frame_index}")
            joint_positions = vr_motion_sequence.get_current_joint_positions()
            
            # Apply offset to position the VR hand next to the robot hand
            vr_offset = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
            vr_hand_visualizer.update_joint_positions(joint_positions, vr_offset)
            
            # Debug: Print first few joint positions occasionally
            if count % 100 == 0:
                print(f"[DEBUG]: VR Hand - Frame {vr_motion_sequence.current_frame_index}, "
                      f"Wrist pos: {joint_positions[0].tolist()}, "
                      f"Palm pos: {joint_positions[2].tolist()}")
            
            # Move to next frame
            if not vr_motion_sequence.next_frame():
                # If we've reached the end, move to next motion in sequence
                print(f"[INFO]: Motion completed, switching to next motion")
                if not vr_motion_sequence.next_motion():
                    print("[WARNING]: Failed to load next motion")
        elif count % 5 == 0:
            print(f"[DEBUG]: No current frame available")
        
        # Apply default actions to the robot hands
        for robot in entities.values():
            # generate joint positions
            joint_pos_target = robot.data.soft_joint_pos_limits[..., grasp_mode]
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=(0.0, -0.5, 1.5), target=(0.0, -0.2, 0.5))
    # design scene
    scene_entities, scene_origins, vr_motion_sequence, vr_hand_visualizer = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, vr_motion_sequence, vr_hand_visualizer)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
