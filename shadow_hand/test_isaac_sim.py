#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Isaac Sim Script Editor
"""

import sys
import os

# Add the custom scripts path
sys.path.append(r'C:\IsaacLab\IsaacLab\custom_scripts\isaac-vr')

try:
    # Import IsaacLab
    import isaaclab
    import isaaclab_tasks
    print("IsaacLab imported successfully!")
    
    # Import gymnasium
    import gymnasium as gym
    print("Gymnasium imported successfully!")
    
    # Check available environments
    print("Available Isaac environments:")
    isaac_envs = [env for env in gym.envs.registry.keys() if 'isaac' in env.lower()]
    for env in isaac_envs[:10]:  # Show first 10
        print(f"  {env}")
    
    # Try to create a simple environment
    try:
        env = gym.make("Isaac-Cartpole-Direct-v0", render_mode="human")
        print("Environment created successfully!")
        
        # Reset environment
        env.reset()
        print("Environment reset successfully!")
        
        # Close environment
        env.close()
        print("Environment closed successfully!")
        
    except Exception as e:
        print(f"Environment creation failed: {e}")
        
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
