# Isaac-VR Quick Start Guide 

## Overview
This guide provides quick start instructions for the Isaac-VR system, which enables VR-based teleoperation and imitation learning for Isaac Sim robots.

## Prerequisites
- Isaac Sim 5.0 (Windows environment)
- Python 3.8+
- SteamVR or Quest 3 (optional for standalone mode)
- IsaacLab environment setup (optional for standalone mode)

## Quick Commands

### Basic Teleoperation
```bash
# Run basic teleoperation
python shadow_hand/run_teleop.py

# Run motion recording
python shadow_hand/record_motion.py

# Test external dataset
python shadow_hand/test_external_data.py
```

### Isaac Lab Mimic Integration
```bash
# Run full pipeline (demo collection → conversion → training → evaluation)
python shadow_hand/run_mimic_pipeline.py --action full --task "Isaac-Repose-Cube-Shadow-Direct-v0"

# Run demo collection only
python shadow_hand/run_mimic_pipeline.py --action collect --task "Isaac-Repose-Cube-Shadow-Direct-v0" --num-demos 10

# Use Isaac Lab Mimic demo recorder directly
python shadow_hand/mimic_demo_recorder.py --task "Isaac-Repose-Cube-Shadow-Direct-v0" --num-demos 5
```

### Standalone Mode (when Isaac Lab/VR not available)
```bash
# Generate dummy demo data
python shadow_hand/run_mimic_standalone.py --action generate --task "test_task" --num-demos 3

# Convert existing dataset to Robomimic format
python shadow_hand/run_mimic_standalone.py --action convert --dataset "./mimic_demos/" --output "./robomimic_data/"
```

## Troubleshooting

### Common Issues
1. **Import Errors**: The system automatically falls back to standalone mode if Isaac Lab or VR components are not available
2. **VR System Warnings**: Fixed infinite warning loop issue in MotionMapper
3. **Encoding Issues**: All Korean text and emojis have been removed for Windows console compatibility

### Error Messages
- `ModuleNotFoundError: No module named 'isaaclab'` → System runs in standalone mode
- `ModuleNotFoundError: No module named 'gymnasium'` → System runs in demo mode
- `UnicodeEncodeError: 'charmap' codec can't encode` → Fixed by removing non-ASCII characters

## Recent Fixes

### Git Commit Message
```
fix: resolve infinite VR system warnings and Korean script compatibility issues

- Fix infinite "unsupported VR system" warning loop in MotionMapper
- Add is_dummy_mode parameter to MotionMapper constructor
- Implement warning suppression for standalone/demo mode
- Remove all Korean text and emojis from scripts for Windows console compatibility
- Translate all Korean messages, docstrings, and comments to English
- Add robust error handling for Isaac Lab and VR system imports
- Enable graceful fallback to standalone mode when dependencies unavailable
- Fix JSON serialization issues with NumPy types in standalone mode
- Update MotionMapper to handle "auto" VR system type gracefully
- Ensure Windows charmap encoding compatibility across all scripts

Files modified:
- shadow_hand/core/motion_mapper.py: Add dummy mode support and warning suppression
- shadow_hand/mimic_demo_recorder.py: Pass dummy mode flag and remove Korean text
- shadow_hand/mimic_integration.py: Remove Korean text and add import error handling
- shadow_hand/robomimic_integration.py: Remove Korean text and emojis
- shadow_hand/run_mimic_pipeline.py: Remove Korean text and configure Windows logging
- shadow_hand/run_mimic_standalone.py: Remove Korean text and fix NumPy JSON serialization
- shadow_hand/core/hand_controller.py: Remove Korean text and emojis
- shadow_hand/core/vr_tracker.py: Remove Korean text and emojis
- shadow_hand/configs/teleop_config.yaml: Update configuration for Isaac Lab Mimic integration
- shadow_hand/README.md: Update documentation and remove Korean text
- shadow_hand/MIMIC_GUIDE.md: Create comprehensive usage guide

This commit resolves the critical issue where the system would get stuck in an infinite
warning loop when running in standalone mode, and ensures full Windows console
compatibility by removing all non-ASCII characters that could cause encoding errors.
```

## Support
For additional help, refer to the main README.md and MIMIC_GUIDE.md files in the shadow_hand directory.
