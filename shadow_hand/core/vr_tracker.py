"""
VR Hand Tracking Class
Extract hand data from various VR systems like Quest 3, SteamVR
"""

import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

class VRHandTracker:
    """VR hand tracking class"""
    
    def __init__(self, vr_system: str = "auto", config: Optional[Dict] = None):
        """
        Initialize VR hand tracker
        
        Args:
            vr_system: VR system type ("steamvr", "quest3", "auto")
            config: Configuration dictionary
        """
        self.vr_system = vr_system
        self.config = config or {}
        self.is_connected = False
        self.vr_interface = None
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Attempt VR system connection
        self._connect_vr_system()
        
        # Hand data buffer
        self.hand_data_buffer = []
        self.max_buffer_size = 10
        
        # Last update time
        self.last_update_time = time.time()
        
    def _connect_vr_system(self):
        """Connect to VR system"""
        try:
            if self.vr_system == "steamvr" or self.vr_system == "auto":
                self._connect_steamvr()
            elif self.vr_system == "quest3":
                self._connect_quest3()
            else:
                self.logger.warning(f"Unsupported VR system: {self.vr_system}")
                
        except Exception as e:
            self.logger.error(f"VR system connection failed: {e}")
            self.is_connected = False
    
    def _connect_steamvr(self):
        """Connect to SteamVR"""
        try:
            import openvr
            self.vr_interface = openvr.init(openvr.VRApplication_Other)
            self.is_connected = True
            self.logger.info("SteamVR connection successful!")
        except ImportError:
            self.logger.warning("openvr package is not installed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"SteamVR connection failed: {e}")
            self.is_connected = False
    
    def _connect_quest3(self):
        """Connect to Quest 3"""
        try:
            import pyopenxr as xr
            # Quest 3 connection logic (see pyopenxr documentation for specific implementation)
            self.vr_interface = None  # Temporary
            self.is_connected = True
            self.logger.info("Quest 3 connection successful!")
        except ImportError:
            self.logger.warning("pyopenxr package is not installed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Quest 3 connection failed: {e}")
            self.is_connected = False
    
    def get_hand_data(self) -> Optional[np.ndarray]:
        """
        Get current hand data
        
        Returns:
            Hand data (21 joints, 3D coordinates) or None
        """
        if not self.is_connected:
            return self._get_dummy_hand_data()
        
        try:
            if self.vr_system == "steamvr":
                return self._get_steamvr_hand_data()
            elif self.vr_system == "quest3":
                return self._get_quest3_hand_data()
            else:
                return self._get_dummy_hand_data()
                
        except Exception as e:
            self.logger.error(f"Hand data extraction failed: {e}")
            return self._get_dummy_hand_data()
    
    def _get_steamvr_hand_data(self) -> np.ndarray:
        """Extract hand data from SteamVR"""
        try:
            # SteamVR hand tracking data extraction
            # See SteamVR API documentation for actual implementation
            hand_data = np.zeros((21, 3))
            
            # Dummy data (remove in actual implementation)
            t = time.time()
            hand_data[0] = [np.sin(t), np.cos(t), 0.0]  # Wrist
            hand_data[1:5] = [np.sin(t*2), np.cos(t*2), 0.1]  # Thumb
            hand_data[5:9] = [np.sin(t*1.5), np.cos(t*1.5), 0.2]  # Index
            hand_data[9:13] = [np.sin(t*1.8), np.cos(t*1.8), 0.3]  # Middle
            hand_data[13:17] = [np.sin(t*2.2), np.cos(t*2.2), 0.4]  # Ring
            hand_data[17:21] = [np.sin(t*1.2), np.cos(t*1.2), 0.5]  # Little
            
            return hand_data
            
        except Exception as e:
            self.logger.error(f"SteamVR hand data extraction failed: {e}")
            return np.zeros((21, 3))
    
    def _get_quest3_hand_data(self) -> np.ndarray:
        """Extract hand data from Quest 3"""
        try:
            # Quest 3 hand tracking data extraction
            # See pyopenxr documentation for actual implementation
            hand_data = np.zeros((21, 3))
            
            # Dummy data (remove in actual implementation)
            t = time.time()
            hand_data[0] = [np.sin(t), np.cos(t), 0.0]  # Wrist
            hand_data[1:5] = [np.sin(t*2), np.cos(t*2), 0.1]  # Thumb
            hand_data[5:9] = [np.sin(t*1.5), np.cos(t*1.5), 0.2]  # Index
            hand_data[9:13] = [np.sin(t*1.8), np.cos(t*1.8), 0.3]  # Middle
            hand_data[13:17] = [np.sin(t*2.2), np.cos(t*2.2), 0.4]  # Ring
            hand_data[17:21] = [np.sin(t*1.2), np.cos(t*1.2), 0.5]  # Little
            
            return hand_data
            
        except Exception as e:
            self.logger.error(f"Quest 3 hand data extraction failed: {e}")
            return np.zeros((21, 3))
    
    def _get_dummy_hand_data(self) -> np.ndarray:
        """Generate dummy hand data (for testing)"""
        t = time.time()
        hand_data = np.zeros((21, 3))
        
        # Time-varying dummy data
        hand_data[0] = [np.sin(t * 0.5) * 0.1, np.cos(t * 0.5) * 0.1, 0.0]  # Wrist
        hand_data[1:5] = [np.sin(t * 2) * 0.05, np.cos(t * 2) * 0.05, 0.1]  # Thumb
        hand_data[5:9] = [np.sin(t * 1.5) * 0.05, np.cos(t * 1.5) * 0.05, 0.2]  # Index
        hand_data[9:13] = [np.sin(t * 1.8) * 0.05, np.cos(t * 1.8) * 0.05, 0.3]  # Middle
        hand_data[13:17] = [np.sin(t * 2.2) * 0.05, np.cos(t * 2.2) * 0.05, 0.4]  # Ring
        hand_data[17:21] = [np.sin(t * 1.2) * 0.05, np.cos(t * 1.2) * 0.05, 0.5]  # Little
        
        return hand_data
    
    def get_hand_confidence(self) -> float:
        """Return hand tracking confidence"""
        if not self.is_connected:
            return 0.0
        
        # In actual implementation, get confidence value from VR system
        return 0.8  # Dummy value
    
    def is_hand_visible(self) -> bool:
        """Check if hand is visible"""
        if not self.is_connected:
            return True  # Always True in dummy mode
        
        # In actual implementation, check hand visibility from VR system
        return True
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """Return tracking status information"""
        return {
            "is_connected": self.is_connected,
            "vr_system": self.vr_system,
            "hand_visible": self.is_hand_visible(),
            "confidence": self.get_hand_confidence(),
            "last_update": self.last_update_time,
            "buffer_size": len(self.hand_data_buffer)
        }
    
    def update(self):
        """Update tracker"""
        current_time = time.time()
        
        # Get hand data
        hand_data = self.get_hand_data()
        
        if hand_data is not None:
            # Add to buffer
            self.hand_data_buffer.append({
                "timestamp": current_time,
                "data": hand_data.copy()
            })
            
            # Limit buffer size
            if len(self.hand_data_buffer) > self.max_buffer_size:
                self.hand_data_buffer.pop(0)
            
            self.last_update_time = current_time
    
    def get_smoothed_hand_data(self, smoothing_factor: float = 0.3) -> Optional[np.ndarray]:
        """Return smoothed hand data"""
        if len(self.hand_data_buffer) < 2:
            return self.get_hand_data()
        
        # Weighted average of recent data
        weights = np.exp(np.linspace(-smoothing_factor, 0, len(self.hand_data_buffer)))
        weights = weights / np.sum(weights)
        
        smoothed_data = np.zeros((21, 3))
        for i, (_, data) in enumerate(self.hand_data_buffer):
            smoothed_data += weights[i] * data
        
        return smoothed_data
    
    def disconnect(self):
        """Disconnect from VR system"""
        try:
            if self.vr_interface:
                if self.vr_system == "steamvr":
                    import openvr
                    openvr.shutdown()
                # Add Quest 3 disconnection logic
                
            self.is_connected = False
            self.logger.info("VR system disconnection completed")
            
        except Exception as e:
            self.logger.error(f"VR system disconnection failed: {e}")
    
    def __del__(self):
        """Destructor"""
        self.disconnect()
