"""
좌표계 변환 유틸리티
VR 시스템과 Isaac Sim 간의 좌표계 변환을 담당
"""

import numpy as np
from typing import Tuple, List, Optional

class CoordinateTransformer:
    """좌표계 변환 클래스"""
    
    def __init__(self):
        # VR 시스템별 좌표계 설정
        self.vr_systems = {
            "steamvr": {
                "handedness": "right",
                "scale": 1.0,  # 미터 단위
                "origin_offset": np.array([0.0, 0.0, 0.0])
            },
            "quest3": {
                "handedness": "right", 
                "scale": 1.0,  # 미터 단위
                "origin_offset": np.array([0.0, 0.0, 0.0])
            }
        }
        
        # Isaac Sim 좌표계 설정
        self.isaac_sim = {
            "handedness": "right",
            "scale": 1.0,  # 미터 단위
            "origin_offset": np.array([0.0, 0.0, 0.0])
        }
    
    def transform_vr_to_isaac(self, 
                             vr_data: np.ndarray, 
                             vr_system: str = "auto") -> np.ndarray:
        """
        VR 데이터를 Isaac Sim 좌표계로 변환
        
        Args:
            vr_data: VR 핸드 데이터 (21개 관절, 3D 좌표)
            vr_system: VR 시스템 타입
            
        Returns:
            Isaac Sim 좌표계로 변환된 데이터
        """
        if vr_system == "auto":
            # 자동 감지 (기본값)
            vr_system = "steamvr"
        
        if vr_system not in self.vr_systems:
            raise ValueError(f"지원하지 않는 VR 시스템: {vr_system}")
        
        # VR 시스템 설정 가져오기
        vr_config = self.vr_systems[vr_system]
        
        # 스케일 변환
        scaled_data = vr_data * vr_config["scale"]
        
        # 원점 오프셋 적용
        transformed_data = scaled_data + vr_config["origin_offset"]
        
        return transformed_data
    
    def normalize_joint_positions(self, 
                                positions: np.ndarray, 
                                joint_limits: dict) -> np.ndarray:
        """
        관절 위치를 제한 범위 내로 정규화
        
        Args:
            positions: 관절 위치 배열
            joint_limits: 관절별 제한 범위
            
        Returns:
            정규화된 관절 위치
        """
        normalized = positions.copy()
        
        # 각 관절 타입별로 제한 범위 적용
        for joint_type, limits in joint_limits.items():
            if joint_type == "wrist":
                # 손목 관절 (WRJ0, WRJ1)
                indices = [17, 18]  # Shadow Hand 인덱스
                normalized[indices] = np.clip(positions[indices], limits[0], limits[1])
            
            elif joint_type == "thumb":
                # 엄지 관절 (THJ0-4)
                indices = [0, 1, 2, 3, 4]
                normalized[indices] = np.clip(positions[indices], limits[0], limits[1])
            
            elif joint_type == "fingers":
                # 손가락 관절 (FFJ, MFJ, RFJ, LFJ)
                indices = list(range(5, 17))  # 5-16
                normalized[indices] = np.clip(positions[indices], limits[0], limits[1])
        
        return normalized
    
    def apply_smoothing(self, 
                       current_pos: np.ndarray, 
                       target_pos: np.ndarray, 
                       smoothing_factor: float = 0.3) -> np.ndarray:
        """
        관절 위치에 스무딩 적용
        
        Args:
            current_pos: 현재 관절 위치
            target_pos: 목표 관절 위치
            smoothing_factor: 스무딩 계수 (0-1)
            
        Returns:
            스무딩이 적용된 관절 위치
        """
        return current_pos * (1 - smoothing_factor) + target_pos * smoothing_factor
    
    def convert_quaternion_to_euler(self, quaternion: np.ndarray) -> np.ndarray:
        """
        쿼터니언을 오일러 각도로 변환
        
        Args:
            quaternion: 쿼터니언 배열 [w, x, y, z]
            
        Returns:
            오일러 각도 [roll, pitch, yaw] (라디안)
        """
        # 간단한 변환 (정확한 변환은 scipy 사용 권장)
        w, x, y, z = quaternion
        
        # Roll (x축 회전)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        
        # Pitch (y축 회전)
        pitch = np.arcsin(2 * (w * y - z * x))
        
        # Yaw (z축 회전)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return np.array([roll, pitch, yaw])
    
    def get_joint_mapping(self, vr_system: str = "auto") -> dict:
        """
        VR 시스템과 Shadow Hand 간의 관절 매핑 반환
        
        Args:
            vr_system: VR 시스템 타입
            
        Returns:
            관절 매핑 딕셔너리
        """
        # 기본 매핑 (Quest 3/SteamVR 공통)
        mapping = {
            "wrist": {
                "vr_indices": [0, 1],  # VR 손목 관절
                "shadow_indices": [17, 18]  # Shadow Hand WRJ0, WRJ1
            },
            "thumb": {
                "vr_indices": [2, 3, 4, 5, 6],  # VR 엄지 관절
                "shadow_indices": [0, 1, 2, 3, 4]  # Shadow Hand THJ0-4
            },
            "index": {
                "vr_indices": [7, 8, 9, 10],  # VR 검지 관절
                "shadow_indices": [5, 6, 7]  # Shadow Hand FFJ1-3
            },
            "middle": {
                "vr_indices": [11, 12, 13, 14],  # VR 중지 관절
                "shadow_indices": [8, 9, 10]  # Shadow Hand MFJ1-3
            },
            "ring": {
                "vr_indices": [15, 16, 17, 18],  # VR 약지 관절
                "shadow_indices": [11, 12, 13]  # Shadow Hand RFJ1-3
            },
            "little": {
                "vr_indices": [19, 20],  # VR 새끼 관절
                "shadow_indices": [14, 15, 16]  # Shadow Hand LFJ1-3
            }
        }
        
        return mapping
