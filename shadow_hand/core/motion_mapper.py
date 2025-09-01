"""
모션 리타겟팅 클래스
VR 핸드 데이터를 Shadow Hand 관절 위치로 변환
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class MotionMapper:
    """모션 리타겟팅 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        모션 매퍼 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # VR 시스템별 관절 매핑
        self.joint_mappings = self._initialize_joint_mappings()
        
        # 변환 파라미터
        self.scale_factors = {
            "wrist": 0.1,      # 손목 관절 스케일
            "thumb": 0.1,      # 엄지 관절 스케일
            "fingers": 0.1     # 손가락 관절 스케일
        }
        
        # 스무딩 설정
        self.smoothing_config = {
            "enabled": True,
            "factor": 0.3,
            "buffer_size": 5
        }
        
        # 스무딩 버퍼
        self.smoothing_buffer = []
    
    def _initialize_joint_mappings(self) -> Dict:
        """관절 매핑 초기화"""
        return {
            "steamvr": {
                "wrist": {
                    "vr_indices": [0, 1],      # VR 손목 관절
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
            },
            "quest3": {
                # Quest 3는 SteamVR과 동일한 매핑 사용
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
        VR 데이터를 Shadow Hand 관절 위치로 매핑
        
        Args:
            vr_data: VR 핸드 데이터 (21개 관절, 3D 좌표)
            vr_system: VR 시스템 타입
            
        Returns:
            Shadow Hand 관절 위치 (20개 관절)
        """
        if vr_system not in self.joint_mappings:
            self.logger.warning(f"지원하지 않는 VR 시스템: {vr_system}, SteamVR 사용")
            vr_system = "steamvr"
        
        mapping = self.joint_mappings[vr_system]
        joint_positions = np.zeros(20)
        
        try:
            # 각 손가락 타입별로 매핑
            for finger_type, finger_mapping in mapping.items():
                vr_indices = finger_mapping["vr_indices"]
                shadow_indices = finger_mapping["shadow_indices"]
                
                # VR 데이터에서 해당 관절들의 위치 추출
                finger_positions = vr_data[vr_indices]
                
                # 3D 위치를 관절 각도로 변환
                finger_angles = self._convert_positions_to_angles(
                    finger_positions, 
                    finger_type
                )
                
                # Shadow Hand 인덱스에 할당
                for i, shadow_idx in enumerate(shadow_indices):
                    if i < len(finger_angles):
                        joint_positions[shadow_idx] = finger_angles[i]
            
            # 스무딩 적용
            if self.smoothing_config["enabled"]:
                joint_positions = self._apply_smoothing(joint_positions)
            
            return joint_positions
            
        except Exception as e:
            self.logger.error(f"VR to Shadow Hand 매핑 실패: {e}")
            return np.zeros(20)
    
    def _convert_positions_to_angles(self, 
                                   positions: np.ndarray, 
                                   finger_type: str) -> np.ndarray:
        """
        3D 위치를 관절 각도로 변환
        
        Args:
            positions: 3D 위치 배열
            finger_type: 손가락 타입
            
        Returns:
            관절 각도 배열
        """
        angles = np.zeros(len(positions))
        
        try:
            if finger_type == "wrist":
                # 손목: X, Y 좌표를 roll, pitch로 변환
                angles[0] = positions[0, 0] * self.scale_factors["wrist"]  # X -> roll
                angles[1] = positions[0, 1] * self.scale_factors["wrist"]  # Y -> pitch
                
            elif finger_type == "thumb":
                # 엄지: 각 관절별로 다른 축 사용
                for i in range(len(positions)):
                    if i == 0:  # THJ0: Y축
                        angles[i] = positions[i, 1] * self.scale_factors["thumb"]
                    elif i == 1:  # THJ1: Z축
                        angles[i] = positions[i, 2] * self.scale_factors["thumb"]
                    else:  # THJ2-4: X축
                        angles[i] = positions[i, 0] * self.scale_factors["thumb"]
                        
            else:  # 손가락들
                # 손가락: 각 관절별로 다른 축 사용
                for i in range(len(positions)):
                    if i == 0:  # MCP: Y축
                        angles[i] = positions[i, 1] * self.scale_factors["fingers"]
                    elif i == 1:  # PIP: Z축
                        angles[i] = positions[i, 2] * self.scale_factors["fingers"]
                    elif i == 2:  # DIP: X축
                        angles[i] = positions[i, 0] * self.scale_factors["fingers"]
                    elif i == 3:  # Tip: Y축
                        angles[i] = positions[i, 1] * self.scale_factors["fingers"]
            
            return angles
            
        except Exception as e:
            self.logger.error(f"위치를 각도로 변환 실패: {e}")
            return np.zeros(len(positions))
    
    def _apply_smoothing(self, joint_positions: np.ndarray) -> np.ndarray:
        """관절 위치에 스무딩 적용"""
        if not self.smoothing_config["enabled"]:
            return joint_positions
        
        # 스무딩 버퍼에 추가
        self.smoothing_buffer.append(joint_positions.copy())
        
        # 버퍼 크기 제한
        if len(self.smoothing_buffer) > self.smoothing_config["buffer_size"]:
            self.smoothing_buffer.pop(0)
        
        # 가중 평균으로 스무딩
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
        관절 제한 범위 적용
        
        Args:
            joint_positions: 관절 위치 배열
            joint_limits: 관절별 제한 범위
            
        Returns:
            제한 범위가 적용된 관절 위치
        """
        limited_positions = joint_positions.copy()
        
        try:
            # 각 관절 타입별로 제한 범위 적용
            for joint_type, limits in joint_limits.items():
                if joint_type == "wrist":
                    # 손목 관절 (WRJ0, WRJ1)
                    indices = [17, 18]
                    limited_positions[indices] = np.clip(
                        limited_positions[indices], 
                        limits[0], 
                        limits[1]
                    )
                
                elif joint_type == "thumb":
                    # 엄지 관절 (THJ0-4)
                    indices = [0, 1, 2, 3, 4]
                    limited_positions[indices] = np.clip(
                        limited_positions[indices], 
                        limits[0], 
                        limits[1]
                    )
                
                elif joint_type == "fingers":
                    # 손가락 관절 (FFJ, MFJ, RFJ, LFJ)
                    indices = list(range(5, 17))
                    limited_positions[indices] = np.clip(
                        limited_positions[indices], 
                        limits[0], 
                        limits[1]
                    )
            
            return limited_positions
            
        except Exception as e:
            self.logger.error(f"관절 제한 범위 적용 실패: {e}")
            return joint_positions
    
    def get_joint_mapping_info(self, vr_system: str = "steamvr") -> Dict:
        """관절 매핑 정보 반환"""
        if vr_system not in self.joint_mappings:
            return {}
        
        return self.joint_mappings[vr_system]
    
    def update_scale_factors(self, new_factors: Dict):
        """스케일 팩터 업데이트"""
        self.scale_factors.update(new_factors)
        self.logger.info(f"스케일 팩터 업데이트: {new_factors}")
    
    def update_smoothing_config(self, new_config: Dict):
        """스무딩 설정 업데이트"""
        self.smoothing_config.update(new_config)
        self.logger.info(f"스무딩 설정 업데이트: {new_config}")
    
    def reset_smoothing_buffer(self):
        """스무딩 버퍼 리셋"""
        self.smoothing_buffer.clear()
        self.logger.info("스무딩 버퍼 리셋 완료")
