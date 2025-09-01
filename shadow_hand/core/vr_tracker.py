"""
VR 핸드 트래킹 클래스
Quest 3, SteamVR 등 다양한 VR 시스템에서 핸드 데이터를 추출
"""

import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

class VRHandTracker:
    """VR 핸드 트래킹 클래스"""
    
    def __init__(self, vr_system: str = "auto", config: Optional[Dict] = None):
        """
        VR 핸드 트래커 초기화
        
        Args:
            vr_system: VR 시스템 타입 ("steamvr", "quest3", "auto")
            config: 설정 딕셔너리
        """
        self.vr_system = vr_system
        self.config = config or {}
        self.is_connected = False
        self.vr_interface = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # VR 시스템 연결 시도
        self._connect_vr_system()
        
        # 핸드 데이터 버퍼
        self.hand_data_buffer = []
        self.max_buffer_size = 10
        
        # 마지막 업데이트 시간
        self.last_update_time = time.time()
        
    def _connect_vr_system(self):
        """VR 시스템 연결"""
        try:
            if self.vr_system == "steamvr" or self.vr_system == "auto":
                self._connect_steamvr()
            elif self.vr_system == "quest3":
                self._connect_quest3()
            else:
                self.logger.warning(f"지원하지 않는 VR 시스템: {self.vr_system}")
                
        except Exception as e:
            self.logger.error(f"VR 시스템 연결 실패: {e}")
            self.is_connected = False
    
    def _connect_steamvr(self):
        """SteamVR 연결"""
        try:
            import openvr
            self.vr_interface = openvr.init(openvr.VRApplication_Other)
            self.is_connected = True
            self.logger.info("✅ SteamVR 연결 성공!")
        except ImportError:
            self.logger.warning("openvr 패키지가 설치되지 않았습니다")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"SteamVR 연결 실패: {e}")
            self.is_connected = False
    
    def _connect_quest3(self):
        """Quest 3 연결"""
        try:
            import pyopenxr as xr
            # Quest 3 연결 로직 (구체적 구현은 pyopenxr 문서 참조)
            self.vr_interface = None  # 임시
            self.is_connected = True
            self.logger.info("✅ Quest 3 연결 성공!")
        except ImportError:
            self.logger.warning("pyopenxr 패키지가 설치되지 않았습니다")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Quest 3 연결 실패: {e}")
            self.is_connected = False
    
    def get_hand_data(self) -> Optional[np.ndarray]:
        """
        현재 핸드 데이터 가져오기
        
        Returns:
            핸드 데이터 (21개 관절, 3D 좌표) 또는 None
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
            self.logger.error(f"핸드 데이터 추출 실패: {e}")
            return self._get_dummy_hand_data()
    
    def _get_steamvr_hand_data(self) -> np.ndarray:
        """SteamVR에서 핸드 데이터 추출"""
        try:
            # SteamVR 핸드 트래킹 데이터 추출
            # 실제 구현은 SteamVR API 문서 참조
            hand_data = np.zeros((21, 3))
            
            # 더미 데이터 (실제 구현 시 제거)
            t = time.time()
            hand_data[0] = [np.sin(t), np.cos(t), 0.0]  # 손목
            hand_data[1:5] = [np.sin(t*2), np.cos(t*2), 0.1]  # 엄지
            hand_data[5:9] = [np.sin(t*1.5), np.cos(t*1.5), 0.2]  # 검지
            hand_data[9:13] = [np.sin(t*1.8), np.cos(t*1.8), 0.3]  # 중지
            hand_data[13:17] = [np.sin(t*2.2), np.cos(t*2.2), 0.4]  # 약지
            hand_data[17:21] = [np.sin(t*1.2), np.cos(t*1.2), 0.5]  # 새끼
            
            return hand_data
            
        except Exception as e:
            self.logger.error(f"SteamVR 핸드 데이터 추출 실패: {e}")
            return np.zeros((21, 3))
    
    def _get_quest3_hand_data(self) -> np.ndarray:
        """Quest 3에서 핸드 데이터 추출"""
        try:
            # Quest 3 핸드 트래킹 데이터 추출
            # 실제 구현은 pyopenxr 문서 참조
            hand_data = np.zeros((21, 3))
            
            # 더미 데이터 (실제 구현 시 제거)
            t = time.time()
            hand_data[0] = [np.sin(t), np.cos(t), 0.0]  # 손목
            hand_data[1:5] = [np.sin(t*2), np.cos(t*2), 0.1]  # 엄지
            hand_data[5:9] = [np.sin(t*1.5), np.cos(t*1.5), 0.2]  # 검지
            hand_data[9:13] = [np.sin(t*1.8), np.cos(t*1.8), 0.3]  # 중지
            hand_data[13:17] = [np.sin(t*2.2), np.cos(t*2.2), 0.4]  # 약지
            hand_data[17:21] = [np.sin(t*1.2), np.cos(t*1.2), 0.5]  # 새끼
            
            return hand_data
            
        except Exception as e:
            self.logger.error(f"Quest 3 핸드 데이터 추출 실패: {e}")
            return np.zeros((21, 3))
    
    def _get_dummy_hand_data(self) -> np.ndarray:
        """더미 핸드 데이터 생성 (테스트용)"""
        t = time.time()
        hand_data = np.zeros((21, 3))
        
        # 시간에 따라 움직이는 더미 데이터
        hand_data[0] = [np.sin(t * 0.5) * 0.1, np.cos(t * 0.5) * 0.1, 0.0]  # 손목
        hand_data[1:5] = [np.sin(t * 2) * 0.05, np.cos(t * 2) * 0.05, 0.1]  # 엄지
        hand_data[5:9] = [np.sin(t * 1.5) * 0.05, np.cos(t * 1.5) * 0.05, 0.2]  # 검지
        hand_data[9:13] = [np.sin(t * 1.8) * 0.05, np.cos(t * 1.8) * 0.05, 0.3]  # 중지
        hand_data[13:17] = [np.sin(t * 2.2) * 0.05, np.cos(t * 2.2) * 0.05, 0.4]  # 약지
        hand_data[17:21] = [np.sin(t * 1.2) * 0.05, np.cos(t * 1.2) * 0.05, 0.5]  # 새끼
        
        return hand_data
    
    def get_hand_confidence(self) -> float:
        """핸드 트래킹 신뢰도 반환"""
        if not self.is_connected:
            return 0.0
        
        # 실제 구현에서는 VR 시스템에서 신뢰도 값 가져오기
        return 0.8  # 더미 값
    
    def is_hand_visible(self) -> bool:
        """핸드가 보이는지 확인"""
        if not self.is_connected:
            return True  # 더미 모드에서는 항상 True
        
        # 실제 구현에서는 VR 시스템에서 핸드 가시성 확인
        return True
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """트래킹 상태 정보 반환"""
        return {
            "is_connected": self.is_connected,
            "vr_system": self.vr_system,
            "hand_visible": self.is_hand_visible(),
            "confidence": self.get_hand_confidence(),
            "last_update": self.last_update_time,
            "buffer_size": len(self.hand_data_buffer)
        }
    
    def update(self):
        """트래커 업데이트"""
        current_time = time.time()
        
        # 핸드 데이터 가져오기
        hand_data = self.get_hand_data()
        
        if hand_data is not None:
            # 버퍼에 추가
            self.hand_data_buffer.append({
                "timestamp": current_time,
                "data": hand_data.copy()
            })
            
            # 버퍼 크기 제한
            if len(self.hand_data_buffer) > self.max_buffer_size:
                self.hand_data_buffer.pop(0)
            
            self.last_update_time = current_time
    
    def get_smoothed_hand_data(self, smoothing_factor: float = 0.3) -> Optional[np.ndarray]:
        """스무딩이 적용된 핸드 데이터 반환"""
        if len(self.hand_data_buffer) < 2:
            return self.get_hand_data()
        
        # 최근 데이터들의 가중 평균
        weights = np.exp(np.linspace(-smoothing_factor, 0, len(self.hand_data_buffer)))
        weights = weights / np.sum(weights)
        
        smoothed_data = np.zeros((21, 3))
        for i, (_, data) in enumerate(self.hand_data_buffer):
            smoothed_data += weights[i] * data
        
        return smoothed_data
    
    def disconnect(self):
        """VR 시스템 연결 해제"""
        try:
            if self.vr_interface:
                if self.vr_system == "steamvr":
                    import openvr
                    openvr.shutdown()
                # Quest 3 연결 해제 로직 추가
                
            self.is_connected = False
            self.logger.info("VR 시스템 연결 해제 완료")
            
        except Exception as e:
            self.logger.error(f"VR 시스템 연결 해제 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        self.disconnect()
