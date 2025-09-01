#!/usr/bin/env python3
"""
Shadow Hand 텔레오퍼레이션 메인 실행 파일
VR 핸드 트래킹을 통해 Shadow Hand를 실시간으로 제어
"""

import time
import sys
import os
import yaml
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shadow_hand.core.vr_tracker import VRHandTracker
from shadow_hand.core.hand_controller import ShadowHandController
from shadow_hand.core.motion_mapper import MotionMapper
from shadow_hand.utils.coordinate_utils import CoordinateTransformer

class ShadowHandTeleop:
    """Shadow Hand 텔레오퍼레이션 메인 클래스"""
    
    def __init__(self, config_path: str = None):
        """
        텔레오퍼레이션 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 로깅 설정
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.vr_tracker = None
        self.hand_controller = None
        self.motion_mapper = None
        self.coordinate_transformer = None
        
        # 실행 상태
        self.is_running = False
        self.update_rate = self.config.get("vr", {}).get("update_rate", 60)
        self.update_interval = 1.0 / self.update_rate
        
        # 성능 모니터링
        self.frame_count = 0
        self.start_time = time.time()
        
        # 컴포넌트 초기화
        self._initialize_components()
    
    def _load_config(self, config_path: str = None) -> dict:
        """설정 파일 로드"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "teleop_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"설정 파일 로드 성공: {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"설정 파일 로드 실패: {e}, 기본 설정 사용")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            "vr": {
                "system_type": "auto",
                "update_rate": 60,
                "hand_tracking": {
                    "enable": True,
                    "confidence_threshold": 0.7,
                    "smoothing_factor": 0.3
                }
            },
            "shadow_hand": {
                "environment_name": "Isaac-Repose-Cube-Shadow-Direct-v0"
            },
            "debug": {
                "log_level": "INFO",
                "verbose": False
            }
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        log_level = self.config.get("debug", {}).get("log_level", "INFO")
        numeric_level = getattr(logging, log_level.upper(), None)
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('teleop.log')
            ]
        )
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # VR 핸드 트래커 초기화
            vr_config = self.config.get("vr", {})
            self.vr_tracker = VRHandTracker(
                vr_system=vr_config.get("system_type", "auto"),
                config=vr_config
            )
            
            # Shadow Hand 컨트롤러 초기화
            shadow_config = self.config.get("shadow_hand", {})
            self.hand_controller = ShadowHandController(
                environment_name=shadow_config.get("environment_name", "Isaac-Repose-Cube-Shadow-Direct-v0")
            )
            
            # 모션 매퍼 초기화
            self.motion_mapper = MotionMapper(config=self.config)
            
            # 좌표계 변환기 초기화
            self.coordinate_transformer = CoordinateTransformer()
            
            self.logger.info("✅ 모든 컴포넌트 초기화 완료!")
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def run(self):
        """메인 텔레오퍼레이션 루프 실행"""
        if not self._check_components():
            self.logger.error("컴포넌트 초기화가 완료되지 않았습니다")
            return
        
        self.logger.info("🚀 Shadow Hand 텔레오퍼레이션 시작!")
        self.logger.info(f"업데이트 주파수: {self.update_rate} Hz")
        self.logger.info("Ctrl+C로 중지할 수 있습니다")
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # VR 핸드 데이터 가져오기
                hand_data = self.vr_tracker.get_hand_data()
                
                if hand_data is not None:
                    # VR 데이터를 Shadow Hand 관절 위치로 변환
                    joint_positions = self.motion_mapper.map_vr_to_shadow_hand(
                        hand_data, 
                        self.vr_tracker.vr_system
                    )
                    
                    # 관절 제한 범위 적용
                    joint_limits = self.hand_controller.get_joint_limits()
                    joint_positions = self.motion_mapper.apply_joint_limits(
                        joint_positions, 
                        joint_limits
                    )
                    
                    # Shadow Hand 업데이트
                    success = self.hand_controller.update_hand(joint_positions)
                    
                    if success:
                        self.frame_count += 1
                        
                        # 성능 통계 출력 (1초마다)
                        if self.frame_count % self.update_rate == 0:
                            self._print_performance_stats()
                    
                    # 환경 스텝 실행
                    self.hand_controller.step_environment()
                
                # VR 트래커 업데이트
                self.vr_tracker.update()
                
                # 업데이트 속도 조절
                elapsed = time.time() - loop_start
                if elapsed < self.update_interval:
                    time.sleep(self.update_interval - elapsed)
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ 사용자에 의해 중지됨")
        except Exception as e:
            self.logger.error(f"텔레오퍼레이션 실행 중 에러: {e}")
        finally:
            self.cleanup()
    
    def _check_components(self) -> bool:
        """컴포넌트 상태 확인"""
        if self.vr_tracker is None:
            self.logger.error("VR 트래커가 초기화되지 않았습니다")
            return False
        
        if self.hand_controller is None or not self.hand_controller.is_initialized:
            self.logger.error("Shadow Hand 컨트롤러가 초기화되지 않았습니다")
            return False
        
        if self.motion_mapper is None:
            self.logger.error("모션 매퍼가 초기화되지 않았습니다")
            return False
        
        return True
    
    def _print_performance_stats(self):
        """성능 통계 출력"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(f"📊 성능 통계: FPS={fps:.1f}, 프레임={self.frame_count}, 시간={elapsed_time:.1f}s")
        
        # VR 트래킹 상태 출력
        tracking_status = self.vr_tracker.get_tracking_status()
        self.logger.info(f"VR 상태: 연결={tracking_status['is_connected']}, 신뢰도={tracking_status['confidence']:.2f}")
        
        # Shadow Hand 상태 출력
        current_joints = self.hand_controller.get_joint_positions()
        joint_range = f"[{current_joints.min():.3f}, {current_joints.max():.3f}]"
        self.logger.info(f"Shadow Hand: 관절 범위={joint_range}")
    
    def cleanup(self):
        """리소스 정리"""
        self.logger.info("🧹 리소스 정리 중...")
        
        self.is_running = False
        
        try:
            if self.vr_tracker:
                self.vr_tracker.disconnect()
            
            if self.hand_controller:
                self.hand_controller.close()
            
            self.logger.info("✅ 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 중 에러: {e}")
    
    def __del__(self):
        """소멸자"""
        self.cleanup()

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shadow Hand 텔레오퍼레이션")
    parser.add_argument(
        "--config", 
        type=str, 
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--vr-system", 
        type=str, 
        choices=["steamvr", "quest3", "auto"],
        default="auto",
        help="VR 시스템 타입"
    )
    
    args = parser.parse_args()
    
    try:
        # 텔레오퍼레이션 시스템 생성 및 실행
        teleop = ShadowHandTeleop(config_path=args.config)
        
        # VR 시스템 타입 설정
        if args.vr_system != "auto":
            teleop.vr_tracker.vr_system = args.vr_system
        
        # 실행
        teleop.run()
        
    except Exception as e:
        logging.error(f"프로그램 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
