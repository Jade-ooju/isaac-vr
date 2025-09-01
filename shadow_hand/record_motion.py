#!/usr/bin/env python3
"""
Shadow Hand 모션 녹화 실행 파일
VR 핸드 움직임을 5초간 녹화하여 저장
"""

import time
import sys
import os
import yaml
import logging
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shadow_hand.core.vr_tracker import VRHandTracker
from shadow_hand.core.hand_controller import ShadowHandController
from shadow_hand.core.motion_mapper import MotionMapper

class MotionRecorder:
    """모션 녹화 클래스"""
    
    def __init__(self, config_path: str = None):
        """
        모션 녹화기 초기화
        
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
        
        # 녹화 설정
        self.recording_config = self.config.get("recording", {})
        self.default_duration = self.recording_config.get("default_duration", 5.0)
        self.save_format = self.recording_config.get("save_format", "h5")
        self.save_path = Path(self.recording_config.get("save_path", "./recorded_motions/"))
        
        # 저장 경로 생성
        self.save_path.mkdir(parents=True, exist_ok=True)
        
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
                "update_rate": 60
            },
            "recording": {
                "default_duration": 5.0,
                "save_format": "h5",
                "save_path": "./recorded_motions/"
            }
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            
            # Shadow Hand 컨트롤러 초기화 (녹화만 할 경우 선택사항)
            try:
                shadow_config = self.config.get("shadow_hand", {})
                self.hand_controller = ShadowHandController(
                    environment_name=shadow_config.get("environment_name", "Isaac-Repose-Cube-Shadow-Direct-v0")
                )
                self.logger.info("✅ Shadow Hand 컨트롤러 초기화 완료")
            except Exception as e:
                self.logger.warning(f"Shadow Hand 컨트롤러 초기화 실패: {e}")
                self.hand_controller = None
            
            # 모션 매퍼 초기화
            self.motion_mapper = MotionMapper(config=self.config)
            
            self.logger.info("✅ 모션 녹화기 초기화 완료!")
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def record_motion(self, duration: float = None, gesture_name: str = None) -> dict:
        """
        모션 녹화 실행
        
        Args:
            duration: 녹화 시간 (초)
            gesture_name: 제스처 이름
            
        Returns:
            녹화된 모션 데이터
        """
        if duration is None:
            duration = self.default_duration
        
        if gesture_name is None:
            gesture_name = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🎬 모션 녹화 시작: {gesture_name} ({duration}초)")
        
        # 녹화 데이터 저장용
        motion_data = {
            "gesture_name": gesture_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "vr_system": self.vr_tracker.vr_system,
            "frames": [],
            "joint_positions": [],
            "hand_data": []
        }
        
        # 녹화 시작
        start_time = time.time()
        frame_count = 0
        target_frames = int(duration * 60)  # 60fps 가정
        
        self.logger.info(f"목표 프레임 수: {target_frames}")
        self.logger.info("VR에서 손을 움직여주세요...")
        
        try:
            while time.time() - start_time < duration:
                frame_start = time.time()
                
                # VR 핸드 데이터 가져오기
                hand_data = self.vr_tracker.get_hand_data()
                
                if hand_data is not None:
                    # VR 데이터를 Shadow Hand 관절 위치로 변환
                    joint_positions = self.motion_mapper.map_vr_to_shadow_hand(
                        hand_data, 
                        self.vr_tracker.vr_system
                    )
                    
                    # 데이터 저장
                    frame_data = {
                        "timestamp": time.time() - start_time,
                        "hand_data": hand_data.copy(),
                        "joint_positions": joint_positions.copy()
                    }
                    
                    motion_data["frames"].append(frame_data)
                    motion_data["joint_positions"].append(joint_positions.copy())
                    motion_data["hand_data"].append(hand_data.copy())
                    
                    frame_count += 1
                    
                    # Shadow Hand 업데이트 (컨트롤러가 있는 경우)
                    if self.hand_controller and self.hand_controller.is_initialized:
                        self.hand_controller.update_hand(joint_positions)
                        self.hand_controller.step_environment()
                    
                    # 진행률 출력
                    if frame_count % 30 == 0:  # 0.5초마다
                        progress = (time.time() - start_time) / duration * 100
                        self.logger.info(f"진행률: {progress:.1f}% ({frame_count}/{target_frames})")
                
                # VR 트래커 업데이트
                self.vr_tracker.update()
                
                # 60fps 유지
                elapsed = time.time() - frame_start
                if elapsed < 1.0/60:
                    time.sleep(1.0/60 - elapsed)
            
            self.logger.info(f"✅ 모션 녹화 완료: {frame_count} 프레임")
            
            # 메타데이터 추가
            motion_data["total_frames"] = frame_count
            motion_data["actual_duration"] = time.time() - start_time
            motion_data["average_fps"] = frame_count / motion_data["actual_duration"]
            
            return motion_data
            
        except Exception as e:
            self.logger.error(f"모션 녹화 중 에러: {e}")
            return motion_data
    
    def save_motion(self, motion_data: dict, filename: str = None) -> str:
        """
        모션 데이터 저장
        
        Args:
            motion_data: 저장할 모션 데이터
            filename: 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gesture_name = motion_data.get("gesture_name", "motion")
            filename = f"{gesture_name}_{timestamp}"
        
        try:
            if self.save_format == "h5":
                filepath = self.save_path / f"{filename}.h5"
                self._save_h5(motion_data, filepath)
            elif self.save_format == "npz":
                filepath = self.save_path / f"{filename}.npz"
                self._save_npz(motion_data, filepath)
            elif self.save_format == "json":
                filepath = self.save_path / f"{filename}.json"
                self._save_json(motion_data, filepath)
            else:
                raise ValueError(f"지원하지 않는 저장 형식: {self.save_format}")
            
            self.logger.info(f"✅ 모션 데이터 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"모션 데이터 저장 실패: {e}")
            raise
    
    def _save_h5(self, motion_data: dict, filepath: Path):
        """HDF5 형식으로 저장"""
        with h5py.File(filepath, 'w') as f:
            # 메타데이터 저장
            f.attrs['gesture_name'] = motion_data['gesture_name']
            f.attrs['duration'] = motion_data['duration']
            f.attrs['timestamp'] = motion_data['timestamp']
            f.attrs['vr_system'] = motion_data['vr_system']
            f.attrs['total_frames'] = motion_data['total_frames']
            f.attrs['average_fps'] = motion_data['average_fps']
            
            # 프레임별 데이터 저장
            frames_group = f.create_group('frames')
            for i, frame in enumerate(motion_data['frames']):
                frame_group = frames_group.create_group(f'frame_{i:04d}')
                frame_group.attrs['timestamp'] = frame['timestamp']
                frame_group.create_dataset('hand_data', data=frame['hand_data'])
                frame_group.create_dataset('joint_positions', data=frame['joint_positions'])
            
            # 전체 배열 데이터 저장
            f.create_dataset('joint_positions_array', data=np.array(motion_data['joint_positions']))
            f.create_dataset('hand_data_array', data=np.array(motion_data['hand_data']))
    
    def _save_npz(self, motion_data: dict, filepath: Path):
        """NPZ 형식으로 저장"""
        # NumPy 배열로 변환
        joint_positions_array = np.array(motion_data['joint_positions'])
        hand_data_array = np.array(motion_data['hand_data'])
        
        # 메타데이터는 별도 파일로 저장
        metadata = {
            'gesture_name': motion_data['gesture_name'],
            'duration': motion_data['duration'],
            'timestamp': motion_data['timestamp'],
            'vr_system': motion_data['vr_system'],
            'total_frames': motion_data['total_frames'],
            'average_fps': motion_data['average_fps']
        }
        
        np.savez_compressed(
            filepath,
            joint_positions=joint_positions_array,
            hand_data=hand_data_array,
            metadata=metadata
        )
    
    def _save_json(self, motion_data: dict, filepath: Path):
        """JSON 형식으로 저장"""
        import json
        
        # NumPy 배열을 리스트로 변환
        json_data = motion_data.copy()
        json_data['joint_positions'] = [pos.tolist() for pos in motion_data['joint_positions']]
        json_data['hand_data'] = [data.tolist() for data in motion_data['hand_data']]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def cleanup(self):
        """리소스 정리"""
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
    
    parser = argparse.ArgumentParser(description="Shadow Hand 모션 녹화")
    parser.add_argument(
        "--duration", 
        type=float, 
        default=5.0,
        help="녹화 시간 (초)"
    )
    parser.add_argument(
        "--gesture", 
        type=str, 
        help="제스처 이름"
    )
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
        # 모션 녹화기 생성
        recorder = MotionRecorder(config_path=args.config)
        
        # VR 시스템 타입 설정
        if args.vr_system != "auto":
            recorder.vr_tracker.vr_system = args.vr_system
        
        # 모션 녹화 실행
        motion_data = recorder.record_motion(
            duration=args.duration,
            gesture_name=args.gesture
        )
        
        # 저장
        if motion_data["total_frames"] > 0:
            filepath = recorder.save_motion(motion_data)
            print(f"\n🎉 모션 녹화 완료!")
            print(f"📁 저장 위치: {filepath}")
            print(f"📊 총 프레임: {motion_data['total_frames']}")
            print(f"⏱️ 실제 시간: {motion_data['actual_duration']:.2f}초")
            print(f"🎯 평균 FPS: {motion_data['average_fps']:.1f}")
        else:
            print("❌ 녹화된 프레임이 없습니다")
        
    except Exception as e:
        logging.error(f"모션 녹화 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
