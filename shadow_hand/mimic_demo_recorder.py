"""
Isaac Lab Mimic 호환 데모 녹화 시스템
VR 텔레오퍼레이션을 사용하여 Isaac Lab Mimic 형식의 데모 데이터를 수집
"""

import os
import time
import json
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# 기존 VR 시스템 import
from core.vr_tracker import VRHandTracker
from core.hand_controller import ShadowHandController
from core.motion_mapper import MotionMapper

@dataclass
class MimicDemoMetadata:
    """Isaac Lab Mimic 데모 메타데이터"""
    task_name: str
    demo_id: int
    timestamp: str
    duration: float
    total_steps: int
    success: bool
    vr_system: str
    hand_tracking_confidence: float
    subtask_annotations: Optional[Dict[str, Any]] = None

@dataclass
class SubTaskAnnotation:
    """서브태스크 주석"""
    name: str
    start_step: int
    end_step: int
    object_ref: Optional[str] = None
    success: bool = True

class MimicDemoRecorder:
    """Isaac Lab Mimic 호환 데모 녹화기"""
    
    def __init__(self, config_path: str = None):
        """
        데모 녹화기 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # VR 시스템 초기화
        self.vr_tracker = None
        self.hand_controller = None
        self.motion_mapper = None
        
        # 녹화 상태
        self.is_recording = False
        self.current_demo = None
        self.demo_data = []
        self.subtask_annotations = []
        
        # Isaac Lab Mimic 설정
        self.mimic_config = self.config.get("mimic", {})
        self.save_path = Path(self.mimic_config.get("save_path", "./mimic_demos/"))
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self._initialize_components()
    
    def _load_config(self, config_path: str = None) -> dict:
        """설정 파일 로드"""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "teleop_config.yaml"
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.warning(f"설정 파일 로드 실패: {e}, 기본 설정 사용")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            "vr_tracker": {
                "system_type": "auto",
                "update_rate": 60,
                "hand_tracking": {
                    "enable": True,
                    "confidence_threshold": 0.7,
                    "smoothing_factor": 0.3
                }
            },
            "shadow_hand": {
                "environment_name": "Isaac-Repose-Cube-Shadow-Direct-v0",
                "joint_limits": {
                    "wrist": [-0.5, 0.5],
                    "thumb": [-0.8, 0.8],
                    "fingers": [-1.57, 1.57]
                }
            },
            "mimic": {
                "save_path": "./mimic_demos/",
                "default_duration": 10.0,
                "subtask_annotation": True,
                "success_criteria": {
                    "min_confidence": 0.7,
                    "min_duration": 2.0
                }
            }
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """VR 컴포넌트 초기화"""
        try:
            # VR 핸드 트래커 초기화
            vr_config = self.config.get("vr_tracker", {})
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
            
            self.logger.info("✅ Isaac Lab Mimic 데모 녹화기 초기화 완료!")
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def start_demo_collection(self, task_name: str, num_demos: int = 5, 
                            duration: float = None, enable_subtask_annotation: bool = True):
        """
        데모 수집 시작
        
        Args:
            task_name: 태스크 이름
            num_demos: 수집할 데모 수
            duration: 각 데모의 길이 (초)
            enable_subtask_annotation: 서브태스크 주석 달기 활성화
        """
        if duration is None:
            duration = self.mimic_config.get("default_duration", 10.0)
        
        self.logger.info(f"🎬 Isaac Lab Mimic 데모 수집 시작")
        self.logger.info(f"📋 태스크: {task_name}")
        self.logger.info(f"📊 목표 데모 수: {num_demos}")
        self.logger.info(f"⏱️ 데모 길이: {duration}초")
        
        for demo_idx in range(num_demos):
            self.logger.info(f"\n=== 데모 {demo_idx + 1}/{num_demos} ===")
            
            # 단일 데모 수집
            demo_metadata = self._collect_single_demo(
                task_name=task_name,
                demo_id=demo_idx,
                duration=duration,
                enable_subtask_annotation=enable_subtask_annotation
            )
            
            if demo_metadata.success:
                self.logger.info(f"✅ 데모 {demo_idx + 1} 수집 성공!")
            else:
                self.logger.warning(f"⚠️ 데모 {demo_idx + 1} 수집 실패")
        
        # 모든 데모 저장
        self._save_mimic_dataset(task_name)
        
        self.logger.info(f"🎉 데모 수집 완료! 총 {len(self.demo_data)}개 데모")
    
    def _collect_single_demo(self, task_name: str, demo_id: int, 
                           duration: float, enable_subtask_annotation: bool) -> MimicDemoMetadata:
        """단일 데모 수집"""
        start_time = time.time()
        step_count = 0
        max_steps = int(duration * 60)  # 60fps 가정
        
        # 데모 데이터 초기화
        demo_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
            "timestamps": [],
            "hand_data": [],
            "joint_positions": []
        }
        
        # 서브태스크 주석 초기화
        current_subtask = None
        subtask_start_step = 0
        
        self.logger.info(f"데모 {demo_id + 1} 녹화 시작...")
        self.logger.info("VR에서 손을 움직여주세요...")
        self.logger.info("키보드 단축키:")
        self.logger.info("  'S' - 서브태스크 시작/종료")
        self.logger.info("  'Q' - 데모 종료")
        
        try:
            while step_count < max_steps:
                frame_start = time.time()
                
                # VR 핸드 데이터 가져오기
                hand_data = self.vr_tracker.get_hand_data()
                
                if hand_data is not None:
                    # VR 데이터를 Shadow Hand 관절 위치로 변환
                    joint_positions = self.motion_mapper.map_vr_to_shadow_hand(
                        hand_data, 
                        self.vr_tracker.vr_system
                    )
                    
                    # Isaac Lab Mimic 형식의 액션 생성
                    action = self._create_mimic_action(joint_positions, hand_data)
                    
                    # 관찰 데이터 생성 (환경에서 가져오기)
                    observation = self._get_environment_observation()
                    
                    # 보상 및 정보 생성
                    reward, done, info = self._evaluate_step(observation, action, hand_data)
                    
                    # 데이터 저장
                    demo_data["observations"].append(observation.copy())
                    demo_data["actions"].append(action.copy())
                    demo_data["rewards"].append(reward)
                    demo_data["dones"].append(done)
                    demo_data["infos"].append(info.copy())
                    demo_data["timestamps"].append(time.time() - start_time)
                    demo_data["hand_data"].append(hand_data.copy())
                    demo_data["joint_positions"].append(joint_positions.copy())
                    
                    # Shadow Hand 업데이트
                    if self.hand_controller and self.hand_controller.is_initialized:
                        self.hand_controller.update_hand(joint_positions)
                        self.hand_controller.step_environment()
                    
                    step_count += 1
                    
                    # 서브태스크 주석 처리
                    if enable_subtask_annotation:
                        self._handle_subtask_annotation(
                            step_count, current_subtask, subtask_start_step
                        )
                    
                    # 진행률 출력
                    if step_count % 60 == 0:  # 1초마다
                        progress = (time.time() - start_time) / duration * 100
                        self.logger.info(f"진행률: {progress:.1f}% ({step_count}/{max_steps})")
                
                # VR 트래커 업데이트
                self.vr_tracker.update()
                
                # 60fps 유지
                elapsed = time.time() - frame_start
                if elapsed < 1.0/60:
                    time.sleep(1.0/60 - elapsed)
            
            # 데모 완료
            actual_duration = time.time() - start_time
            success = self._evaluate_demo_success(demo_data)
            
            # 메타데이터 생성
            metadata = MimicDemoMetadata(
                task_name=task_name,
                demo_id=demo_id,
                timestamp=datetime.now().isoformat(),
                duration=actual_duration,
                total_steps=step_count,
                success=success,
                vr_system=self.vr_tracker.vr_system,
                hand_tracking_confidence=self._calculate_average_confidence(demo_data),
                subtask_annotations=self.subtask_annotations.copy() if enable_subtask_annotation else None
            )
            
            # 데모 데이터 저장
            self.demo_data.append({
                "metadata": metadata,
                "data": demo_data
            })
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"데모 수집 중 에러: {e}")
            return MimicDemoMetadata(
                task_name=task_name,
                demo_id=demo_id,
                timestamp=datetime.now().isoformat(),
                duration=time.time() - start_time,
                total_steps=step_count,
                success=False,
                vr_system=self.vr_tracker.vr_system,
                hand_tracking_confidence=0.0
            )
    
    def _create_mimic_action(self, joint_positions: np.ndarray, hand_data: np.ndarray) -> np.ndarray:
        """Isaac Lab Mimic 형식의 액션 생성"""
        # Shadow Hand 관절 위치를 액션으로 변환
        # 실제 구현에서는 환경의 액션 스페이스에 맞춰 조정 필요
        action = joint_positions.copy()
        
        # 추가 정보 포함 (그리퍼 상태 등)
        if len(action) < 24:  # Shadow Hand는 20개 관절 + 추가 정보
            # 그리퍼 상태 추가
            gripper_state = self._extract_gripper_state(hand_data)
            action = np.concatenate([action, gripper_state])
        
        return action
    
    def _extract_gripper_state(self, hand_data: np.ndarray) -> np.ndarray:
        """핸드 데이터에서 그리퍼 상태 추출"""
        # 엄지와 검지 사이의 거리를 기반으로 그리퍼 상태 계산
        thumb_tip = hand_data[4]  # 엄지 끝
        index_tip = hand_data[8]  # 검지 끝
        
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        # 거리를 0-1 범위의 그리퍼 상태로 변환
        gripper_state = np.clip(1.0 - distance / 0.1, 0.0, 1.0)  # 10cm 기준
        
        return np.array([gripper_state])
    
    def _get_environment_observation(self) -> np.ndarray:
        """환경에서 관찰 데이터 가져오기"""
        if self.hand_controller and self.hand_controller.is_initialized:
            # Shadow Hand의 현재 상태를 관찰로 사용
            joint_positions = self.hand_controller.get_joint_positions()
            
            # 추가 관찰 정보 (손목 위치, 오브젝트 위치 등)
            # 실제 구현에서는 환경의 관찰 스페이스에 맞춰 조정 필요
            observation = joint_positions.copy()
            
            return observation
        else:
            # 더미 관찰 데이터
            return np.zeros(20)
    
    def _evaluate_step(self, observation: np.ndarray, action: np.ndarray, 
                      hand_data: np.ndarray) -> Tuple[float, bool, Dict]:
        """스텝 평가 (보상, 종료 조건, 정보)"""
        reward = 0.0
        done = False
        info = {}
        
        # 손 추적 신뢰도 기반 보상
        confidence = self._calculate_hand_confidence(hand_data)
        reward += confidence * 0.1
        
        # 관절 제한 위반 페널티
        if self._check_joint_limits_violation(action):
            reward -= 0.5
            info["joint_limit_violation"] = True
        
        # 안정성 보상 (관절 속도가 너무 크지 않음)
        if hasattr(self, '_prev_action'):
            joint_velocity = np.linalg.norm(action - self._prev_action)
            if joint_velocity < 0.1:  # 안정적인 움직임
                reward += 0.05
        
        self._prev_action = action.copy()
        
        return reward, done, info
    
    def _calculate_hand_confidence(self, hand_data: np.ndarray) -> float:
        """핸드 추적 신뢰도 계산"""
        # 간단한 신뢰도 계산 (실제로는 VR 시스템에서 제공)
        # 손가락 끝점들의 일관성을 기반으로 계산
        finger_tips = [hand_data[4], hand_data[8], hand_data[12], hand_data[16], hand_data[20]]
        
        # 손가락 끝점들이 합리적인 거리에 있는지 확인
        palm_center = np.mean([hand_data[0], hand_data[5], hand_data[9], hand_data[13], hand_data[17]], axis=0)
        
        distances = [np.linalg.norm(tip - palm_center) for tip in finger_tips]
        avg_distance = np.mean(distances)
        
        # 거리가 합리적인 범위(5-15cm)에 있으면 높은 신뢰도
        if 0.05 <= avg_distance <= 0.15:
            confidence = 1.0
        else:
            confidence = max(0.0, 1.0 - abs(avg_distance - 0.1) / 0.1)
        
        return confidence
    
    def _check_joint_limits_violation(self, action: np.ndarray) -> bool:
        """관절 제한 위반 확인"""
        joint_limits = self.hand_controller.get_joint_limits()
        
        # 손목 관절 확인
        wrist_limits = joint_limits["wrist"]
        if np.any(action[17:19] < wrist_limits[0]) or np.any(action[17:19] > wrist_limits[1]):
            return True
        
        # 엄지 관절 확인
        thumb_limits = joint_limits["thumb"]
        if np.any(action[0:5] < thumb_limits[0]) or np.any(action[0:5] > thumb_limits[1]):
            return True
        
        # 손가락 관절 확인
        finger_limits = joint_limits["fingers"]
        if np.any(action[5:17] < finger_limits[0]) or np.any(action[5:17] > finger_limits[1]):
            return True
        
        return False
    
    def _evaluate_demo_success(self, demo_data: Dict) -> bool:
        """데모 성공 여부 평가"""
        success_criteria = self.mimic_config.get("success_criteria", {})
        
        # 최소 신뢰도 확인
        min_confidence = success_criteria.get("min_confidence", 0.7)
        avg_confidence = self._calculate_average_confidence(demo_data)
        if avg_confidence < min_confidence:
            return False
        
        # 최소 지속 시간 확인
        min_duration = success_criteria.get("min_duration", 2.0)
        if demo_data["timestamps"][-1] < min_duration:
            return False
        
        # 관절 제한 위반 확인
        for action in demo_data["actions"]:
            if self._check_joint_limits_violation(action):
                return False
        
        return True
    
    def _calculate_average_confidence(self, demo_data: Dict) -> float:
        """데모의 평균 신뢰도 계산"""
        confidences = []
        for hand_data in demo_data["hand_data"]:
            confidence = self._calculate_hand_confidence(hand_data)
            confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _handle_subtask_annotation(self, step_count: int, current_subtask: Optional[str], 
                                 subtask_start_step: int):
        """서브태스크 주석 처리"""
        # 키보드 입력 확인 (실제 구현에서는 키보드 이벤트 처리)
        # 현재는 더미 구현
        pass
    
    def _save_mimic_dataset(self, task_name: str):
        """Isaac Lab Mimic 형식으로 데이터셋 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = self.save_path / f"{task_name}_mimic_dataset_{timestamp}.h5"
        
        with h5py.File(dataset_path, 'w') as f:
            # 메타데이터 저장
            f.attrs['task_name'] = task_name
            f.attrs['num_demos'] = len(self.demo_data)
            f.attrs['timestamp'] = timestamp
            f.attrs['vr_system'] = self.vr_tracker.vr_system
            
            # 각 데모 저장
            for i, demo in enumerate(self.demo_data):
                demo_group = f.create_group(f'demo_{i}')
                
                # 메타데이터 저장
                metadata = demo["metadata"]
                demo_group.attrs['demo_id'] = metadata.demo_id
                demo_group.attrs['timestamp'] = metadata.timestamp
                demo_group.attrs['duration'] = metadata.duration
                demo_group.attrs['total_steps'] = metadata.total_steps
                demo_group.attrs['success'] = metadata.success
                demo_group.attrs['vr_system'] = metadata.vr_system
                demo_group.attrs['hand_tracking_confidence'] = metadata.hand_tracking_confidence
                
                # 데이터 저장
                data = demo["data"]
                demo_group.create_dataset('observations', data=np.array(data['observations']))
                demo_group.create_dataset('actions', data=np.array(data['actions']))
                demo_group.create_dataset('rewards', data=np.array(data['rewards']))
                demo_group.create_dataset('dones', data=np.array(data['dones']))
                demo_group.create_dataset('timestamps', data=np.array(data['timestamps']))
                demo_group.create_dataset('hand_data', data=np.array(data['hand_data']))
                demo_group.create_dataset('joint_positions', data=np.array(data['joint_positions']))
                
                # 서브태스크 주석 저장
                if metadata.subtask_annotations:
                    subtask_group = demo_group.create_group('subtasks')
                    for j, subtask in enumerate(metadata.subtask_annotations):
                        subtask_group.attrs[f'subtask_{j}_name'] = subtask.name
                        subtask_group.attrs[f'subtask_{j}_start'] = subtask.start_step
                        subtask_group.attrs[f'subtask_{j}_end'] = subtask.end_step
                        if subtask.object_ref:
                            subtask_group.attrs[f'subtask_{j}_object_ref'] = subtask.object_ref
        
        # JSON 메타데이터도 저장
        metadata_path = self.save_path / f"{task_name}_metadata_{timestamp}.json"
        metadata_dict = {
            "task_name": task_name,
            "num_demos": len(self.demo_data),
            "timestamp": timestamp,
            "vr_system": self.vr_tracker.vr_system,
            "demos": [asdict(demo["metadata"]) for demo in self.demo_data]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Isaac Lab Mimic 데이터셋 저장 완료: {dataset_path}")
        self.logger.info(f"📄 메타데이터 저장 완료: {metadata_path}")
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.vr_tracker:
                self.vr_tracker.disconnect()
            
            if self.hand_controller:
                self.hand_controller.close()
            
            self.logger.info("✅ Isaac Lab Mimic 데모 녹화기 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 중 에러: {e}")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Isaac Lab Mimic 데모 수집")
    parser.add_argument("--task", type=str, required=True, help="태스크 이름")
    parser.add_argument("--num-demos", type=int, default=5, help="수집할 데모 수")
    parser.add_argument("--duration", type=float, default=10.0, help="각 데모의 길이 (초)")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--no-subtask", action="store_true", help="서브태스크 주석 비활성화")
    
    args = parser.parse_args()
    
    try:
        # 데모 녹화기 생성
        recorder = MimicDemoRecorder(config_path=args.config)
        
        # 데모 수집 시작
        recorder.start_demo_collection(
            task_name=args.task,
            num_demos=args.num_demos,
            duration=args.duration,
            enable_subtask_annotation=not args.no_subtask
        )
        
    except Exception as e:
        logging.error(f"데모 수집 실패: {e}")
        return 1
    
    finally:
        recorder.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())
