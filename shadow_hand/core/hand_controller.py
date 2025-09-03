"""
Shadow Hand 제어 클래스
Isaac Sim 환경에서 Shadow Hand를 직접 제어
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
import logging
import gymnasium as gym

class ShadowHandController:
    """Shadow Hand 제어 클래스"""
    
    def __init__(self, environment_name: str = "Isaac-Repose-Cube-Shadow-Direct-v0"):
        """
        Shadow Hand 컨트롤러 초기화
        
        Args:
            environment_name: IsaacLab 환경 이름
        """
        self.environment_name = environment_name
        self.env = None
        self.robot = None
        self.is_initialized = False
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 관절 제한 범위 (라디안)
        self.joint_limits = {
            "wrist": [-0.5, 0.5],      # WRJ0, WRJ1
            "thumb": [-0.8, 0.8],      # THJ0-4
            "fingers": [-1.57, 1.57]   # FFJ, MFJ, RFJ, LFJ
        }
        
        # 현재 관절 상태
        self.current_joint_positions = np.zeros(20)
        self.target_joint_positions = np.zeros(20)
        
        # 제어 설정
        self.control_config = {
            "enable_ik": True,
            "damping_factor": 0.1,
            "stiffness_factor": 1.0,
            "max_velocity": 2.0,  # rad/s
            "position_tolerance": 0.01  # rad
        }
        
        # 환경 초기화
        self._initialize_environment()
    
    def _initialize_environment(self):
        """IsaacLab 환경 초기화"""
        try:
            self.logger.info(f"환경 초기화 중: {self.environment_name}")
            
            # 환경 생성 (Isaac Sim 없이도 실행 가능하도록)
            try:
                # IsaacLab 환경 등록 시도
                import isaaclab
                import isaaclab_tasks
                self.logger.info("✅ IsaacLab 환경 등록 완료!")
                
                # IsaacLab 환경인 경우
                if "Isaac-" in self.environment_name:
                    if "Shadow" in self.environment_name:
                        # Shadow Hand 환경
                        from isaaclab_tasks.direct.shadow_hand.shadow_hand_env import ShadowHandEnvCfg
                        self.env = gym.make(self.environment_name, cfg=ShadowHandEnvCfg())
                    elif "Cartpole" in self.environment_name:
                        # Cartpole 환경
                        from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg
                        self.env = gym.make(self.environment_name, cfg=CartpoleEnvCfg())
                    else:
                        # 기타 IsaacLab 환경
                        self.env = gym.make(self.environment_name)
                else:
                    # 일반 gymnasium 환경인 경우
                    self.env = gym.make(self.environment_name)
                self.logger.info("✅ IsaacLab 환경 생성 성공!")
                
            except ImportError as e:
                # Isaac Sim이 없는 경우 일반 gymnasium 환경 사용
                self.logger.warning(f"Isaac Sim을 찾을 수 없습니다: {e}")
                self.logger.info("일반 gymnasium 환경을 사용합니다.")
                self.env = gym.make(self.environment_name)
                self.logger.info("✅ 일반 gymnasium 환경 생성 성공!")
            
            # 환경 리셋하여 Isaac Sim 윈도우에 로드
            obs, info = self.env.reset()
            self.logger.info("✅ Isaac Sim 윈도우에 환경 로드 완료!")
            
            # Shadow Hand 객체 찾기
            self.robot = self._find_shadow_hand()
            
            if self.robot is not None:
                self.is_initialized = True
                self.logger.info("✅ Shadow Hand 객체 찾기 성공!")
                
                # 초기 관절 상태 가져오기
                self._update_current_state()
                
            else:
                self.logger.error("❌ Shadow Hand 객체를 찾을 수 없습니다")
                self.is_initialized = False
                
        except Exception as e:
            self.logger.error(f"환경 초기화 실패: {e}")
            self.is_initialized = False
    
    def _find_shadow_hand(self):
        """환경에서 Shadow Hand 객체 찾기"""
        if self.env is None:
            return None
        
        # 여러 방법으로 Shadow Hand 객체 찾기 시도
        robot_candidates = []
        
        # 1. 직접 robot 속성 확인
        if hasattr(self.env, 'robot'):
            robot_candidates.append(('env.robot', self.env.robot))
        
        # 2. _env 속성 확인
        if hasattr(self.env, '_env'):
            if hasattr(self.env._env, 'robot'):
                robot_candidates.append(('env._env.robot', self.env._env.robot))
        
        # 3. env 속성 확인
        if hasattr(self.env, 'env'):
            if hasattr(self.env.env, 'robot'):
                robot_candidates.append(('env.env.robot', self.env.env.robot))
        
        # 4. 환경 구조 출력 (디버깅용)
        if not robot_candidates:
            self.logger.info("환경 구조 분석:")
            self._print_environment_structure(self.env)
        
        # 첫 번째 후보 반환
        if robot_candidates:
            name, robot = robot_candidates[0]
            self.logger.info(f"Shadow Hand 객체 발견: {name}")
            return robot
        
        return None
    
    def _print_environment_structure(self, env, max_depth=3, current_depth=0):
        """환경 구조를 출력 (디버깅용)"""
        if current_depth >= max_depth:
            return
        
        indent = "  " * current_depth
        for attr in dir(env):
            if not attr.startswith('_'):
                try:
                    value = getattr(env, attr)
                    if hasattr(value, '__class__'):
                        self.logger.info(f"{indent}{attr}: {value.__class__.__name__}")
                    else:
                        self.logger.info(f"{indent}{attr}: {type(value)}")
                except:
                    pass
    
    def _update_current_state(self):
        """현재 관절 상태 업데이트"""
        if self.robot is None:
            return
        
        try:
            # 현재 관절 위치 가져오기
            if hasattr(self.robot, 'data') and hasattr(self.robot.data, 'joint_pos'):
                self.current_joint_positions = self.robot.data.joint_pos.clone().cpu().numpy()
            elif hasattr(self.robot, 'get_joint_positions'):
                self.current_joint_positions = self.robot.get_joint_positions()
            else:
                self.logger.warning("관절 위치를 가져올 수 없습니다")
                
        except Exception as e:
            self.logger.error(f"현재 상태 업데이트 실패: {e}")
    
    def update_hand(self, hand_data: np.ndarray, smoothing: bool = True) -> bool:
        """
        VR 핸드 데이터로 Shadow Hand 업데이트
        
        Args:
            hand_data: VR 핸드 데이터 (21개 관절, 3D 좌표)
            smoothing: 스무딩 적용 여부
            
        Returns:
            업데이트 성공 여부
        """
        if not self.is_initialized:
            self.logger.error("컨트롤러가 초기화되지 않았습니다")
            return False
        
        try:
            # VR 데이터를 Shadow Hand 관절 위치로 변환
            joint_positions = self._convert_hand_data_to_joints(hand_data)
            
            # 관절 제한 범위 적용
            joint_positions = self._apply_joint_limits(joint_positions)
            
            # 스무딩 적용
            if smoothing:
                joint_positions = self._apply_smoothing(joint_positions)
            
            # Shadow Hand 업데이트
            success = self._set_joint_positions(joint_positions)
            
            if success:
                self.target_joint_positions = joint_positions.copy()
                self.logger.debug(f"핸드 업데이트 성공: {joint_positions[:5]}...")
            
            return success
            
        except Exception as e:
            self.logger.error(f"핸드 업데이트 실패: {e}")
            return False
    
    def _convert_hand_data_to_joints(self, hand_data: np.ndarray) -> np.ndarray:
        """
        VR 핸드 데이터를 Shadow Hand 관절 위치로 변환
        
        Args:
            hand_data: VR 핸드 데이터 (21개 관절, 3D 좌표)
            
        Returns:
            Shadow Hand 관절 위치 (20개 관절)
        """
        # 간단한 변환: 3D 위치를 관절 각도로 변환
        # 실제 구현에서는 더 정교한 변환 로직 필요
        
        joint_positions = np.zeros(20)
        
        # 손목 관절 (WRJ0, WRJ1)
        joint_positions[17:19] = hand_data[0:2, 0] * 0.1  # X 좌표를 각도로
        
        # 엄지 관절 (THJ0-4)
        joint_positions[0:5] = hand_data[1:6, 1] * 0.1  # Y 좌표를 각도로
        
        # 검지 관절 (FFJ1-3)
        joint_positions[5:8] = hand_data[5:8, 2] * 0.1  # Z 좌표를 각도로
        
        # 중지 관절 (MFJ1-3)
        joint_positions[8:11] = hand_data[9:12, 1] * 0.1
        
        # 약지 관절 (RFJ1-3)
        joint_positions[11:14] = hand_data[13:16, 2] * 0.1
        
        # 새끼 관절 (LFJ1-3)
        joint_positions[14:17] = hand_data[17:20, 0] * 0.1
        
        return joint_positions
    
    def _apply_joint_limits(self, joint_positions: np.ndarray) -> np.ndarray:
        """관절 제한 범위 적용"""
        limited_positions = joint_positions.copy()
        
        # 손목 관절 제한
        limited_positions[17:19] = np.clip(
            limited_positions[17:19], 
            self.joint_limits["wrist"][0], 
            self.joint_limits["wrist"][1]
        )
        
        # 엄지 관절 제한
        limited_positions[0:5] = np.clip(
            limited_positions[0:5], 
            self.joint_limits["thumb"][0], 
            self.joint_limits["thumb"][1]
        )
        
        # 손가락 관절 제한
        limited_positions[5:17] = np.clip(
            limited_positions[5:17], 
            self.joint_limits["fingers"][0], 
            self.joint_limits["fingers"][1]
        )
        
        return limited_positions
    
    def _apply_smoothing(self, target_positions: np.ndarray) -> np.ndarray:
        """관절 위치에 스무딩 적용"""
        smoothing_factor = 0.3
        
        smoothed_positions = (
            self.current_joint_positions * (1 - smoothing_factor) + 
            target_positions * smoothing_factor
        )
        
        return smoothed_positions
    
    def _set_joint_positions(self, joint_positions: np.ndarray) -> bool:
        """Shadow Hand 관절 위치 설정"""
        if self.robot is None:
            return False
        
        try:
            # 관절 위치 설정
            if hasattr(self.robot, 'set_joint_position_target'):
                self.robot.set_joint_position_target(joint_positions)
            elif hasattr(self.robot, 'set_joint_positions'):
                self.robot.set_joint_positions(joint_positions)
            else:
                self.logger.warning("관절 위치 설정 메서드를 찾을 수 없습니다")
                return False
            
            # 시뮬레이션에 데이터 쓰기
            if hasattr(self.robot, 'write_data_to_sim'):
                self.robot.write_data_to_sim()
            
            # 현재 상태 업데이트
            self.current_joint_positions = joint_positions.copy()
            
            return True
            
        except Exception as e:
            self.logger.error(f"관절 위치 설정 실패: {e}")
            return False
    
    def get_joint_positions(self) -> np.ndarray:
        """현재 관절 위치 반환"""
        return self.current_joint_positions.copy()
    
    def get_joint_limits(self) -> Dict[str, List[float]]:
        """관절 제한 범위 반환"""
        return self.joint_limits.copy()
    
    def reset_hand(self):
        """Shadow Hand를 초기 위치로 리셋"""
        if not self.is_initialized:
            return False
        
        try:
            # 초기 관절 위치 (모든 관절 0도)
            initial_positions = np.zeros(20)
            
            # 리셋
            success = self._set_joint_positions(initial_positions)
            
            if success:
                self.logger.info("✅ Shadow Hand 리셋 완료")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Shadow Hand 리셋 실패: {e}")
            return False
    
    def step_environment(self):
        """환경 스텝 실행"""
        if self.env is None:
            return False
        
        try:
            # 환경 스텝 실행
            if hasattr(self.env, 'step'):
                self.env.step(None)
                return True
            else:
                self.logger.warning("환경 step 메서드를 찾을 수 없습니다")
                return False
                
        except Exception as e:
            self.logger.error(f"환경 스텝 실행 실패: {e}")
            return False
    
    def close(self):
        """컨트롤러 정리"""
        try:
            if self.env:
                self.env.close()
                self.env = None
            
            self.robot = None
            self.is_initialized = False
            
            self.logger.info("✅ Shadow Hand 컨트롤러 정리 완료")
            
        except Exception as e:
            self.logger.error(f"컨트롤러 정리 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        self.close()
