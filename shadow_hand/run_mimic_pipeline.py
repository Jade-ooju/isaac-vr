"""
Isaac Lab Mimic 통합 파이프라인 실행 스크립트
Windows 환경에서 VR 텔레오퍼레이션을 사용한 Isaac Lab Mimic 워크플로우
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mimic_integration import VRMimicDemoCollector, MimicDemoConfig
from mimic_demo_recorder import MimicDemoRecorder
from robomimic_integration import RobomimicIntegration

class MimicPipeline:
    """Isaac Lab Mimic 통합 파이프라인"""
    
    def __init__(self, config_path: str = None):
        """
        파이프라인 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.demo_recorder = None
        self.robomimic_integration = None
        
        self._initialize_components()
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mimic_pipeline.log')
            ]
        )
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 데모 녹화기 초기화
            self.demo_recorder = MimicDemoRecorder(config_path=self.config_path)
            
            # Robomimic 통합 초기화
            self.robomimic_integration = RobomimicIntegration()
            
            self.logger.info("✅ Isaac Lab Mimic 파이프라인 초기화 완료!")
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def collect_demos(self, task_name: str, num_demos: int = 5, 
                     duration: float = 10.0, enable_subtask_annotation: bool = True):
        """
        VR을 사용한 데모 수집
        
        Args:
            task_name: 태스크 이름
            num_demos: 수집할 데모 수
            duration: 각 데모의 길이 (초)
            enable_subtask_annotation: 서브태스크 주석 달기 활성화
        """
        self.logger.info(f"🎬 데모 수집 시작: {task_name}")
        
        try:
            # 데모 수집 실행
            self.demo_recorder.start_demo_collection(
                task_name=task_name,
                num_demos=num_demos,
                duration=duration,
                enable_subtask_annotation=enable_subtask_annotation
            )
            
            self.logger.info("✅ 데모 수집 완료!")
            
        except Exception as e:
            self.logger.error(f"데모 수집 실패: {e}")
            raise
    
    def convert_to_robomimic(self, mimic_dataset_path: str, output_path: str = None):
        """
        Isaac Lab Mimic 데이터를 Robomimic 형식으로 변환
        
        Args:
            mimic_dataset_path: Isaac Lab Mimic 데이터셋 경로
            output_path: 변환된 데이터셋 저장 경로
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(mimic_dataset_path), "robomimic_format")
        
        self.logger.info(f"🔄 Robomimic 형식으로 변환 중...")
        
        try:
            # 데이터 변환
            self.robomimic_integration.convert_to_robomimic_format(
                mimic_dataset_path=mimic_dataset_path,
                output_path=output_path,
                task_name="shadow_hand_manipulation"
            )
            
            self.logger.info(f"✅ 변환 완료: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"데이터 변환 실패: {e}")
            raise
    
    def train_policy(self, dataset_path: str, output_dir: str = None):
        """
        Robomimic을 사용한 정책 훈련
        
        Args:
            dataset_path: 훈련 데이터셋 경로
            output_dir: 출력 디렉토리
        """
        if output_dir is None:
            output_dir = os.path.join(dataset_path, "trained_models")
        
        self.logger.info(f"🤖 정책 훈련 시작...")
        
        try:
            # 정책 훈련
            model_path = self.robomimic_integration.train_policy(
                dataset_path=dataset_path,
                output_dir=output_dir
            )
            
            self.logger.info(f"✅ 정책 훈련 완료: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"정책 훈련 실패: {e}")
            raise
    
    def evaluate_policy(self, model_path: str, dataset_path: str):
        """
        훈련된 정책 평가
        
        Args:
            model_path: 훈련된 모델 경로
            dataset_path: 평가 데이터셋 경로
        """
        self.logger.info(f"📊 정책 평가 시작...")
        
        try:
            # 정책 평가
            metrics = self.robomimic_integration.evaluate_policy(
                model_path=model_path,
                dataset_path=dataset_path
            )
            
            self.logger.info(f"✅ 정책 평가 완료!")
            self.logger.info(f"📈 평가 결과: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"정책 평가 실패: {e}")
            raise
    
    def run_full_pipeline(self, task_name: str, num_demos: int = 5, 
                         duration: float = 10.0, enable_subtask_annotation: bool = True):
        """
        전체 파이프라인 실행 (데모 수집 → 변환 → 훈련 → 평가)
        
        Args:
            task_name: 태스크 이름
            num_demos: 수집할 데모 수
            duration: 각 데모의 길이 (초)
            enable_subtask_annotation: 서브태스크 주석 달기 활성화
        """
        self.logger.info(f"🚀 Isaac Lab Mimic 전체 파이프라인 시작: {task_name}")
        
        try:
            # 1. 데모 수집
            self.logger.info("=== 1단계: VR 데모 수집 ===")
            self.collect_demos(
                task_name=task_name,
                num_demos=num_demos,
                duration=duration,
                enable_subtask_annotation=enable_subtask_annotation
            )
            
            # 2. Robomimic 형식으로 변환
            self.logger.info("=== 2단계: Robomimic 형식 변환 ===")
            mimic_dataset_path = self._find_latest_mimic_dataset(task_name)
            robomimic_dataset_path = self.convert_to_robomimic(mimic_dataset_path)
            
            # 3. 정책 훈련
            self.logger.info("=== 3단계: 정책 훈련 ===")
            model_path = self.train_policy(robomimic_dataset_path)
            
            # 4. 정책 평가
            self.logger.info("=== 4단계: 정책 평가 ===")
            metrics = self.evaluate_policy(model_path, robomimic_dataset_path)
            
            self.logger.info("🎉 전체 파이프라인 완료!")
            self.logger.info(f"📊 최종 결과: {metrics}")
            
            return {
                "mimic_dataset": mimic_dataset_path,
                "robomimic_dataset": robomimic_dataset_path,
                "model_path": model_path,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"파이프라인 실행 실패: {e}")
            raise
    
    def _find_latest_mimic_dataset(self, task_name: str) -> str:
        """최신 Isaac Lab Mimic 데이터셋 찾기"""
        mimic_save_path = Path(self.demo_recorder.save_path)
        
        # 태스크 이름으로 시작하는 HDF5 파일 찾기
        pattern = f"{task_name}_mimic_dataset_*.h5"
        matching_files = list(mimic_save_path.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"태스크 '{task_name}'에 대한 Mimic 데이터셋을 찾을 수 없습니다")
        
        # 가장 최신 파일 반환
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.demo_recorder:
                self.demo_recorder.cleanup()
            
            self.logger.info("✅ 파이프라인 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 중 에러: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Isaac Lab Mimic 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행
  python run_mimic_pipeline.py --action full --task "Isaac-Repose-Cube-Shadow-Direct-v0"
  
  # 데모 수집만 실행
  python run_mimic_pipeline.py --action collect --task "Isaac-Repose-Cube-Shadow-Direct-v0" --num-demos 10
  
  # 데이터 변환만 실행
  python run_mimic_pipeline.py --action convert --dataset "./mimic_demos/task_dataset.h5"
  
  # 정책 훈련만 실행
  python run_mimic_pipeline.py --action train --dataset "./robomimic_format"
        """
    )
    
    parser.add_argument("--action", 
                       choices=["full", "collect", "convert", "train", "evaluate"],
                       required=True,
                       help="실행할 작업")
    parser.add_argument("--task", type=str, help="태스크 이름")
    parser.add_argument("--dataset", type=str, help="데이터셋 경로")
    parser.add_argument("--output", type=str, help="출력 경로")
    parser.add_argument("--num-demos", type=int, default=5, help="수집할 데모 수")
    parser.add_argument("--duration", type=float, default=10.0, help="각 데모의 길이 (초)")
    parser.add_argument("--no-subtask", action="store_true", help="서브태스크 주석 비활성화")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--model", type=str, help="모델 경로 (평가 시)")
    
    args = parser.parse_args()
    
    try:
        # 파이프라인 초기화
        pipeline = MimicPipeline(config_path=args.config)
        
        if args.action == "full":
            # 전체 파이프라인 실행
            if not args.task:
                raise ValueError("전체 파이프라인 실행 시 --task가 필요합니다")
            
            result = pipeline.run_full_pipeline(
                task_name=args.task,
                num_demos=args.num_demos,
                duration=args.duration,
                enable_subtask_annotation=not args.no_subtask
            )
            
            print(f"\n🎉 파이프라인 완료!")
            print(f"📁 Mimic 데이터셋: {result['mimic_dataset']}")
            print(f"📁 Robomimic 데이터셋: {result['robomimic_dataset']}")
            print(f"🤖 훈련된 모델: {result['model_path']}")
            print(f"📊 평가 결과: {result['metrics']}")
            
        elif args.action == "collect":
            # 데모 수집만 실행
            if not args.task:
                raise ValueError("데모 수집 시 --task가 필요합니다")
            
            pipeline.collect_demos(
                task_name=args.task,
                num_demos=args.num_demos,
                duration=args.duration,
                enable_subtask_annotation=not args.no_subtask
            )
            
        elif args.action == "convert":
            # 데이터 변환만 실행
            if not args.dataset:
                raise ValueError("데이터 변환 시 --dataset이 필요합니다")
            
            output_path = pipeline.convert_to_robomimic(args.dataset, args.output)
            print(f"✅ 변환 완료: {output_path}")
            
        elif args.action == "train":
            # 정책 훈련만 실행
            if not args.dataset:
                raise ValueError("정책 훈련 시 --dataset이 필요합니다")
            
            model_path = pipeline.train_policy(args.dataset, args.output)
            print(f"✅ 훈련 완료: {model_path}")
            
        elif args.action == "evaluate":
            # 정책 평가만 실행
            if not args.model or not args.dataset:
                raise ValueError("정책 평가 시 --model과 --dataset이 필요합니다")
            
            metrics = pipeline.evaluate_policy(args.model, args.dataset)
            print(f"✅ 평가 완료: {metrics}")
        
    except Exception as e:
        logging.error(f"파이프라인 실행 실패: {e}")
        return 1
    
    finally:
        pipeline.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())
