#!/usr/bin/env python3
"""
GR00T-Eval 데이터셋을 IsaacLab에서 활용하는 스크립트

이 스크립트는 NVIDIA의 GR00T-Eval 데이터셋을 로드하고 분석하여
IsaacLab 환경에서 태스크 평가나 벤치마킹에 활용할 수 있도록 합니다.

사용법:
    python gr00t_eval_integration.py --analyze_tasks
    python gr00t_eval_integration.py --export_task_list --output_file ./datasets/gr00t_tasks.json
"""

import argparse
import json
import os
from datasets import load_dataset
from typing import Dict, List, Any
import re


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze and utilize GR00T-Eval dataset")
    parser.add_argument(
        "--input_dataset", 
        type=str, 
        default="nvidia/PhysicalAI-Robotics-GR00T-Eval",
        help="Name of the GR00T-Eval dataset"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Cache directory for datasets (optional)"
    )
    parser.add_argument(
        "--analyze_tasks", 
        action="store_true",
        help="Analyze and categorize tasks in the dataset"
    )
    parser.add_argument(
        "--export_task_list", 
        action="store_true",
        help="Export task list to JSON file"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="./datasets/gr00t_tasks.json",
        help="Output file for task list"
    )
    parser.add_argument(
        "--filter_by_keywords", 
        type=str, 
        nargs="+",
        help="Filter tasks by keywords (e.g., 'pick', 'place', 'open')"
    )
    return parser.parse_args()


class GR00TEvalAnalyzer:
    """GR00T-Eval 데이터셋 분석 및 활용 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.dataset = None
        self.tasks = []
        
    def load_dataset(self):
        """GR00T-Eval 데이터셋을 로드합니다."""
        print(f"Loading GR00T-Eval dataset: {self.args.input_dataset}")
        
        cache_dir = self.args.cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "NVIDIA")
        
        self.dataset = load_dataset(
            self.args.input_dataset, 
            cache_dir=cache_dir
        )
        
        print(f"Dataset loaded successfully!")
        print(f"  - Split: {list(self.dataset.keys())}")
        print(f"  - Total tasks: {len(self.dataset['train'])}")
        
        # 태스크 리스트 추출
        self.tasks = [item['text'] for item in self.dataset['train']]
        
    def categorize_tasks(self) -> Dict[str, List[str]]:
        """태스크를 카테고리별로 분류합니다."""
        categories = {
            "pick_and_place": [],
            "manipulation": [],
            "opening_closing": [],
            "cleaning": [],
            "cooking": [],
            "music": [],
            "writing_drawing": [],
            "other": []
        }
        
        # 키워드 기반 분류
        for task in self.tasks:
            task_lower = task.lower()
            
            if any(keyword in task_lower for keyword in ["pick up", "place", "put", "move"]):
                categories["pick_and_place"].append(task)
            elif any(keyword in task_lower for keyword in ["open", "close", "shut"]):
                categories["opening_closing"].append(task)
            elif any(keyword in task_lower for keyword in ["wipe", "clean", "erase", "sweep"]):
                categories["cleaning"].append(task)
            elif any(keyword in task_lower for keyword in ["cook", "stir", "pour", "mix", "cut"]):
                categories["cooking"].append(task)
            elif any(keyword in task_lower for keyword in ["strum", "play", "hit", "shake", "wave"]):
                categories["music"].append(task)
            elif any(keyword in task_lower for keyword in ["write", "draw", "press", "tap"]):
                categories["writing_drawing"].append(task)
            elif any(keyword in task_lower for keyword in ["use", "grab", "hold", "turn", "press"]):
                categories["manipulation"].append(task)
            else:
                categories["other"].append(task)
        
        return categories
    
    def analyze_tasks(self):
        """태스크를 분석하고 통계를 출력합니다."""
        if not self.tasks:
            self.load_dataset()
        
        print("\n" + "=" * 60)
        print("GR00T-Eval Task Analysis")
        print("=" * 60)
        
        # 기본 통계
        print(f"Total tasks: {len(self.tasks)}")
        print(f"Average task length: {sum(len(task) for task in self.tasks) / len(self.tasks):.1f} characters")
        
        # 카테고리별 분석
        categories = self.categorize_tasks()
        
        print(f"\nTask Categories:")
        for category, tasks in categories.items():
            if tasks:
                print(f"  {category.replace('_', ' ').title()}: {len(tasks)} tasks")
        
        # 키워드 분석
        print(f"\nMost common action words:")
        action_words = {}
        for task in self.tasks:
            words = re.findall(r'\b(?:use|pick|place|open|close|grab|hold|turn|press|move|put|take|get)\b', task.lower())
            for word in words:
                action_words[word] = action_words.get(word, 0) + 1
        
        sorted_words = sorted(action_words.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:10]:
            print(f"  {word}: {count} times")
        
        # 샘플 태스크 출력
        print(f"\nSample tasks by category:")
        for category, tasks in categories.items():
            if tasks:
                print(f"\n  {category.replace('_', ' ').title()}:")
                for i, task in enumerate(tasks[:3]):  # 각 카테고리에서 최대 3개
                    print(f"    {i+1}. {task}")
                if len(tasks) > 3:
                    print(f"    ... and {len(tasks) - 3} more")
    
    def filter_tasks_by_keywords(self, keywords: List[str]) -> List[str]:
        """키워드로 태스크를 필터링합니다."""
        filtered_tasks = []
        for task in self.tasks:
            task_lower = task.lower()
            if any(keyword.lower() in task_lower for keyword in keywords):
                filtered_tasks.append(task)
        return filtered_tasks
    
    def export_task_list(self):
        """태스크 리스트를 JSON 파일로 내보냅니다."""
        if not self.tasks:
            self.load_dataset()
        
        # 필터링 적용
        tasks_to_export = self.tasks
        if self.args.filter_by_keywords:
            tasks_to_export = self.filter_tasks_by_keywords(self.args.filter_by_keywords)
            print(f"Filtered {len(tasks_to_export)} tasks by keywords: {self.args.filter_by_keywords}")
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 태스크 데이터 구성
        task_data = {
            "dataset_info": {
                "name": "GR00T-Eval",
                "source": self.args.input_dataset,
                "total_tasks": len(self.tasks),
                "exported_tasks": len(tasks_to_export)
            },
            "categories": self.categorize_tasks(),
            "tasks": [
                {
                    "id": i,
                    "text": task,
                    "category": self._get_task_category(task)
                }
                for i, task in enumerate(tasks_to_export)
            ]
        }
        
        # JSON 파일로 저장
        with open(self.args.output_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        
        print(f"Task list exported to: {self.args.output_file}")
        print(f"  - Total tasks: {len(tasks_to_export)}")
        
    def _get_task_category(self, task: str) -> str:
        """태스크의 카테고리를 반환합니다."""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["pick up", "place", "put", "move"]):
            return "pick_and_place"
        elif any(keyword in task_lower for keyword in ["open", "close", "shut"]):
            return "opening_closing"
        elif any(keyword in task_lower for keyword in ["wipe", "clean", "erase", "sweep"]):
            return "cleaning"
        elif any(keyword in task_lower for keyword in ["cook", "stir", "pour", "mix", "cut"]):
            return "cooking"
        elif any(keyword in task_lower for keyword in ["strum", "play", "hit", "shake", "wave"]):
            return "music"
        elif any(keyword in task_lower for keyword in ["write", "draw", "press", "tap"]):
            return "writing_drawing"
        elif any(keyword in task_lower for keyword in ["use", "grab", "hold", "turn", "press"]):
            return "manipulation"
        else:
            return "other"
    
    def generate_isaaclab_task_configs(self):
        """IsaacLab에서 사용할 수 있는 태스크 설정을 생성합니다."""
        # 이 기능은 향후 확장 가능
        print("Task config generation feature coming soon...")
    
    def run(self):
        """메인 실행 함수"""
        if self.args.analyze_tasks:
            self.analyze_tasks()
        
        if self.args.export_task_list:
            self.export_task_list()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("GR00T-Eval Dataset Analyzer")
    print("=" * 60)
    print(f"Input dataset: {args.input_dataset}")
    if args.output_file:
        print(f"Output file: {args.output_file}")
    print("=" * 60)
    
    # 분석기 생성 및 실행
    analyzer = GR00TEvalAnalyzer(args)
    analyzer.run()
    
    print("\n✅ Analysis completed successfully!")
    print(f"\nUsage suggestions:")
    print(f"  - Use exported task list for IsaacLab environment configuration")
    print(f"  - Filter tasks by specific keywords for targeted evaluation")
    print(f"  - Integrate with IsaacLab's task evaluation framework")


if __name__ == "__main__":
    main()
