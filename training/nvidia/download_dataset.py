from datasets import load_dataset
import os

# 데이터셋 이름
dataset_name = "nvidia/PhysicalAI-GR00T-Tuned-Tasks"

# 다운로드 경로 설정 (NVIDIA 폴더 안에 데이터셋 이름과 같은 폴더 생성)
dataset_folder_name = dataset_name.split("/")[-1]  # "PhysicalAI-Robotics-GR00T-Eval"
download_path = os.path.join(r"C:\IsaacLab\IsaacLab\custom_scripts\datasets\NVIDIA", dataset_folder_name)
os.makedirs(download_path, exist_ok=True)

print(f"데이터셋 다운로드 중: {dataset_name}")
print(f"다운로드 경로: {download_path}")

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset(dataset_name, cache_dir=download_path)

print("다운로드 완료!")
print(ds)