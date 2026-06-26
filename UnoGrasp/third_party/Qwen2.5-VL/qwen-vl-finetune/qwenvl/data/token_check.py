from transformers import AutoProcessor
from PIL import Image
import torch

# 模型路径（换成你实际的）
model_path = "/leonardo_scratch/fast/IscrC_4grasp/project/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct"

# 加载处理器
processor = AutoProcessor.from_pretrained(model_path).image_processor

# 随便加载一张图像
image_path = "/leonardo_scratch/fast/IscrC_4grasp/FreeGraspData/Meta_reason_data/data_ifl_0_scene0_5_labeled.png"  # 你替换成任意一张
image = Image.open(image_path).convert("RGB")

# 打印原始图像尺寸
print("Original image size:", image.size)  # (width, height)

# 预处理
out = processor.preprocess(image, return_tensors="pt")

# 打印结果
print("pixel_values shape:", out["pixel_values"].shape)
print("grid_thw:", out["image_grid_thw"])   # (T, H, W)

# 计算视觉 token 数
T, H, W = out["image_grid_thw"][0].tolist()
num_tokens = T * H * W
print(f"Total visual tokens: {num_tokens} (T={T}, H={H}, W={W})")
