import os
import torch
import numpy as np
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# --- 实验环境配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "./checkpoints/scaled_offline.pth"  # 确保权重文件在此路径
VIDEO_PATH = "/root/co-tracker/assets/zhebi1.mp4"      # 你的测试视频

def run_experiment(exp_type, video_tensor, model, grid_size):
    """
    运行特定的对比实验
    """
    print(f"\n🚀 正在启动 {exp_type} (grid_size={grid_size})...")
    
    # 执行推理
    # 实验 A (grid_size=0) 通常需要手动指定点，这里按传统方式模拟单点
    # 实验 B (grid_size=20) 开启密集网格
    
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video_tensor, 
            grid_size=grid_size,
            backward_tracking=True # 开启双向跟踪以满足“3D轮廓感”观察要求
        )

    # 可视化设置
    save_dir = f"./experiments/{exp_type.replace(' ', '_')}"
    # 使用 linewidth=1 以更好地展现 3D 轮廓感
    vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=1)
    
    vis.visualize(
        video_tensor, 
        pred_tracks, 
        pred_visibility, 
        filename="result"
    )
    print(f"✅ {exp_type} 完成，结果保存在: {save_dir}")

if __name__ == "__main__":
    # 1. 初始化模型 (采用离线模式以获得更高的鲁棒性)
    model = CoTrackerPredictor(checkpoint=CHECKPOINT, offline=True, window_len=60)
    model = model.to(DEVICE)

    # 2. 准备视频数据
    if not os.path.exists(VIDEO_PATH):
        print(f"找不到视频文件: {VIDEO_PATH}")
    else:
        video = read_video_from_path(VIDEO_PATH)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEVICE)

        # --- 执行实验 A: 单点跟踪 ---
        # 对应图2：传统方式，仅跟踪单个孤立点
        # 注意：在CoTracker中，grid_size=0通常需配合queries使用，这里设为3模拟极稀疏跟踪
        run_experiment("Experiment_A_Single_Point", video_tensor, model, grid_size=1)

        # --- 执行实验 B: 密集网格跟踪 ---
        # 对应图2：开启密集网格，跟踪物体及其周围环境
        run_experiment("Experiment_B_Dense_Grid", video_tensor, model, grid_size=20)

        # --- 执行图1要求的“并行能力验证”测试 ---
        # 对应图1：采用较大的网格尺寸 (grid_size >= 30)
        run_experiment("Robustness_Test_High_Density", video_tensor, model, grid_size=35)