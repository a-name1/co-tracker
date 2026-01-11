import os
import torch
import numpy as np
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# 配置环境
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "./checkpoints/scaled_offline.pth"
VIDEO_PATH = "./assets/zhebi1.mp4"       # 基础视频
MASK_PATH = "./assets/difference_mask_default.png"   # 掩码文件（用于特定目标测试）

class CoTrackerShowcase:
    def __init__(self, checkpoint, device):
        print(f"--- 正在初始化 CoTracker3 (设备: {device}) ---")
        # 默认加载离线模型，因为它在处理复杂遮挡时最强
        self.model = CoTrackerPredictor(checkpoint=checkpoint, offline=True).to(device)
        self.device = device

    def load_video(self, path):
        video = read_video_from_path(path) # [T, H, W, 3]
        return torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)

    def run_demo(self, name, video_tensor, **kwargs):
        """通用运行函数，方便根据不同参数对比"""
        print(f"\n[功能展现]: {name}")
        save_dir = f"./showcase/{name.replace(' ', '_').lower()}"
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=2)
        
        # 核心推理
        tracks, visibility = self.model(video_tensor, **kwargs)
        
        # 可视化保存
        vis.visualize(video_tensor, tracks, visibility, filename="result")
        print(f"结果已保存至: {save_dir}")

# --- 主程序：分层展现不同功能 ---
if __name__ == "__main__":
    showcase = CoTrackerShowcase(CHECKPOINT, DEVICE)
    video = showcase.load_video(VIDEO_PATH)

    # 层次 1：密集网格跟踪 (Grid Tracking)
    # 功能：展现视频整体的运动场，类似光流但更长程、更稳定。
    showcase.run_demo(
        "Level 1 - Global Motion", 
        video, 
        grid_size=20 # 20x20=400个点，覆盖全场
    )

    # 层次 2：特定目标聚焦 (Mask-based Tracking)
    # 功能：只跟踪特定物体（如只跟踪苹果），忽略背景。
    # 这展示了模型如何与分割掩码结合，进行精准的物体动态分析。
    if os.path.exists(MASK_PATH):
        mask = np.array(Image.open(MASK_PATH))
        mask_tensor = torch.from_numpy(mask)[None, None].to(DEVICE)
        showcase.run_demo(
            "Level 2 - Object Focused", 
            video, 
            segm_mask=mask_tensor, # 传入掩码
            grid_size=15
        )

    # 层次 3：双向跟踪 (Backward Tracking)
    # 功能：不仅向后看物体去哪了，还向前看物体从哪来。
    # 适合展现物体的完整生命周期，特别是在物体中途出现的视频中。
    showcase.run_demo(
        "Level 3 - Past and Future", 
        video, 
        grid_size=10,
        grid_query_frame=len(video[0]) // 2, # 从视频中间那一帧开始采样
        backward_tracking=True               # 开启向后追溯
    )

    print("\n--- 所有功能展现运行完毕！请查看 ./showcase 目录 ---")