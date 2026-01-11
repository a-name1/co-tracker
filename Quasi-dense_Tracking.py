import os
import torch
import numpy as np
import warnings
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# 忽略不影响运行的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 测试配置 ---
# 1. 修正后的路径：确保这里的文件名与服务器上真实文件名一致
VIDEO_PATH = "./assets/Quasi2.mp4" 
CHECKPOINT = "./checkpoints/scaled_offline.pth"
GRID_SIZE = 20 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_robustness_test():
    print(f"--- 开始极端鲁棒性测试 (并行点数: {GRID_SIZE**2}) ---")
    
    # 检查视频是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误：在路径 {VIDEO_PATH} 找不到视频文件！")
        print("请运行 'ls ./assets' 检查文件名。")
        return

    # 1. 初始化模型
    print(f"正在加载权重: {CHECKPOINT}...")
    model = CoTrackerPredictor(checkpoint=CHECKPOINT, offline=True, window_len=60)
    model = model.to(DEVICE)

    # 2. 加载视频并增加异常捕获
    print(f"正在读取视频: {VIDEO_PATH}...")
    video = read_video_from_path(VIDEO_PATH)
    
    if video is None:
        print("❌ 错误：视频读取失败，返回值为 None。可能是编码格式不支持或路径错误。")
        return
        
    # 转换为 Tensor [1, T, 3, H, W]
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEVICE)
    
    # 3. 推理阶段
    print(f"🚀 正在并行跟踪 {GRID_SIZE**2} 个点，请稍候...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video_tensor, 
            grid_size=GRID_SIZE,
            backward_tracking=True 
        )

    # 4. 可视化
    save_dir = "./robustness_results"
    os.makedirs(save_dir, exist_ok=True)
    
    vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=1, show_first_frame=True)
    
    output_name = "Quasi_dense_Tracking_Result"
    vis.visualize(video_tensor, pred_tracks, pred_visibility, filename=output_name)
    
    print(f"✅ 测试成功！结果保存至: {save_dir}/{output_name}.mp4")

if __name__ == "__main__":
    run_robustness_test()