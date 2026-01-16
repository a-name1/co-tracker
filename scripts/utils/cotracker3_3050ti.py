import os
import torch
import numpy as np
import cv2
import gc
from PIL import Image
import torch.nn.functional as F
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

# ========== 1. 显存策略配置 ==========
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_video_360p(video_path):
    """
    读取视频并强制缩放至 360P 宽度（保持比例或强制 640x360）
    """
    video = read_video_from_path(video_path) # numpy [T, H, W, C]
    T, H, W, C = video.shape
    
    # 计算缩放比例，目标高度为 360
    new_h = 360
    new_w = int(W * (new_h / H))
    # 确保宽度是 8 的倍数（对于某些卷积网络更友好）
    new_w = (new_w // 8) * 8
    
    print(f"--- 正在缩放视频: 原分辨率 {W}x{H} -> 缩放后 {new_w}x{new_h} ---")
    
    # 使用 OpenCV 进行快速缩放
    resized_video = []
    for frame in video:
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_video.append(resized_frame)
    
    video_np = np.stack(resized_video)
    # 转换为 Tensor: [1, T, C, H, W]
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
    return video_tensor

# ========== 2. 深度优化的推理函数 ==========
def run_tracker_360p(model, video_path, grid_size=6, task_name="task"):
    print(f"\n>>> 开始任务: {task_name}")
    
    # 获取缩放后的视频张量
    video_tensor = preprocess_video_360p(video_path)
    
    try:
        # 转移到 GPU 并开启半精度
        video_tensor = video_tensor.to(DEVICE)
        if DEVICE == "cuda":
            video_tensor = video_tensor.half()

        with torch.inference_mode():
            # 使用 autocast 自动协调类型冲突
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # 在 360P 下，grid_size 可以稍微大一点，比如 6 或 8
                pred_tracks, pred_visibility = model(
                    video_tensor,
                    grid_size=grid_size,
                    grid_query_frame=0
                )
        
        # 保存结果
        save_dir = "./cotracker3_results_360p"
        os.makedirs(save_dir, exist_ok=True)
        
        # 可视化前转回 FP32 避免报错
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=2)
        vis.visualize(
            video_tensor.float(), 
            pred_tracks.float(), 
            pred_visibility.float(), 
            seq_name=f"{task_name}_360p"
        )
        print(f"--- 任务 {task_name} 成功！结果已存至 {save_dir} ---")

    except Exception as e:
        print(f"!!! 任务 {task_name} 失败: {str(e)}")
    finally:
        # 强制释放显存
        del video_tensor
        torch.cuda.empty_cache()
        gc.collect()

# ========== 3. 主逻辑 ==========
def main():
    video_tasks = {
        "occlusion": "./assets/occlusion_test.mp4",
        "geometry": "./assets/geometry_test.mp4",
        "out_of_view": "./assets/out_of_view_test.mp4",
        "drastic_motion": "./assets/drastic_motion_test.mp4"
    }

    # 加载模型 (CoTracker3 Offline)
    # 确保权重在 ./checkpoints/scaled_offline.pth
    checkpoint = "./checkpoints/scaled_offline.pth"
    if not os.path.exists(checkpoint):
        print("错误：请先下载权重文件到 ./checkpoints 目录")
        return

    model = CoTrackerPredictor(checkpoint=checkpoint, offline=True)
    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model = model.half()
    model.eval()

    # 依次执行任务
    for name, path in video_tasks.items():
        if os.path.exists(path):
            run_tracker_360p(model, path, grid_size=6, task_name=name)
        else:
            print(f"跳过任务 {name}，找不到视频文件: {path}")

if __name__ == "__main__":
    main()