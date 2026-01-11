import os
import torch
import numpy as np
from PIL import Image
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

# --- 配置 ---
VIDEO_PATH = "./assets/ge.mp4" 
MASK_PATH = "/root/co-tracker/assets/mask_result.png"  # 您生成的掩码图路径
CHECKPOINT = "./checkpoints/scaled_offline.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 即使设置了 GRID_SIZE，我们也会根据掩码进行过滤
GRID_SIZE = 40  

def load_video_and_mask(video_path, mask_path, max_width=640):
    """使用 PIL 读取视频帧并处理掩码"""
    import imageio
    print(f"正在读取视频: {video_path}...")
    reader = imageio.get_reader(video_path)
    frames = []
    
    # 1. 加载并调整视频帧
    for frame in reader:
        img = Image.fromarray(frame)
        if img.width > max_width:
            w, h = max_width, int(img.height * max_width / img.width)
            img = img.resize((w, h), Image.BILINEAR)
        frames.append(np.array(img))
    video_np = np.stack(frames)
    
    # 2. 加载并调整掩码尺寸，确保与缩放后的视频一致
    target_h, target_w = video_np.shape[1], video_np.shape[2]
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask_img) > 128  # 转换为布尔掩码
    
    return video_np, mask_np

def run_masked_experiment():
    # 1. 环境检查
    if not os.path.exists(MASK_PATH):
        print(f"错误: 找不到掩码文件 {MASK_PATH}，请先生成掩码！")
        return

    # 2. 加载模型
    model = CoTrackerPredictor(checkpoint=CHECKPOINT, offline=True, window_len=60).to(DEVICE)

    # 3. 准备视频和掩码
    video_np, mask_np = load_video_and_mask(VIDEO_PATH, MASK_PATH)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(DEVICE)
    
    # 4. 关键步骤：过滤跟踪点
    # 获取网格点并只保留掩码为 True (白色部分) 的点
    print(f"正在根据掩码筛选跟踪点...")
    # 这里我们使用 CoTracker 内置的逻辑，但手动应用 segm_mask
    segm_mask = torch.from_numpy(mask_np).to(DEVICE).float()[None, None] # [1, 1, H, W]

    # --- 5. 执行双向推理 ---
    # 我们从视频的中间帧 (mid_frame) 开始采样点
    mid_frame = video_tensor.shape[1] // 2

    with torch.no_grad():
        print(f"正在从第 {mid_frame} 帧开始双向追踪...")
        pred_tracks, pred_visibility = model(
            video_tensor, 
            grid_size=GRID_SIZE, 
            segm_mask=segm_mask, 
            grid_query_frame=mid_frame, # 关键：从中间开始采样
            backward_tracking=True      # 关键：开启双向追溯
        )

    # --- 6. 生成“极度流动”的彩虹轨迹 ---
    save_dir = "./rainbow_results"
    os.makedirs(save_dir, exist_ok=True)

    vis = Visualizer(
        save_dir=save_dir,
        linewidth=2,                # 线条略微加粗，流动感更强
        mode="rainbow",             # 彩虹模式
        tracks_leave_trace=len(video_np), # 关键：设为视频总长度，使线条贯穿始终不消失
        fps=25
    )

    output_name = "fluid_bidirectional_rainbow"
    # 注意：确保可视化时包含完整的预测
    vis.visualize(
        video_tensor, 
        pred_tracks, 
        pred_visibility, 
        filename=output_name,
        query_frame=mid_frame       # 告诉可视化器从哪一帧开始着色
    )
    
    output_name = "masked_rainbow_experiment"
    vis.visualize(video_tensor, pred_tracks, pred_visibility, filename=output_name)
    print(f"实验完成！只包含掩码区域的彩虹线视频已保存至: {save_dir}/{output_name}.mp4")

if __name__ == "__main__":
    run_masked_experiment()