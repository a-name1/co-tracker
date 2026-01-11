import os
import torch
import argparse
import numpy as np
from PIL import Image

# 确保这些库已安装
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/root/co-tracker/assets/car.mp4", help="视频文件路径或文件夹路径")
    parser.add_argument("--checkpoint", default="./checkpoints/scaled_offline.pth", help="预训练权重路径")
    parser.add_argument("--grid_size", type=int, default=40, help="网格采样密度")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="从哪一帧开始跟踪")
    parser.add_argument("--backward_tracking", action="store_true", help="是否双向跟踪")
    parser.add_argument("--offline", action="store_true", default=True, help="默认使用离线模式以获得更好效果")
    parser.add_argument("--save_dir", default="./saved_videos", help="结果保存目录")

    args = parser.parse_args()

    # 1. 确保保存目录存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"创建目录: {args.save_dir}")

    # 2. 严格的本地模型加载逻辑 (绕过 torch.hub)
    print(f"正在从本地加载模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到权重文件: {args.checkpoint}。请手动下载并放入指定位置。")

    # 针对 CoTracker3 的参数设置
    window_len = 60 if args.offline else 16
    
    model = CoTrackerPredictor(
        checkpoint=args.checkpoint,
        offline=args.offline,
        window_len=window_len
    )
    model = model.to(DEFAULT_DEVICE)
    print(f"模型已加载至: {DEFAULT_DEVICE}")

    # 3. 检查输入是单个文件还是文件夹
    video_list = []
    if os.path.isdir(args.video_path):
        video_list = [os.path.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    else:
        video_list = [args.video_path]

    # 4. 循环处理视频
    vis = Visualizer(save_dir=args.save_dir, pad_value=120, linewidth=3)
    
    for v_path in video_list:
        print(f"正在处理: {v_path}")
        try:
            video = read_video_from_path(v_path)
            # 转换格式为 [B, T, C, H, W]
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)

            # 运行跟踪
            pred_tracks, pred_visibility = model(
                video_tensor,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                backward_tracking=args.backward_tracking
            )

            # 可视化并保存
            # 自动提取文件名作为保存名称
            video_filename = os.path.basename(v_path)
            vis.visualize(
                video_tensor,
                pred_tracks,
                pred_visibility,
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
                filename=video_filename.split('.')[0] # 传入文件名避免覆盖
            )
            print(f"成功保存结果: {video_filename}")
        except Exception as e:
            print(f"处理视频 {v_path} 时出错: {e}")

    print("所有任务完成！")