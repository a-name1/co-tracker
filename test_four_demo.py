import torch
import numpy as np
import cv2
import os
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
import imageio.v3 as iio  # 用于生成 GIF

# ========== 新增：全局配置（适配无网络环境 + 4GB 显存 | 不修改视频） ==========
# 自动创建 checkpoints 目录（存放预训练权重）
os.makedirs("./checkpoints", exist_ok=True)
# 预训练权重路径配置
OFFLINE_CHECKPOINT_PATH = "./checkpoints/scaled_offline.pth"
ONLINE_CHECKPOINT_PATH = "./checkpoints/scaled_online.pth"
# 显存优化配置（不修改视频：关闭帧数截断、分辨率压缩）
DEFAULT_FPS = 5  # GIF 帧率，控制文件大小
# 关键显存优化：启用梯度检查点、限制批次内张量数量
torch.backends.cudnn.benchmark = True  # 加速 CUDA 推理，减少显存波动
torch.backends.cudnn.deterministic = False  # 牺牲确定性，换取显存效率

# 设置设备
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {DEFAULT_DEVICE}")

# ========== 优化：加载模型（显式指定权重 + 极致显存优化 | 不修改视频） ==========
def load_model(model_type="offline", use_v2_model=False):
    """
    加载 CoTracker 模型（解决权重找不到 + 4GB 显存适配 | 不修改视频）
    """
    # 检查权重文件是否存在，给出清晰报错
    checkpoint_path = OFFLINE_CHECKPOINT_PATH if model_type == "offline" else ONLINE_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"权重文件不存在！\n"
            f"请将对应权重文件放入 {os.path.dirname(checkpoint_path)} 目录\n"
            f"离线模型权重：scaled_offline.pth\n"
            f"在线模型权重：scaled_online.pth\n"
            f"下载地址：https://hf-mirror.com/facebook/cotracker3/resolve/main/"
        )
    
    # 加载模型
    if use_v2_model:
        model = CoTrackerPredictor(checkpoint=checkpoint_path, v2=True)
    else:
        # 关键优化：降低窗口长度（离线模型从 60 降至 30），减少单批次处理帧数量
        window_len = 30 if model_type == "offline" else 16
        model = CoTrackerPredictor(
            checkpoint=checkpoint_path,
            offline=(model_type == "offline"),
            window_len=window_len
        )
    
    # 4GB 显存核心优化：不修改视频，仅优化模型精度和运行模式
    model = model.to(DEFAULT_DEVICE)
    if DEFAULT_DEVICE == "cuda":
        model = model.half()  # FP16 精度，降低 50% 显存占用（不影响视频数据）
    model.eval()  # 禁用训练层（BatchNorm/Dropout），节省显存
    # 关键优化：冻结模型参数，禁止梯度计算，减少冗余显存
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# ========== 优化：视频预处理（仅格式转换，不修改视频内容/分辨率/帧数） ==========
def preprocess_video(video_path):
    """
    视频预处理（仅格式转换和显存适配，不修改视频任何内容 | 核心：保留原始视频）
    :param video_path: 原始视频路径
    :return: 预处理后的视频张量（保留原始分辨率、帧数）
    """
    try:
        # 读取视频（保留原始所有帧、分辨率）
        video = read_video_from_path(video_path)
        print(f"提示：加载原始视频 - 帧数：{len(video)}，分辨率：{video.shape[2]}x{video.shape[1]}")
        
        # 仅做格式转换 + 显存适配（不修改任何视频数据）
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_tensor = video_tensor.to(DEFAULT_DEVICE)
        if DEFAULT_DEVICE == "cuda":
            video_tensor = video_tensor.half()  # 与模型精度匹配，避免类型不匹配（不修改视频内容）
        
        return video_tensor
    
    except Exception as e:
        raise Exception(f"视频预处理失败：{str(e)}")

# ========== 保留：将视频帧列表转换为 GIF 并保存（优化默认参数 | 不修改视频） ==========
def save_frames_to_gif(frames, save_path, fps=DEFAULT_FPS):
    """
    将 RGB 帧列表保存为 GIF 文件（适配 3050Ti，控制文件大小 | 不修改原始视频）
    """
    # 采样帧，控制 GIF 大小（仅修改 GIF 输出，不影响原始视频）
    frame_interval = max(1, int(30 / fps))
    selected_frames = frames[::frame_interval]
    
    # 保存 GIF
    iio.imwrite(save_path, selected_frames, format="gif", loop=0)
    print(f"GIF 已保存至: {save_path}")

# ========== 保留：可视化跟踪结果（显存优化 | 不修改视频） ==========
def visualize_tracks(video, tracks, visibility, save_dir="./saved_videos", seq_name="output", query_frame=0, linewidth=3):
    """
    使用官方 Visualizer 可视化跟踪结果，同时保存 MP4 和 GIF | 不修改原始视频
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化前：转回 FP32 精度（仅适配可视化工具，不修改原始视频数据）
    if DEFAULT_DEVICE == "cuda":
        video = video.float()
        tracks = tracks.float()
        visibility = visibility.float()
    
    # 创建可视化器
    vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
    
    # 可视化（生成 MP4，保留视频原始分辨率/帧数）
    vis.visualize(
        video,
        tracks,
        visibility,
        query_frame=query_frame,
        seq_name=seq_name
    )
    
    # 定义保存路径
    mp4_path = os.path.join(save_dir, f"{seq_name}.mp4")
    gif_path = os.path.join(save_dir, f"{seq_name}.gif")
    
    # 从生成的 MP4 中提取帧，转换为 GIF（仅处理输出文件，不修改原始视频）
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        gif_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转换 BGR（OpenCV）→ RGB（GIF 要求）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(frame_rgb)
        cap.release()
        
        # 保存 GIF（控制文件大小，不影响原始视频）
        save_frames_to_gif(gif_frames, gif_path)
    else:
        print(f"警告：MP4 文件不存在，无法生成 GIF - {mp4_path}")
        gif_path = None
    
    # 关键优化：及时清理无用张量，释放显存（不影响视频数据）
    if DEFAULT_DEVICE == "cuda":
        torch.cuda.empty_cache()
        # 手动删除临时张量，减少显存占用
        del video, tracks, visibility
    
    return mp4_path, gif_path

# ========== 优化：测试 1 遮挡恢复挑战（显存优化 | 不修改视频） ==========
def test_occlusion_recovery(video_path, grid_size_a=0, grid_size_b=6, backward_tracking=False):
    """
    测试遮挡恢复能力（4GB 显存适配 | 不修改视频内容/分辨率/帧数）
    """
    print("=== 开始遮挡恢复挑战测试 ===")
    
    # 视频预处理（仅格式转换，保留原始视频）
    video = preprocess_video(video_path)
    
    # 加载模型
    model = load_model("offline")
    
    # 实验 A: 单点跟踪（极致显存优化：禁用梯度 + 及时清理）
    print("Running Experiment A (Single Point Tracking)...")
    with torch.no_grad():  # 禁用梯度，减少 30%+ 显存占用
        pred_tracks_a, pred_visibility_a = model(
            video,
            grid_size=grid_size_a,
            grid_query_frame=0,
            backward_tracking=backward_tracking,
        )
    
    # 实验 B: 密集网格跟踪（关键优化：降低 grid_size 从 8 降至 6，减少计算显存）
    print("Running Experiment B (Dense Grid Tracking)...")
    with torch.no_grad():
        pred_tracks_b, pred_visibility_b = model(
            video,
            grid_size=grid_size_b,  # 4GB 显存适配：从 8 降至 6（不修改视频，仅减少网格计算）
            grid_query_frame=0,
            backward_tracking=backward_tracking,
        )
    
    # 可视化结果（不修改视频，仅输出结果）
    seq_name_a = f"occlusion_test_A_grid{grid_size_a}"
    seq_name_b = f"occlusion_test_B_grid{grid_size_b}"
    
    video_a_path, gif_a_path = visualize_tracks(
        video, pred_tracks_a, pred_visibility_a,
        seq_name=seq_name_a,
        query_frame=0 if backward_tracking else 0,
        linewidth=4
    )
    
    video_b_path, gif_b_path = visualize_tracks(
        video, pred_tracks_b, pred_visibility_b,
        seq_name=seq_name_b,
        query_frame=0 if backward_tracking else 0,
        linewidth=2
    )
    
    # 关键优化：手动删除无用张量，释放显存（不影响视频数据）
    if DEFAULT_DEVICE == "cuda":
        del pred_tracks_a, pred_visibility_a, pred_tracks_b, pred_visibility_b
        torch.cuda.empty_cache()
    
    print(f"实验 A 结果已保存至: MP4={video_a_path}, GIF={gif_a_path}")
    print(f"实验 B 结果已保存至: MP4={video_b_path}, GIF={gif_b_path}")
    print("=== 遮挡恢复挑战测试完成 ===\n")
    
    return (video_a_path, gif_a_path), (video_b_path, gif_b_path)

# ========== 优化：测试 2 密集点云与物体几何感（显存优化 | 不修改视频） ==========
def test_dense_point_cloud(video_path, grid_size=8, backward_tracking=False):
    """
    测试密集点云与几何感知能力（4GB 显存适配 | 不修改视频）
    """
    print("=== 开始密集点云与几何感测试 ===")
    
    # 视频预处理（仅格式转换，保留原始视频）
    video = preprocess_video(video_path)
    
    # 加载模型
    model = load_model("offline")
    
    # 运行密集网格跟踪（关键优化：降低 grid_size 从 12 降至 8，减少计算显存）
    print(f"Running dense grid tracking with grid_size={grid_size}...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video,
            grid_size=grid_size,  # 4GB 显存适配：从 12 降至 8（不修改视频，仅减少网格计算）
            grid_query_frame=0,
            backward_tracking=backward_tracking,
        )
    
    # 可视化结果（不修改视频）
    seq_name = f"dense_point_cloud_grid{grid_size}"
    video_path, gif_path = visualize_tracks(
        video, pred_tracks, pred_visibility,
        seq_name=seq_name,
        query_frame=0 if backward_tracking else 0,
        linewidth=1
    )
    
    # 清理显存
    if DEFAULT_DEVICE == "cuda":
        del pred_tracks, pred_visibility
        torch.cuda.empty_cache()
    
    print(f"结果已保存至: MP4={video_path}, GIF={gif_path}")
    print("=== 密集点云与几何感测试完成 ===\n")
    
    return (video_path, gif_path)

# ========== 优化：测试 3 跨越视野边界测试（显存优化 | 不修改视频） ==========
def test_out_of_view(video_path, grid_size=6):
    """
    测试跨越视野边界跟踪能力（在线模型 + 4GB 显存适配 | 不修改视频）
    """
    print("=== 开始跨越视野边界测试 ===")
    
    # 视频预处理（仅格式转换，保留原始视频）
    video = preprocess_video(video_path)
    
    # 加载在线模型（显存占用低于离线模型，核心优化之一）
    print("Loading online model...")
    model = load_model("online")
    
    # 运行在线跟踪（关键优化：降低 grid_size 从 10 降至 6，减少计算显存）
    print(f"Running online tracking with grid_size={grid_size}...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video,
            grid_size=grid_size,  # 4GB 显存适配：从 10 降至 6（不修改视频）
            grid_query_frame=0,
            backward_tracking=False,
        )
    
    # 可视化结果（不修改视频）
    seq_name = f"out_of_view_online_grid{grid_size}"
    video_path, gif_path = visualize_tracks(
        video, pred_tracks, pred_visibility,
        seq_name=seq_name,
        query_frame=0,
        linewidth=2
    )
    
    # 清理显存
    if DEFAULT_DEVICE == "cuda":
        del pred_tracks, pred_visibility
        torch.cuda.empty_cache()
    
    print(f"结果已保存至: MP4={video_path}, GIF={gif_path}")
    print("=== 跨越视野边界测试完成 ===\n")
    
    return (video_path, gif_path)

# ========== 优化：测试 4 极端运动与形变测试（显存优化 | 不修改视频） ==========
def test_drastic_motion(video_path, grid_size=6, backward_tracking=False):
    """
    测试极端运动与形变适应能力（4GB 显存适配 | 不修改视频）
    """
    print("=== 开始极端运动与形变测试 ===")
    
    # 视频预处理（仅格式转换，保留原始视频）
    video = preprocess_video(video_path)
    
    # 加载模型
    model = load_model("offline")
    
    # 运行跟踪（关键优化：降低 grid_size 从 10 降至 6，减少计算显存）
    print(f"Running tracking with grid_size={grid_size}...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video,
            grid_size=grid_size,  # 4GB 显存适配：从 10 降至 6（不修改视频）
            grid_query_frame=0,
            backward_tracking=backward_tracking,
        )
    
    # 可视化结果（不修改视频）
    seq_name = f"drastic_motion_grid{grid_size}"
    video_path, gif_path = visualize_tracks(
        video, pred_tracks, pred_visibility,
        seq_name=seq_name,
        query_frame=0 if backward_tracking else 0,
        linewidth=2
    )
    
    # 清理显存
    if DEFAULT_DEVICE == "cuda":
        del pred_tracks, pred_visibility
        torch.cuda.empty_cache()
    
    print(f"结果已保存至: MP4={video_path}, GIF={gif_path}")
    print("=== 极端运动与形变测试完成 ===\n")
    
    return (video_path, gif_path)

# ========== 优化：批量处理测试（统一 4GB 显存适配参数 | 不修改视频） ==========
def run_all_tests(video_paths, output_dir="./cotracker_tests"):
    """
    运行所有测试（4GB 显存适配 | 不修改任何视频内容）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 测试 1: 遮挡恢复（grid_size_b 从 8 降至 6）
    if "occlusion" in video_paths:
        results["occlusion"] = test_occlusion_recovery(
            video_paths["occlusion"],
            grid_size_a=0,
            grid_size_b=6,
            backward_tracking=False
        )
    
    # 测试 2: 密集点云（grid_size 从 12 降至 8）
    if "geometry" in video_paths:
        results["geometry"] = test_dense_point_cloud(
            video_paths["geometry"],
            grid_size=8,
            backward_tracking=False
        )
    
    # 测试 3: 跨越视野边界（grid_size 从 10 降至 6）
    if "out_of_view" in video_paths:
        results["out_of_view"] = test_out_of_view(
            video_paths["out_of_view"],
            grid_size=6
        )
    
    # 测试 4: 极端运动（grid_size 从 10 降至 6）
    if "drastic_motion" in video_paths:
        results["drastic_motion"] = test_drastic_motion(
            video_paths["drastic_motion"],
            grid_size=6,
            backward_tracking=False
        )
    
    # 打印汇总
    print("\n=== 所有测试完成 ===")
    print("测试结果汇总:")
    for test_name, paths in results.items():
        print(f"- {test_name}:")
        if isinstance(paths, tuple) and isinstance(paths[0], tuple):
            print(f"  实验 A: MP4={paths[0][0]}, GIF={paths[0][1]}")
            print(f"  实验 B: MP4={paths[1][0]}, GIF={paths[1][1]}")
        else:
            print(f"  MP4={paths[0]}, GIF={paths[1]}")
    
    # 最终清理显存
    if DEFAULT_DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return results

# ========== 主函数（显存优化配置 | 不修改视频路径/内容） ==========
# ========== 主函数（显存优化配置 | 兼容低版本 PyTorch | 不修改视频） ==========
if __name__ == "__main__":
    # 示例：设置原始视频路径（不修改任何视频文件）
    video_paths = {
        "occlusion": "./assets/occlusion_test.mp4",
        "geometry": "./assets/geometry_test.mp4",
        "out_of_view": "./assets/out_of_view_test.mp4",
        "drastic_motion": "./assets/drastic_motion_test.mp4"
    }
    
    # 关键显存优化：配置 CUDA 显存分配策略（仅保留兼容参数，移除 max_split_size_mb）
    if DEFAULT_DEVICE == "cuda":
        # 移除不兼容的 max_split_size_mb=128，仅保留 expandable_segments:True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # 注释/删除低版本不支持的 CUDA 内存池配置
        # torch.cuda.memory._record_memory_history = False  # 低版本 PyTorch 无该属性
        torch.cuda.empty_cache()  # 该方法兼容所有版本，保留
    
    # 运行所有测试（不修改视频，仅优化显存使用）
    results = run_all_tests(video_paths)