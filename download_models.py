import os
import subprocess
import sys

# 设置国内镜像源，提升Hugging Face下载速度
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def install_dependencies():
    """安装CoTracker模型加载/运行所需的依赖库"""
    print("正在检查并安装CoTracker依赖库...")
    try:
        # CoTracker核心依赖（含torch、einops、timm等）
        dependencies = [
            "torch>=2.0.0",
            "torchvision",
            "einops",
            "timm",
            "huggingface-hub>=0.16.4",
            "opencv-python",
            "numpy"
        ]
        
        # 静默安装（-q），支持断点续传（--no-cache-dir避免缓存问题）
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *dependencies],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✅ 所有依赖库安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖库安装失败: {e.stderr.decode('utf-8')}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 依赖库安装异常: {e}")
        sys.exit(1)

# 先安装依赖，再导入模型相关模块
install_dependencies()

from huggingface_hub import snapshot_download

def download_cotracker_model(model_name, save_dir="./cotracker_models"):
    """
    下载并保存CoTracker预训练模型
    
    Args:
        model_name (str): CoTracker模型名称（HF Hub仓库名）
        save_dir (str): 模型保存根目录
    """
    # 规范化模型保存路径（替换/避免目录冲突）
    model_save_path = os.path.join(save_dir, model_name.replace("/", "_"))
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"\n正在下载模型: {model_name}")
    print(f"保存路径: {model_save_path}")
    
    try:
        # 下载预训练模型（忽略冗余文件，仅保留权重/配置）
        snapshot_download(
            repo_id=model_name,
            repo_type="model",
            local_dir=model_save_path,
            ignore_patterns=["*.md", "*.git*", "LICENSE", "README*"],
            # 断点续传 + 禁用并行下载（避免网络问题）
            resume_download=True,
            max_workers=1
        )
        
        # 验证核心文件是否存在
        core_files = ["pytorch_model.bin", "config.json"]
        missing_files = [f for f in core_files if not os.path.exists(os.path.join(model_save_path, f))]
        if missing_files:
            raise FileNotFoundError(f"核心文件缺失: {missing_files}")
        
        print(f"✅ 模型 {model_name} 下载完成！")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        # 清理不完整的目录
        if os.path.exists(model_save_path) and len(os.listdir(model_save_path)) == 0:
            os.rmdir(model_save_path)
        return False

def main():
    # CoTracker官方预训练模型（base/large版本）
    # 仓库地址：https://huggingface.co/facebookresearch
    cotracker_models = [
        # CoTracker-Base（轻量版，速度快）
        "facebookresearch/cotracker-base",
        # CoTracker-Large（高精度版，适合复杂场景）
        "facebookresearch/cotracker-large"
    ]
    
    print("="*60)
    print("开始下载CoTracker预训练模型...")
    print(f"总计 {len(cotracker_models)} 个模型（base + large）")
    print("="*60)
    
    success_count = 0
    for i, model in enumerate(cotracker_models, 1):
        print(f"\n[{i}/{len(cotracker_models)}]")
        if download_cotracker_model(model):
            success_count += 1
    
    # 下载结果汇总
    print("\n" + "="*60)
    print("🎉 下载任务结束！")
    print(f"成功下载: {success_count}/{len(cotracker_models)} 个模型")
    print("="*60)
    
    if success_count > 0:
        print(f"\n📁 模型保存目录: {os.path.abspath('./cotracker_models')}")
        print("\n💡 模型使用示例：")
        print("""
from cotracker import CoTracker
model = CoTracker.from_pretrained("./cotracker_models/facebookresearch_cotracker-base")
# 或加载large版本
# model = CoTracker.from_pretrained("./cotracker_models/facebookresearch_cotracker-large")
        """)
    else:
        print("\n❌ 无模型下载成功，请检查网络或重试！")

if __name__ == "__main__":
    main()