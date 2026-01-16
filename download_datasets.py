import os
import zipfile
import tarfile
import subprocess
from huggingface_hub import snapshot_download

# 配置HF镜像源，提升下载速度
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_cotracker_datasets(root_dir="./cotracker_datasets", skip_existing=True):
    """
    下载 CoTracker 训练/评估所需的核心数据集
    :param root_dir: 数据集根目录
    :param skip_existing: 是否跳过已下载的数据集（避免重复下载）
    """
    os.makedirs(root_dir, exist_ok=True)
    
    # ========== 1. Kubric 合成数据集 (训练核心) ==========
    kubric_dir = os.path.join(root_dir, "kubric")
    if skip_existing and os.path.exists(kubric_dir):
        print("✅ Kubric 数据集已存在，跳过下载")
    else:
        print("📥 正在下载 Kubric 合成数据集 (CoTracker 训练核心)...")
        # Kubric数据集托管在HF Hub，按需下载
        snapshot_download(
            repo_id="facebookresearch/cotracker-kubric",
            repo_type="dataset",
            local_dir=kubric_dir,
            ignore_patterns=["*.git*", "README.md"]
        )
        print("✅ Kubric 数据集下载完成")

    # ========== 2. TapVid 基准数据集 (评估核心) ==========
    tapvid_dir = os.path.join(root_dir, "tapvid")
    tapvid_subsets = {
        "tapvid_kinetics": "facebookresearch/tapvid-kinetics",
        "tapvid_robotap": "facebookresearch/tapvid-robotap",
        "tapvid_davis": "facebookresearch/tapvid-davis"
    }
    
    for subset_name, hf_repo in tapvid_subsets.items():
        subset_dir = os.path.join(tapvid_dir, subset_name)
        if skip_existing and os.path.exists(subset_dir):
            print(f"✅ TapVid-{subset_name} 已存在，跳过下载")
            continue
        
        print(f"📥 正在下载 TapVid-{subset_name} 数据集...")
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=subset_dir,
            ignore_patterns=["*.git*", "LICENSE"]
        )
    print("✅ TapVid 全量数据集下载完成")

    # ========== 3. Dynamic Replica 数据集 (动态场景评估) ==========
    dynamic_replica_dir = os.path.join(root_dir, "dynamic_replica")
    if skip_existing and os.path.exists(dynamic_replica_dir):
        print("✅ Dynamic Replica 数据集已存在，跳过下载")
    else:
        print("📥 正在下载 Dynamic Replica 数据集 (动态场景评估)...")
        # 官方下载链接 + 断点续传
        dr_url = "https://dl.fbaipublicfiles.com/cotracker/dynamic_replica.tar.gz"
        dr_tar = os.path.join(root_dir, "dynamic_replica.tar.gz")
        
        # 使用wget断点续传下载
        subprocess.run(
            ["wget", "-c", dr_url, "-O", dr_tar],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 解压
        print("📦 正在解压 Dynamic Replica 数据集...")
        with tarfile.open(dr_tar, "r:gz") as tar:
            tar.extractall(dynamic_replica_dir)
        os.remove(dr_tar)  # 删除压缩包节省空间
        print("✅ Dynamic Replica 数据集下载&解压完成")

    # ========== 4. Real Data 真实场景数据集 (可选训练) ==========
    real_data_dir = os.path.join(root_dir, "real_data")
    if skip_existing and os.path.exists(real_data_dir):
        print("✅ Real Data 数据集已存在，跳过下载")
    else:
        print("📥 正在下载 Real Data 真实场景数据集 (可选训练)...")
        real_data_url = "https://dl.fbaipublicfiles.com/cotracker/real_data.zip"
        real_data_zip = os.path.join(root_dir, "real_data.zip")
        
        subprocess.run(
            ["wget", "-c", real_data_url, "-O", real_data_zip],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 解压
        print("📦 正在解压 Real Data 数据集...")
        with zipfile.ZipFile(real_data_zip, "r") as zf:
            zf.extractall(real_data_dir)
        os.remove(real_data_zip)
        print("✅ Real Data 数据集下载&解压完成")

    # ========== 数据集路径汇总 ==========
    print("\n" + "="*50)
    print("📋 CoTracker 数据集下载完成！目录结构：")
    print(f"  根目录: {root_dir}")
    print(f"  - Kubric 训练集: {kubric_dir}")
    print(f"  - TapVid 评估集: {tapvid_dir}")
    print(f"  - Dynamic Replica 评估集: {dynamic_replica_dir}")
    print(f"  - Real Data 训练集: {real_data_dir}")
    print("="*50)

if __name__ == "__main__":
    # 执行下载（默认跳过已存在的数据集）
    download_cotracker_datasets(
        root_dir="./cotracker_datasets",
        skip_existing=True
    )