#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, subprocess, argparse
from pathlib import Path

def run(cmd: str, check: bool = True, cwd: str | None = None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)

def run_real(args):
    """仅触发一次全真实训练：import 调用 src/train.run(args)。"""
    from argparse import Namespace
    from pathlib import Path

    repo_root = Path(args.repo_dir).resolve()
    sys.path.insert(0, repo_root.as_posix())  

    
    if args.data:
        dataset_yaml = Path(args.data)
    else:
        if args.real_drive:
            real_link = repo_root / "real"
            
            if real_link.is_symlink():
                real_link.unlink()
            elif real_link.exists():
                import shutil
                shutil.rmtree(real_link)
            os.symlink(args.real_drive, real_link, target_is_directory=True)
            dataset_yaml = real_link / "dataset.yaml"
        else:
            dataset_yaml = repo_root / "real" / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml}")

    cfg_path = (repo_root / "configs" / "train.yaml").as_posix()

    print("✅ Entered REAL mode")
    print(f"   • data = {dataset_yaml.as_posix()}")
    print(f"   • cfg  = {cfg_path}")

    
    from src.train import run as train_run
    train_run(Namespace(data=dataset_yaml.as_posix(), cfg=cfg_path))

    print("🏁 REAL training finished.")

def run_mixed(args):
    
    repo_root = Path(args.repo_dir).resolve()
    real_root  = args.real_root  or (repo_root / "real").as_posix()
    assets_dir = args.assets_dir or (repo_root / "assets").as_posix()
    out_base   = args.out_base   or (repo_root / "out_epoch").as_posix()
    weights    = args.weights    or (repo_root / "yolo11n.pt").as_posix()

    print("✅ Entered MIXED mode")
    print(f"   • real_root  = {real_root}")
    print(f"   • assets_dir = {assets_dir}")
    print(f"   • out_base   = {out_base}")
    print(f"   • weights    = {weights}")
    print(f"   • device     = {args.device}")
    if args.mix_valtest:
        print("   • mix_valtest = True")

    mix_cmd = [
        "python", "src/train_mix.py",
        "--real_root",  real_root,
        "--assets_dir", assets_dir,
        "--out_base",   out_base,
        "--weights",    weights,
        "--device",     str(args.device),
    ]
    if args.mix_valtest:
        mix_cmd.append("--mix_valtest")

    run(" ".join(mix_cmd), cwd=repo_root.as_posix())
    print("🏁 MIXED training finished.")
    print("   • Weights & metrics: runs/mix/exp*")
    print("   • Per-epoch lists & YAML: epoch_work/")

def main():
    ap = argparse.ArgumentParser(description="Colab starter for Object_Detection_Tutorial")
    # BASIC
    ap.add_argument("--mode", choices=["real", "mixed"], default="real")
    ap.add_argument("--skip_drive", action="store_true", help="不挂载 Google Drive")
    ap.add_argument("--drive_mount", default="/content/drive", help="Drive 挂载点")
    ap.add_argument("--repo_url", default="https://github.com/Wangjx1995/Object_Detection_Tutorial.git")
    ap.add_argument("--repo_dir", default="/content/Object_Detection_Tutorial")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--no_requirements", action="store_true",
                    help="跳过安装 requirements.txt（已手动对齐 numpy/matplotlib 时很有用）")

    # REAL
    ap.add_argument("--data", "--dataset_yaml", dest="data", default=None,
                    help="真实数据集 dataset.yaml 的绝对路径（优先级最高）")
    ap.add_argument("--real_drive", default=None,
                    help="真实数据根目录（含 images/labels/dataset.yaml）。若提供，将软链为 repo_dir/real/")

    # MIXED
    ap.add_argument("--real_root",  default=None)
    ap.add_argument("--assets_dir", default=None)
    ap.add_argument("--out_base",   default=None)
    ap.add_argument("--weights",    default=None)
    ap.add_argument("--device",     default="0")
    ap.add_argument("--mix_valtest", action="store_true")

    args = ap.parse_args()

    
    Path("/content").mkdir(exist_ok=True)
    os.chdir("/content")

    if not args.skip_drive:
        try:
            from google.colab import drive
            drive.mount(args.drive_mount, force_remount=False)
            print(f"✅ Drive mounted at: {args.drive_mount}")
        except Exception:
            print("ℹ️ 非 Colab 或子进程：如需 Drive，请先在 Notebook 里 drive.mount('/content/drive')")

    
    run(f"rm -rf '{args.repo_dir}'", check=False)
    clone_cmd = f"git clone -vv {args.repo_url} '{args.repo_dir}'"
    if args.branch:
        clone_cmd = f"git clone -vv --branch {args.branch} {args.repo_url} '{args.repo_dir}'"
    run(clone_cmd)

    
    run("python -m pip install -U pip")
    if not args.no_requirements:
        run(f"python -m pip install --no-cache-dir --upgrade --force-reinstall -r '{args.repo_dir}/requirements.txt'")
    run("python -m pip install -U ultralytics pillow pyyaml", check=False)
    run("python -m pip uninstall -y numpy matplotlib scipy", check=False)
    run("python -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps numpy==2.1.2 matplotlib==3.9.2 scipy==1.14.1", check=True)
    run("python -c \"import numpy,scipy,matplotlib; print('NumPy',numpy.__version__,'SciPy',scipy.__version__,'Matplotlib',matplotlib.__version__)\"")

    
    if args.mode == "real":
        run_real(args)
    else:
        run_mixed(args)

    print("\n✅ All done.")

if __name__ == "__main__":
    main()
