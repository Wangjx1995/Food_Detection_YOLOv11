#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, subprocess, argparse
from pathlib import Path
import sys

DRIVE_MOUNT = "/content/drive"
REPO_URL    = "https://github.com/Wangjx1995/Object_Detection_Tutorial.git"
REPO_DIR    = "/content/Object_Detection_Tutorial"
REAL_DRIVE = "/content/drive/MyDrive/odt_data/real"
REAL_LINK  = f"{REPO_DIR}/real"
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
# ===== CLI =====
p = argparse.ArgumentParser(description="Colab starter for Object_Detection_Tutorial")
p.add_argument("--mode", choices=["real", "mix"], default="real",
               help="训练模式：real=全真实数据；mix=真实+生成数据（调用 train_mix.py）")
p.add_argument("--dataset_yaml", default="/content/Object_Detection_Tutorial/configs/dataset.yaml",
               help="仅在 --mode real 时使用：真实数据集的 dataset.yaml 路径")
p.add_argument("--skip_drive", action="store_true",
               help="不挂载 Google Drive（在纯本地/容器环境下用）")
p.add_argument("--branch", default=None, help="可选：git 指定分支")
p.add_argument("--no_requirements", action="store_true",
               help="跳过安装 requirements.txt")
p.add_argument("--real_drive", default=None,
               help="Drive 上真实数据目录（含 images/labels/dataset.yaml），将链接到仓库 real/ 下。")               

args = p.parse_args()

def run(cmd, check=True, cwd=None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)

def run_real(args):
    from argparse import Namespace
    from pathlib import Path
    import os, shutil
    from src.train import run as train_run

    print("✅ Entered REAL mode")

    repo_root = Path(getattr(args, "repo_dir", "/content/Object_Detection_Tutorial"))

    
    if getattr(args, "data", None):
        dataset_yaml = Path(args.data)

    else:
        
        if getattr(args, "real_drive", None):
            real_link = repo_root / "real"
            if real_link.is_symlink():
                real_link.unlink()
            elif real_link.exists():
                shutil.rmtree(real_link)
            os.symlink(args.real_drive, real_link, target_is_directory=True)
            dataset_yaml = real_link / "dataset.yaml"
        else:
            
            dataset_yaml = repo_root / "real" / "dataset.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml}")

    cfg_path = (repo_root / "configs" / "train.yaml").as_posix()
    train_args = Namespace(data=dataset_yaml.as_posix(), cfg=cfg_path)

    print(f"   • data = {train_args.data}")
    print(f"   • cfg  = {train_args.cfg}")
    train_run(train_args)
    print("🏁 REAL training finished.")


def run_mixed(args):

    from src.train_mix import main as run_mix
    print("✅ Entered MIXED mode")
    print(f"   • real_root   = {args.real_root}")
    print(f"   • assets_dir  = {args.assets_dir}")
    print(f"   • out_base    = {args.out_base}")
    print(f"   • weights     = {args.weights}")
    print(f"   • device      = {args.device}")
    try:

        sys_argv_backup = list(sys.argv)
        sys.argv = [
            sys.argv[0],
            "--real_root",   args.real_root,
            "--assets_dir",  args.assets_dir,
            "--out_base",    args.out_base,
            "--weights",     args.weights,
            "--device",      args.device,
        ] + (["--mix_valtest"] if args.mix_valtest else [])
        run_mix()  
    finally:
        sys.argv = sys_argv_backup
    print("🏁 MIXED training finished.")
    print("   • Check weights & metrics under: runs/mix/exp*")
    print("   • Per-epoch lists & YAML under: epoch_work/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real","mixed"], default="real")
    ap.add_argument("--data", "--dataset_yaml", dest="data", default=None,help="Path to dataset.yaml for REAL mode (overrides --real_root)")
    ap.add_argument("--repo_dir", default="/content/Object_Detection_Tutorial",help="repo clone dir")
    ap.add_argument("--real_drive", default=None, help="Drive data dir")
    ap.add_argument("--real_root",  default=str((PROJECT_ROOT/"real").resolve()))
    ap.add_argument("--assets_dir", default=str((PROJECT_ROOT/"assets").resolve()))
    ap.add_argument("--out_base",   default=str((PROJECT_ROOT/"out_epoch").resolve()))
    ap.add_argument("--weights",    default=str((PROJECT_ROOT/"yolo11n.pt").resolve()))
    ap.add_argument("--device",     default="0")
    ap.add_argument("--mix_valtest", action="store_true")
    args = ap.parse_args()

    if args.mode == "mixed":
        run_mixed(args)
    else:
        run_real(args)


Path("/content").mkdir(exist_ok=True)
os.chdir("/content")


if not args.skip_drive:
    try:
        from google.colab import drive
        drive.mount(DRIVE_MOUNT, force_remount=False)
    except Exception:
        print("If running in a subprocess, please first run drive.mount('/content/drive') in a notebook cell.")


run(f"rm -rf '{REPO_DIR}'", check=False)
clone_cmd = f"git clone -vv {REPO_URL} '{REPO_DIR}'"
if args.branch:
    clone_cmd = f"git clone -vv --branch {args.branch} {REPO_URL} '{REPO_DIR}'"
run(clone_cmd)


run("python -m pip install -U pip")
if not args.no_requirements:
    run(f"python -m pip install -r '{REPO_DIR}/requirements.txt'")

run("python -m pip install -U ultralytics pillow numpy", check=False)


if args.mode == "real":
    if getattr(args, "real_drive", None):
        REAL_LINK = f"{REPO_DIR}/real"
        run(f"[ -L '{REAL_LINK}' ] && unlink '{REAL_LINK}' || rm -rf '{REAL_LINK}'", check=False, cwd="/")
        run(f"ln -s '{args.real_drive}' '{REAL_LINK}'", cwd="/")
        DATASET_YAML = "real/dataset.yaml"           
    else:
        DATASET_YAML = args.dataset_yaml

    TRAIN_CMD = f"python src/train.py --data '{DATASET_YAML}'"
    run(TRAIN_CMD, cwd=REPO_DIR)

elif args.mode == "mix":
    MIX_CMD = "python train_mix.py"
    run(MIX_CMD, cwd=REPO_DIR)

print("\n✅ All done.")

if __name__ == "__main__":
    main()