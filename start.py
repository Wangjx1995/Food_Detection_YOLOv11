#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, subprocess, argparse
from pathlib import Path

# Matplotlib headless & cache dir
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path("/tmp/mpl").mkdir(exist_ok=True)

# 禁用 user-site，避免 ~/.local 里的旧包被加载
os.environ["PYTHONNOUSERSITE"] = "1"

def run(cmd: str, check: bool = True, cwd: str | None = None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)

# 安全的多行 Python here-doc 执行助手（避免 \n 被当作字面量）
def run_py(code: str, env: str = "", cwd: str | None = None, check: bool = True):
    prefix = (env + " ") if env else ""
    run(prefix + "python - <<'PY'\n" + code + "\nPY", check=check, cwd=cwd)

def run_real(args):
    """仅触发一次全真实训练：import 调用 src/train.run(args)。"""
    from argparse import Namespace

    repo_root = Path(args.repo_dir).resolve()
    # 让仓库根优先
    sys.path.insert(0, repo_root.as_posix())

    # 如果 vendored 在 src/ 下，也让 src/ 优先
    if (repo_root / "src" / "ultralytics" / "__init__.py").exists():
        sys.path.insert(0, (repo_root / "src").as_posix())
        os.environ["PYTHONPATH"] = f"{(repo_root/'src').as_posix()}:{os.environ.get('PYTHONPATH','')}"

    # 自检：实际导入 ultralytics 来源
    run_py(
        "import ultralytics\n"
        "print('✅ ultralytics from:', ultralytics.__file__)\n"
        "try:\n"
        "  from ultralytics import __version__ as V\n"
        "  print('   version:', V)\n"
        "except Exception:\n"
        "  pass\n",
        cwd=repo_root.as_posix()
    )

    # 数据集路径解析
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

    # 确保包标记存在（幂等）
    (repo_root / "src").mkdir(exist_ok=True)
    (repo_root / "src" / "__init__.py").touch()
    (repo_root / "utils").mkdir(exist_ok=True)
    (repo_root / "utils" / "__init__.py").touch()

    # 子进程的 PYTHONPATH：仓库根优先
    os.environ["PYTHONPATH"] = f"{repo_root.as_posix()}:{os.environ.get('PYTHONPATH','')}"
    # 如果 vendored 在 src/ 下，把 src/ 也加进去（优先）
    if (repo_root / "src" / "ultralytics" / "__init__.py").exists():
        os.environ["PYTHONPATH"] = f"{(repo_root/'src').as_posix()}:{os.environ.get('PYTHONPATH','')}"

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

    # 预检：路径与包来源
    run_py(
        "import os, sys\n"
        "print('CWD =', os.getcwd())\n"
        "print('sys.path[0] =', sys.path[0])\n"
        "print('PYTHONPATH =', os.environ.get('PYTHONPATH'))\n"
        "import src, utils\n"
        "print('src =', getattr(src,'__file__',src))\n"
        "print('utils =', getattr(utils,'__file__',utils))\n"
        "import ultralytics\n"
        "print('✅ ultralytics from:', ultralytics.__file__)\n",
        cwd=repo_root.as_posix()
    )

    mix_cmd = [
        "python", "-m", "src.train_mix",
        "--real_root",  real_root,
        "--assets_dir", assets_dir,
        "--out_base",   out_base,
        "--weights",    weights,
        "--device",     str(args.device),
    ]
    if args.mix_valtest:
        mix_cmd.append("--mix_valtest")

    try:
        run(" ".join(mix_cmd), cwd=repo_root.as_posix())
    except SystemExit as e:
        print(f"⚠️ Module run failed (exit={e.code}), fallback to runpy path-run ...")
        argv = [
            "--real_root",  real_root,
            "--assets_dir", assets_dir,
            "--out_base",   out_base,
            "--weights",    weights,
            "--device",     str(args.device),
        ]
        if args.mix_valtest:
            argv.append("--mix_valtest")

        fallback = (
            "python - <<'PY'\n"
            "import os, sys, runpy\n"
            f"repo = r'''{repo_root.as_posix()}'''\n"
            "sys.path.insert(0, repo)\n"
            "os.chdir(repo)\n"
            f"sys.argv = ['src/train_mix.py'] + {argv!r}\n"
            "runpy.run_path(os.path.join(repo, 'src', 'train_mix.py'), run_name='__main__')\n"
            "PY"
        )
        run(fallback, cwd=repo_root.as_posix())

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

    # 准备 Colab 工作目录
    Path("/content").mkdir(exist_ok=True)
    os.chdir("/content")

    # Drive
    if not args.skip_drive:
        try:
            from google.colab import drive
            drive.mount(args.drive_mount, force_remount=False)
            print(f"✅ Drive mounted at: {args.drive_mount}")
        except Exception:
            print("ℹ️ 非 Colab 或子进程：如需 Drive，请先在 Notebook 里 drive.mount('/content/drive')")

    # 拉仓库
    run(f"rm -rf '{args.repo_dir}'", check=False)
    clone_cmd = f"git clone -vv {args.repo_url} '{args.repo_dir}'"
    if args.branch:
        clone_cmd = f"git clone -vv --branch {args.branch} {args.repo_url} '{args.repo_dir}'"
    run(clone_cmd)

    # pip 基础
    run("python -m pip install -U pip")

    # requirements（可跳过）
    if not args.no_requirements:
        run(f"python -m pip install --no-cache-dir --upgrade --force-reinstall -r '{args.repo_dir}/requirements.txt'")

    # ========= 数值栈：彻底清理 + 固定版本重装 =========
    # 先卸载（忽略失败）
    run("python -m pip uninstall -y numpy scipy matplotlib ultralytics", check=False)

    # 物理清理：site-packages + user-site + sysconfig 的 purelib/platlib
    run_py(
        "import site, sysconfig, shutil, os, glob\n"
        "dirs = set()\n"
        "dirs.update(site.getsitepackages())\n"
        "try:\n"
        "    dirs.add(site.getusersitepackages())\n"
        "except Exception:\n"
        "    pass\n"
        "paths = sysconfig.get_paths()\n"
        "for k in ('purelib','platlib'):\n"
        "    p = paths.get(k)\n"
        "    if p: dirs.add(p)\n"
        "print('🔧 purge dirs:', dirs)\n"
        "PATTERNS = ('numpy*','scipy*','matplotlib*')\n"
        "for sp in sorted(dirs):\n"
        "    for pat in PATTERNS:\n"
        "        for p in glob.glob(os.path.join(sp, pat)):\n"
        "            print('Removing', p); shutil.rmtree(p, ignore_errors=True)\n"
    )

    # 固定版本重装（wheel-only，避免本地编译；不解析依赖链以免顶掉固定版本）
    run(
        "python -m pip install --only-binary=:all: --no-cache-dir --upgrade --force-reinstall --no-deps "
        "numpy==2.1.2 scipy==1.14.1 matplotlib==3.9.2"
    )

    # 检测是否 vendored ultralytics（项目根或 src/ 下均支持）
    repo_root = Path(args.repo_dir)
    vendored_parent = None
    for cand in [repo_root / "ultralytics", repo_root / "src" / "ultralytics"]:
        if (cand / "__init__.py").exists():
            vendored_parent = cand.parent  # 项目根 或 项目根/src
            break

    # 安装 ultralytics / 依赖（不关闭绘图）
    if vendored_parent:
        print(f"🔒 Detected vendored ultralytics at: {(vendored_parent/'ultralytics').as_posix()} (skip pip ultralytics)")
        run("python -m pip install -U pillow pyyaml", check=False)
        # 让后续子进程优先从本地导入
        os.environ["PYTHONPATH"] = f"{vendored_parent.as_posix()}:{os.environ.get('PYTHONPATH','')}"
    else:
        # 装官方包时使用 --no-deps，避免它把 numpy/scipy 又升级
        run("python -m pip install -U --no-deps ultralytics", check=False)
        run("python -m pip install -U pillow pyyaml", check=False)

    # 数值栈健康自检（包含 ndimage）——⚠️ 这里用 run_py，是真换行
    run_py(
        "import numpy, scipy, matplotlib, inspect\n"
        "from scipy.ndimage import gaussian_filter1d\n"
        "print(f\"NumPy {numpy.__version__} | SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__} - ndimage OK\")\n"
        "print('numpy at:', inspect.getfile(numpy))\n"
        "print('scipy at:', inspect.getfile(scipy))\n"
        "print('mpl   at:', inspect.getfile(matplotlib))\n",
        env="MPLBACKEND=Agg"
    )

    # 额外：打印将要使用的 ultralytics 来源（在仓库根执行以命中本地）
    run_py(
        "import ultralytics\n"
        "print('✅ ultralytics from:', ultralytics.__file__)\n"
        "try:\n"
        "  from ultralytics import __version__ as V\n"
        "  print('   version:', V)\n"
        "except Exception:\n"
        "  pass\n",
        cwd=repo_root.as_posix()
    )
    # ========= /数值栈 =========

    # 分支执行
    if args.mode == "real":
        run_real(args)
    else:
        run_mixed(args)

    print("\n✅ All done.")

if __name__ == "__main__":
    main()