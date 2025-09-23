#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-epoch data mixing driver for Ultralytics YOLOv11:
- Keeps 70% real + 30% synthetic for train/val/test each epoch
- Regenerates synthetic split-wise every epoch
- No change to YOLO source code required

Requires: pip install ultralytics pillow numpy
"""
import os, sys, json, math, shutil, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import yaml
import torch
from utils.data_generator import GenConfig, generate_dataset
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT)) 

# User settings (EDIT THESE)
REAL_ROOT = PROJECT_ROOT / "real"                # your real YOLO dataset root
ASSETS_DIR = PROJECT_ROOT / "assets"             # generator assets root
OUT_BASE  = PROJECT_ROOT / "out_epoch"           # synthetic output base
MODEL_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"      # or your checkpoint
EPOCHS = 20
IMGSZ = 640
BATCH = 16
DEVICE = "0"                        # -1 for CPU; "0,1" for multi-GPU
MIX_VALTEST = False
REAL_FRACTION = 0.70
SYN_FRACTION  = 0.30

#GEN_SCRIPT = Path("data_generator.py")
IMAGE_SIZE = (1280, 720)
MIN_OBJS, MAX_OBJS = 1, 4
CLASS_RATIOS = {}
PER_CLASS_MINMAX = {}
ALLOW_OVERLAP = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def list_pairs(root: Path, split: str) -> List[Path]:
    img_dir = root/"images"/split
    lbl_dir = root/"labels"/split
    imgs = {}
    for p in img_dir.glob("*.*"):
        if p.suffix.lower() in IMG_EXTS:
         imgs.setdefault(p.stem, p)
    pairs = []
    for l in sorted(lbl_dir.glob("*.txt")):
        if l.stem in imgs:
            pairs.append(imgs[l.stem])
    return sorted(pairs)

def ensure_import(gen_script: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("synth_gen_plus", str(gen_script.resolve()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["synth_gen_plus"] = mod
    spec.loader.exec_module(mod)
    return mod

def synth_needed(n_real: int) -> int:
    return max(0, int(round(n_real * SYN_FRACTION / max(1e-8, REAL_FRACTION))))

def write_list(paths, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.resolve()).replace("\\","/") + "\n")

def load_names(real_root: Path):
    y = real_root/"dataset.yaml"
    if y.exists():
        import re
        txt = y.read_text(encoding="utf-8")
        m = re.search(r"names:\s*(\[.*?\])", txt, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    cj = real_root/"classes.json"
    if cj.exists():
        mp = json.loads(cj.read_text(encoding="utf-8"))
        keys = sorted(map(int, mp.keys()))
        return [mp[str(k)] if str(k) in mp else mp[k] for k in keys]
    raise RuntimeError("Cannot determine class names from real dataset.")

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root",   type=str, default=str(REAL_ROOT))
    ap.add_argument("--assets_dir",  type=str, default=str(ASSETS_DIR))
    ap.add_argument("--out_base",    type=str, default=str(OUT_BASE))
    ap.add_argument("--weights",     type=str, default=str(MODEL_WEIGHTS))
    ap.add_argument("--device",      type=str, default=str(DEVICE))
    ap.add_argument("--mix_valtest", action="store_true", help="mix generated data for test&val ")
    return ap.parse_args()

def main():
    args = parse_args()
    
    global REAL_ROOT, ASSETS_DIR, OUT_BASE, MODEL_WEIGHTS, DEVICE, MIX_VALTEST
    REAL_ROOT     = Path(args.real_root)
    ASSETS_DIR    = Path(args.assets_dir)
    OUT_BASE      = Path(args.out_base)
    MODEL_WEIGHTS = args.weights
    DEVICE        = args.device
    MIX_VALTEST   = bool(args.mix_valtest)

    from ultralytics import YOLO
    names = load_names(REAL_ROOT)
    #gen = ensure_import(GEN_SCRIPT)

    splits = ["train", "val", "test"]
    real_lists = {sp: list_pairs(REAL_ROOT, sp) for sp in splits}
    for sp in splits:
        if not real_lists[sp]:
            raise RuntimeError(f"No real data found in {REAL_ROOT}/images/{sp}")
    base_seed = 2025
    set_global_seed(base_seed)

    model = YOLO(MODEL_WEIGHTS)
    workdir = Path("epoch_work"); workdir.mkdir(exist_ok=True)

    for ep in range(EPOCHS):
        print(f"\n========== Epoch {ep+1}/{EPOCHS} ==========")
        set_global_seed(base_seed + ep)
        ep_out = OUT_BASE / f"ep_{ep:03d}"
        if ep_out.exists():
            shutil.rmtree(ep_out)
        ep_out.mkdir(parents=True, exist_ok=True)

        R = {sp: len(real_lists[sp]) for sp in splits}
        S = {sp: synth_needed(R[sp]) for sp in splits}
        print(f"[counts] real={R} -> synth={S}")

        mix = {}
        for sp in splits:
            seed_offset = {"train": 0, "val": 100_000, "test": 200_000}[sp]
            R = len(real_lists[sp])
            S = synth_needed(R)
            print(f"[{sp}] real={R} -> synth={S}")
            cfg = GenConfig(
                assets_dir=str(ASSETS_DIR),
                out_dir=str(ep_out),
                image_size=IMAGE_SIZE,
                train_count=S if sp=='train' else 0,
                val_count=S if sp=='val' else 0,
                test_count=S if sp=='test' else 0,
                min_objects_per_image=MIN_OBJS,
                max_objects_per_image=MAX_OBJS,
                class_ratios=CLASS_RATIOS,
                per_class_min_max=PER_CLASS_MINMAX,
                allow_overlap=ALLOW_OVERLAP,
                yaml_abs=True,
                seed=base_seed + ep,
            )
            generate_dataset(cfg)
            synth_dir = ep_out/"images"/sp
            synth_imgs = sorted(synth_dir.glob("*.*")) if synth_dir.exists() else []
            mixed = list(real_lists[sp]) + synth_imgs
            mix[sp] = mixed

        # Write file lists & yaml
        tl = workdir/f"train_ep{ep:03d}.txt"
        vl = workdir/f"val_ep{ep:03d}.txt"
        te = workdir/f"test_ep{ep:03d}.txt"
        write_list(mix["train"], tl); write_list(mix["val"], vl); write_list(mix["test"], te)

        yaml = workdir/f"dataset_ep{ep:03d}.yaml"
        yaml.write_text(
            "train: "+str(tl.resolve()).replace("\\","/")+"\n"+
            "val: "+str(vl.resolve()).replace("\\","/")+"\n"+
            "test: "+str(te.resolve()).replace("\\","/")+"\n"+
            f"nc: {len(names)}\n"+
            f"names: {json.dumps(names, ensure_ascii=False)}\n",
            encoding="utf-8"
        )
        yaml_path.write_text(yaml.safe_dump(yaml_content, allow_unicode=True, sort_keys=False), encoding="utf-8")
        model.train(
            data=str(yaml),
            epochs=1,
            imgsz=IMGSZ,
            batch=BATCH,
            device=str(DEVICE),
            resume=False,                
            project="runs/mix",
            name="exp",
            exist_ok=True,
            verbose=True,
        )

    print("Done. Check runs/detect/train for weights and metrics.")
    print(f"(Lists/YAML per epoch are under {workdir.as_posix()})")

if __name__=="__main__":
    main()
