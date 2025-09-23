import argparse, yaml
from ultralytics import YOLO
from pathlib import Path
import torch
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def _resolve(p: str | Path) -> str:
    p = Path(p)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p.as_posix()

def _device_str(cfg_val) -> str:
    
    if not torch.cuda.is_available():
        return "cpu"

    
    if cfg_val is None:
        return "0"  
    if isinstance(cfg_val, int):
        return "cpu" if cfg_val < 0 else str(cfg_val)

    s = str(cfg_val).strip().lower()
    if s in {"-1", "cpu"}:
        return "cpu"
    return s  

def run(args):
    cfg_path  = _resolve(args.cfg)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_yaml = _resolve(args.data)
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")

    model_path = _resolve(cfg.get("model", PROJECT_ROOT / "yolo11n.pt"))
    if not Path(model_path).exists():
        print(f"⚠️ weights not found at {model_path} (Ultralytics may download a default model)")


    dev = _device_str(getattr(args, "device", None) or cfg.get("device", None)) 

    print("✅ REAL training (Ultralytics)")
    print(f"   • data   = {data_yaml}")
    print(f"   • cfg    = {cfg_path}")
    print(f"   • model  = {model_path}")
    print(f"   • device = {dev}")
    print(f"   • imgsz/batch/epochs = {cfg.get('imgsz', 512)}/{cfg.get('batch', 16)}/{cfg.get('epochs', 15)}")


    seed = int(cfg.get("seed", 42))
    try:
      import random, numpy as np
      random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
      if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        imgsz=int(cfg.get("imgsz", 512)),
        epochs=int(cfg.get("epochs", 15)),
        batch=int(cfg.get("batch", 16)),
        patience=int(cfg.get("patience", 5)),
        workers=int(cfg.get("workers", 2)),
        device=dev,                                
        project=str(cfg.get("project", "runs")),
        name=str(cfg.get("name", "train")),
        cache=cfg.get("cache", "ram"),
        plots=bool(cfg.get("plots", True)),
        seed=seed,
        exist_ok=True,                             
        verbose=True,
    )
    print("🏁 REAL training finished. Saved to:", results.save_dir)
    return results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=(PROJECT_ROOT/"real/dataset.yaml").as_posix(), help="dataset yaml")
    ap.add_argument("--cfg",  default=(PROJECT_ROOT/"configs/train.yaml").as_posix(), help="training config yaml")
    ap.add_argument("--device", default=None, help="CPU or GPU NUM")
    args = ap.parse_args()
    run(args)
if __name__ == "__main__":
    main()