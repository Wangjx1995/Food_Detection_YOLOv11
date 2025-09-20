#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgraded Synthetic dataset generator for YOLO (labels in YOLO txt).

Features:
1) Class appearance control: class_ratios + per-class (min,max) per image
2) Visual realism: drop shadow, contact shadow, mild perspective jitter, shallow depth-of-field (background blur)
3) dataset.yaml auto-write; optional absolute paths (--yaml_abs)
4) Exact split counts via --train_count/--val_count/--test_count (override ratios)

Input tree:
  assets/
    objects/<class_name>/*.png  (RGBA cutouts)
    backgrounds/*.(jpg|png|jpeg)

Output tree:
  out_dir/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
    classes.json
    stats.json
    dataset.yaml

Dependencies: Pillow, numpy
"""
import os, glob, json, time, random, argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

@dataclass
class GenConfig:
    assets_dir: str = "assets"
    out_dir: str = "out"
    image_size: Tuple[int,int] = (1280,720)
    num_images: int = 200
    splits: Tuple[float,float,float] = (0.8,0.1,0.1)
    train_count: Optional[int] = None
    val_count: Optional[int] = None
    test_count: Optional[int] = None

    min_objects_per_image: int = 1
    max_objects_per_image: int = 5

    class_ratios: Dict[str, float] = field(default_factory=dict)
    per_class_min_max: Dict[str, Tuple[int,int]] = field(default_factory=dict)

    allow_overlap: bool = True
    max_iou_allowed: float = 0.5
    max_place_trials: int = 50

    scale_range: Tuple[float,float] = (0.08,0.35)
    angle_range: Tuple[float,float] = (-25,25)
    prob_hflip: float = 0.5

    brightness: Tuple[float,float] = (0.85,1.2)
    contrast: Tuple[float,float] = (0.85,1.2)
    color: Tuple[float,float] = (0.9,1.1)
    sharpness: Tuple[float,float] = (0.85,1.2)
    prob_blur: float = 0.25
    blur_radius: Tuple[float,float] = (0.3,1.2)
    prob_noise: float = 0.25
    noise_sigma: Tuple[float,float] = (4.0,12.0)

    allow_truncation: bool = True
    min_box_pixels: int = 10

    enable_drop_shadow: bool = True
    shadow_opacity: float = 0.35
    shadow_offset_px: Tuple[int,int] = (12,10)
    shadow_blur_radius: float = 12.0

    enable_contact_shadow: bool = True
    contact_shadow_opacity: float = 0.35
    contact_shadow_scale: Tuple[float,float] = (1.0,0.25)
    contact_shadow_blur: float = 18.0
    contact_shadow_offset_y: int = 4

    enable_perspective: bool = True
    persp_max_jitter_ratio: float = 0.06

    enable_shallow_dof: bool = True
    bg_blur_sigma_range: Tuple[float,float] = (0.0,1.6)

    yaml_abs: bool = False
    seed: int = 2025

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def list_imgs(d):
    exts=('*.jpg','*.png','*.jpeg','*.JPG','*.PNG','*.JPEG')
    out=[]
    for e in exts: out+=glob.glob(os.path.join(d,e))
    return sorted(out)
def list_pngs(d):
    exts=('*.png','*.PNG')
    out=[]
    for e in exts: out+=glob.glob(os.path.join(d,e))
    return sorted(out)
def randf(a,b): return a+(b-a)*random.random()

def load_objects(root):
    class_map={}
    images_map={}
    class_names=sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root,x))])
    if not class_names:
        raise RuntimeError(f"No class folders under {root}")
    cid=0
    for cname in class_names:
        paths=list_pngs(os.path.join(root,cname))
        if not paths: continue
        imgs=[]
        for p in paths:
            im=Image.open(p).convert('RGBA')
            if im.getbands()[-1] != 'A':
                rgba=Image.new('RGBA', im.size, (0,0,0,0))
                rgba.paste(im,(0,0))
                im=rgba
            imgs.append(im)
        if not imgs: continue
        class_map[cid]=cname
        images_map[cid]=imgs
        cid+=1
    if not class_map:
        raise RuntimeError("No object PNGs found.")
    return class_map, images_map

def apply_patch_augments(p, cfg):
    if random.random()<cfg.prob_hflip: p=p.transpose(Image.FLIP_LEFT_RIGHT)
    p=ImageEnhance.Brightness(p).enhance(randf(*cfg.brightness))
    p=ImageEnhance.Contrast(p).enhance(randf(*cfg.contrast))
    p=ImageEnhance.Color(p).enhance(randf(*cfg.color))
    p=ImageEnhance.Sharpness(p).enhance(randf(*cfg.sharpness))
    if random.random()<cfg.prob_blur: p=p.filter(ImageFilter.GaussianBlur(radius=randf(*cfg.blur_radius)))
    if random.random()<cfg.prob_noise:
        arr=np.array(p).astype(np.int16)
        rgb, a = arr[...,:3], arr[...,3:4]
        sigma=randf(*cfg.noise_sigma)
        noise=np.random.normal(0,sigma,rgb.shape).astype(np.int16)
        rgb=np.clip(rgb+noise,0,255).astype(np.uint8)
        arr=np.concatenate([rgb,a.astype(np.uint8)],axis=-1)
        p=Image.fromarray(arr,mode='RGBA')
    return p

def perspective_jitter(p, cfg):
    if not cfg.enable_perspective: return p
    w,h=p.size; j=cfg.persp_max_jitter_ratio
    src=[(0,0),(w,0),(w,h),(0,h)]
    dst=[(x+randf(-j,j)*w, y+randf(-j,j)*h) for (x,y) in src]
    import numpy as np
    def coeffs(pa,pb):
        M=[]
        for (x,y),(u,v) in zip(pa,pb):
            M+=[[x,y,1,0,0,0,-u*x,-u*y],
                [0,0,0,x,y,1,-v*x,-v*y]]
        A=np.array(M,dtype=np.float64)
        B=np.array([p[0] for p in pb]+[p[1] for p in pb],dtype=np.float64)
        return np.linalg.lstsq(A,B,rcond=None)[0]
    c=coeffs(src,dst)
    return p.transform((w,h), Image.PERSPECTIVE, data=c, resample=Image.BICUBIC)

def drop_shadow_from_alpha(p, offset, blur, opacity):
    w,h=p.size
    alpha=p.split()[-1]
    black=Image.new('RGBA',(w,h),(0,0,0,int(255*opacity)))
    sh=Image.new('RGBA',(w,h),(0,0,0,0))
    sh.paste(black,(0,0),mask=alpha)
    sh=sh.filter(ImageFilter.GaussianBlur(radius=blur))
    dx,dy=offset
    canvas=Image.new('RGBA',(w+abs(dx),h+abs(dy)),(0,0,0,0))
    ox=max(0,dx); oy=max(0,dy)
    canvas.alpha_composite(sh,(ox,oy))
    return canvas,(-ox,-oy)

def add_contact_shadow(canvas, bbox, cfg):
    if not cfg.enable_contact_shadow: return
    x0,y0,x1,y1=bbox; bw=max(1,x1-x0); bh=max(1,y1-y0)
    ew=int(bw*cfg.contact_shadow_scale[0]); eh=int(bh*cfg.contact_shadow_scale[1])
    cx=x0+bw//2; cy=y1+cfg.contact_shadow_offset_y
    ex0=int(cx-ew//2); ey0=int(cy-eh//2); ex1=ex0+ew; ey1=ey0+eh
    if ex1<=0 or ey1<=0 or ex0>=canvas.size[0] or ey0>=canvas.size[1]: return
    from PIL import ImageDraw
    ell=Image.new('L', canvas.size, 0); d=ImageDraw.Draw(ell)
    d.ellipse((ex0,ey0,ex1,ey1), fill=int(255*cfg.contact_shadow_opacity))
    ell=ell.filter(ImageFilter.GaussianBlur(radius=cfg.contact_shadow_blur))
    shade=Image.new('RGBA', canvas.size, (0,0,0,0)); shade.putalpha(ell)
    canvas.alpha_composite(shade)

def bbox_from_alpha(patch, xy, canvas_size):
    W,H=canvas_size
    alpha=np.array(patch.split()[-1])
    ys,xs=np.where(alpha>0)
    if len(xs)==0 or len(ys)==0: return None
    x0l,x1l=int(xs.min()), int(xs.max())
    y0l,y1l=int(ys.min()), int(ys.max())
    x0=xy[0]+x0l; y0=xy[1]+y0l; x1=xy[0]+x1l; y1=xy[1]+y1l
    if x1<0 or y1<0 or x0>=W or y0>=H: return None
    x0c=max(0,x0); y0c=max(0,y0); x1c=min(W-1,x1); y1c=min(H-1,y1)
    if x1c<=x0c or y1c<=y0c: return None
    return (x0c,y0c,x1c,y1c)

def iou(a,b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1])
    xB=min(a[2],b[2]); yB=min(a[3],b[3])
    interW=max(0,xB-xA); interH=max(0,yB-yA)
    inter=interW*interH
    if inter==0: return 0.0
    areaA=(a[2]-a[0])*(a[3]-a[1]); areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter+1e-9)

def yolo_line(cid, box, size):
    W,H=size; x0,y0,x1,y1=box
    w=x1-x0; h=y1-y0; cx=x0+w/2.0; cy=y0+w*0+ h/2.0
    return f"{cid} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}"

def plan_counts(class_names, cfg):
    total=random.randint(cfg.min_objects_per_image, cfg.max_objects_per_image)
    counts={c:0 for c in class_names}; rem=total
    for c in class_names:
        if c in cfg.per_class_min_max:
            mn,mx=cfg.per_class_min_max[c]
            counts[c]=max(0,mn); rem-=counts[c]
    if rem<0:
        for c in sorted(class_names, key=lambda x: cfg.per_class_min_max.get(x,(0,9999))[0], reverse=True):
            if rem>=0: break
            mn,_=cfg.per_class_min_max.get(c,(0,9999))
            if counts[c]>0:
                dec=min(counts[c], -rem); counts[c]-=dec; rem+=dec
    weights=[cfg.class_ratios.get(c,1.0) for c in class_names]
    s=sum(weights) or 1.0
    while rem>0:
        r=random.random()*s; acc=0.0; chosen=class_names[-1]
        for c,w in zip(class_names,weights):
            acc+=w
            if r<=acc: chosen=c; break
        mx=cfg.per_class_min_max.get(chosen,(0,9999))[1]
        if counts[chosen]<mx:
            counts[chosen]+=1; rem-=1
        else:
            found=False
            for alt in class_names:
                mx_alt=cfg.per_class_min_max.get(alt,(0,9999))[1]
                if counts[alt]<mx_alt:
                    counts[alt]+=1; rem-=1; found=True; break
            if not found: break
    return counts

def split_exact(cfg: GenConfig):
    if cfg.train_count is not None or cfg.val_count is not None or cfg.test_count is not None:
        t = cfg.train_count or 0
        v = cfg.val_count or 0
        e = cfg.test_count or 0
        return t, v, e
    tr = int(round(cfg.num_images * cfg.splits[0]))
    va = int(round(cfg.num_images * cfg.splits[1]))
    te = max(0, cfg.num_images - tr - va)
    return tr, va, te

def generate_dataset(cfg: GenConfig):
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    obj_root=os.path.join(cfg.assets_dir,'objects')
    bg_root=os.path.join(cfg.assets_dir,'backgrounds')
    out_images=os.path.join(cfg.out_dir,'images'); out_labels=os.path.join(cfg.out_dir,'labels')
    for sp in ['train','val','test']:
        ensure_dir(os.path.join(out_images,sp)); ensure_dir(os.path.join(out_labels,sp))
    class_map, images_map = load_objects(obj_root)
    bg_list=list_imgs(bg_root)
    if not bg_list: raise RuntimeError(f"No backgrounds under {bg_root}")

    with open(os.path.join(cfg.out_dir,'classes.json'),'w',encoding='utf-8') as f:
        json.dump(class_map,f,ensure_ascii=False,indent=2)

    ntr,nva,nte = split_exact(cfg)
    W,H=cfg.image_size
    name_by_id=class_map; id_by_name={v:k for k,v in name_by_id.items()}
    class_names=[name_by_id[k] for k in sorted(name_by_id.keys())]
    img_id=0

    for split_name, split_count in (('train',ntr),('val',nva),('test',nte)):
        for _ in range(split_count):
            bg_path=random.choice(bg_list)
            bg=Image.open(bg_path).convert('RGB').resize((W,H), Image.BICUBIC)
            if cfg.enable_shallow_dof:
                sigma=randf(*cfg.bg_blur_sigma_range)
                if sigma>0.05: bg=bg.filter(ImageFilter.GaussianBlur(radius=sigma))
            canvas=Image.new('RGBA',(W,H),(0,0,0,0)); canvas.paste(bg,(0,0))

            bboxes=[]; cls_ids=[]
            plan=plan_counts(class_names,cfg)
            for cname,count in plan.items():
                cid=id_by_name[cname]
                for _k in range(count):
                    src=random.choice(images_map[cid]).copy()
                    patch=src
                    short=min(W,H); target_short=max(2,int(short*randf(*cfg.scale_range)))
                    pw,ph=patch.size
                    if pw<ph: new_w=int(round(pw*target_short/float(ph))); new_h=target_short
                    else: new_w=target_short; new_h=int(round(ph*target_short/float(pw)))
                    patch=patch.resize((new_w,new_h), Image.BICUBIC)
                    patch=perspective_jitter(patch,cfg)
                    patch=patch.rotate(randf(*cfg.angle_range), resample=Image.BICUBIC, expand=True)
                    patch=apply_patch_augments(patch,cfg)
                    shadow_layer=None; sh_off=(0,0)
                    if cfg.enable_drop_shadow:
                        shadow_layer, sh_off = drop_shadow_from_alpha(patch, cfg.shadow_offset_px, cfg.shadow_blur_radius, cfg.shadow_opacity)
                    px,py=patch.size
                    x_min=-int(px*(0.35 if cfg.allow_truncation else 0.0))
                    y_min=-int(py*(0.35 if cfg.allow_truncation else 0.0))
                    x_max=W-int(px*(0.65 if cfg.allow_truncation else 1.0))
                    y_max=H-int(py*(0.65 if cfg.allow_truncation else 1.0))
                    placed=False
                    for _try in range(cfg.max_place_trials):
                        x=random.randint(x_min, max(x_min,x_max)) if x_max>=x_min else 0
                        y=random.randint(y_min, max(y_min,y_max)) if y_max>=y_min else 0
                        bbox=bbox_from_alpha(patch,(x,y),(W,H))
                        if bbox is None: continue
                        bw=bbox[2]-bbox[0]; bh=bbox[3]-bbox[1]
                        if bw<cfg.min_box_pixels or bh<cfg.min_box_pixels: continue
                        if not cfg.allow_overlap and any(iou(b,bbox)>cfg.max_iou_allowed for b in bboxes): 
                            continue
                        if shadow_layer is not None:
                            canvas.alpha_composite(shadow_layer,(x+sh_off[0], y+sh_off[1]))
                        canvas.alpha_composite(patch,(x,y))
                        if cfg.enable_contact_shadow: add_contact_shadow(canvas, bbox, cfg)
                        bboxes.append(bbox); cls_ids.append(cid); placed=True; break
                    if not placed: pass
            if not bboxes: continue
            final=canvas.convert('RGB')
            stem=f"{split_name}_{img_id:06d}"
            img_path=os.path.join(out_images,split_name,stem+'.jpg')
            lbl_path=os.path.join(out_labels,split_name,stem+'.txt')
            final.save(img_path, quality=90)
            with open(lbl_path,'w',encoding='utf-8') as f:
                for cid,box in zip(cls_ids,bboxes):
                    f.write(yolo_line(cid,box,(W,H))+'\n')
            img_id+=1

    stats={'classes': {int(k):v for k,v in class_map.items()}, 'total_images': img_id, 'config': cfg.__dict__,'timestamp': int(time.time())}
    with open(os.path.join(cfg.out_dir,'stats.json'),'w',encoding='utf-8') as f: json.dump(stats,f,ensure_ascii=False,indent=2)

    names=[class_map[k] for k in sorted(class_map.keys())]
    out_root = os.path.abspath(cfg.out_dir) if cfg.yaml_abs else cfg.out_dir
    def pjoin(*a): return os.path.as_posix(os.path.join(*a))
    yaml=(
        f"train: {pjoin(out_root,'images','train')}\n"
        f"val: {pjoin(out_root,'images','val')}\n"
        f"test: {pjoin(out_root,'images','test')}\n"
        f"nc: {len(names)}\n"
        f"names: {json.dumps(names, ensure_ascii=False)}\n"
    )
    with open(os.path.join(cfg.out_dir,'dataset.yaml'),'w',encoding='utf-8') as f: f.write(yaml)
    print('[DONE] images:', img_id)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assets_dir", type=str, default="assets")
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--image_width", type=int, default=1280)
    p.add_argument("--image_height", type=int, default=720)
    p.add_argument("--num_images", type=int, default=200)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--train_count", type=int, default=None)
    p.add_argument("--val_count", type=int, default=None)
    p.add_argument("--test_count", type=int, default=None)
    p.add_argument("--min_objs", type=int, default=1)
    p.add_argument("--max_objs", type=int, default=5)
    p.add_argument("--allow_overlap", action="store_true")
    p.add_argument("--no_overlap", dest="allow_overlap", action="store_false")
    p.set_defaults(allow_overlap=True)
    p.add_argument("--class_ratios", type=str, default="")      # JSON: {"classA": 2.0, ...}
    p.add_argument("--per_class_min_max", type=str, default="") # JSON: {"classA": [0,2], ...}
    p.add_argument("--yaml_abs", action="store_true")
    args = p.parse_args()
    cfg = GenConfig(
        assets_dir=args.assets_dir,
        out_dir=args.out_dir,
        image_size=(args.image_width, args.image_height),
        num_images=args.num_images,
        splits=(args.train_ratio, args.val_ratio, args.test_ratio),
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        min_objects_per_image=args.min_objs,
        max_objects_per_image=args.max_objs,
        allow_overlap=args.allow_overlap,
        yaml_abs=args.yaml_abs,
    )
    if args.class_ratios:
        try: cfg.class_ratios = json.loads(args.class_ratios)
        except Exception as e: print("[WARN] parse --class_ratios failed:", e)
    if args.per_class_min_max:
        try:
            raw=json.loads(args.per_class_min_max); norm={}
            for k,v in raw.items():
                if isinstance(v,(list,tuple)) and len(v)==2:
                    mn,mx=int(v[0]),int(v[1]); mx=max(mx,mn); norm[k]=(mn,mx)
            cfg.per_class_min_max=norm
        except Exception as e: print("[WARN] parse --per_class_min_max failed:", e)
    return cfg

if __name__=="__main__":
    cfg=parse_args()
    generate_dataset(cfg)
