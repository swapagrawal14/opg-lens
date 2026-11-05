# Dental Radiograph AI - v0 training (Kaggle GPU). DENTEX CC BY-NC-SA 4.0, non-commercial.
import subprocess, sys, os, json, glob, random, shutil
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx"], check=True)
from ultralytics import YOLO
from PIL import Image
js = [p for p in glob.glob("/kaggle/input/**/*.json", recursive=True) if "metadata" not in p][0]
root = os.path.dirname(js)
D = json.load(open(js))
imgs = {i["id"]: i for i in D["images"]}
ims = {os.path.basename(p): p for p in glob.glob(root + "/**/*.png", recursive=True) + glob.glob(root + "/**/*.jpg", recursive=True)}
print("json", js, "images", len(imgs), "files", len(ims), "anns", len(D["annotations"]))
print("keys", list(D["annotations"][0].keys()))
DIS = [c["name"] for c in sorted(D["categories_3"], key=lambda c: c["id"])]
FDI = [f"{q+1}{t+1}" for q in range(4) for t in range(8)]
ids = list(imgs); random.seed(0); random.shuffle(ids)
val = set(ids[: max(50, len(ids) // 10)])
by = {}
for a in D["annotations"]:
    by.setdefault(a["image_id"], []).append(a)
def build(name, clsfn, names):
    base = f"/kaggle/working/{name}"
    for s in ("train", "val"):
        os.makedirs(f"{base}/images/{s}", exist_ok=True); os.makedirs(f"{base}/labels/{s}", exist_ok=True)
    for iid, im in imgs.items():
        fn = os.path.basename(im["file_name"])
        if fn not in ims: continue
        s = "val" if iid in val else "train"
        W, H = im["width"], im["height"]
        os.symlink(ims[fn], f"{base}/images/{s}/{fn}") if not os.path.exists(f"{base}/images/{s}/{fn}") else None
        with open(f"{base}/labels/{s}/{os.path.splitext(fn)[0]}.txt", "w") as f:
            for a in by.get(iid, []):
                x, y, w, h = a["bbox"]
                f.write(f"{clsfn(a)} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")
    yml = f"{base}/data.yaml"
    open(yml, "w").write(f"path: {base}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n")
    return yml
res = {}
for name, fn, names in [("diagnosis", lambda a: a["category_id_3"], DIS),
                        ("fdi", lambda a: a["category_id_1"] * 8 + a["category_id_2"], FDI)]:
    yml = build(name, fn, names)
    m = YOLO("yolo11s.pt")
    m.train(data=yml, imgsz=1024, epochs=80, batch=8, patience=20, project="/kaggle/working/runs", name=name, fliplr=0.0, plots=False)
    best = YOLO(f"/kaggle/working/runs/{name}/weights/best.pt")
    v = best.val(data=yml, imgsz=1024)
    res[name] = {"mAP50": float(v.box.map50), "mAP50_95": float(v.box.map), "per_class_mAP50": dict(zip(names, map(float, v.box.ap50))) if len(v.box.ap50)==len(names) else None}
    best.export(format="onnx", imgsz=1024, simplify=True)
json.dump(res, open("/kaggle/working/metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
