import os, sys, glob, shutil
from scipy.io import loadmat

def load_points(mat_path):
    return loadmat(mat_path)["image_info"][0, 0][0, 0][0]

def convert_split(src_root, out_root, split):
    img_dir = os.path.join(src_root, split, "images")
    gt_dir = os.path.join(src_root, split, "ground_truth")
    out_img = os.path.join(out_root, split, "images")
    out_gt = os.path.join(out_root, split, "ground_truth")
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_gt, exist_ok=True)
    lines = []
    for p in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        base = os.path.splitext(os.path.basename(p))[0]
        pts = load_points(os.path.join(gt_dir, f"GT_{base}.mat"))
        shutil.copy(p, os.path.join(out_img, base + ".jpg"))
        with open(os.path.join(out_gt, base + ".txt"), "w") as f:
            for x, y in pts:
                f.write(f"{float(x)} {float(y)}\n")
        lines.append(f"{split}/images/{base}.jpg {split}/ground_truth/{base}.txt")
    return lines

src, out = sys.argv[1], sys.argv[2]
for split, name in [("train_data", "shanghai_tech_part_a_train.list"),
                    ("test_data", "shanghai_tech_part_a_test.list")]:
    lines = convert_split(src, out, split)
    open(os.path.join(out, name), "w").write("\n".join(lines) + "\n")
    print(split, len(lines), "images")
