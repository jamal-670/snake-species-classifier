from pathlib import Path
import pandas as pd
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm

TARGET = 200
AUG_DIR = Path("data/augmented")

def get_aug():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.2,
                           rotate_limit=30,p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.ColorJitter(brightness=0.3,contrast=0.3,
                      saturation=0.3,hue=0.1,p=0.6),
        A.CLAHE(clip_limit=2.0,p=0.3),
        A.ImageCompression(quality_lower=80,quality_upper=100,p=0.2),
    ])

def balance_dataset(manifest_csv="data/processed/manifest.csv",
                    target=TARGET):
    df      = pd.read_csv(manifest_csv)
    counts  = df["label"].value_counts()
    aug     = get_aug()
    records = df.to_dict("records")

    print(f"\nTarget per class: {target}")
    print(f"Classes needing augmentation: "
          f"{(counts < target).sum()} / {len(counts)}")

    for label, count in counts.items():
        if count >= target:
            # Copy as-is
            out_dir = AUG_DIR / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for row in df[df["label"]==label].itertuples():
                img = Image.open(row.path).convert("RGB")
                img.save(out_dir / Path(row.path).name, "JPEG", quality=95)
            continue

        needed  = target - count
        src     = df[df["label"]==label]
        out_dir = AUG_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy originals first
        for row in src.itertuples():
            img = Image.open(row.path).convert("RGB")
            img.save(out_dir / Path(row.path).name, "JPEG", quality=95)
            records.append({"path": str(out_dir/Path(row.path).name),
                            "label": label})

        # Generate augmented images
        gen = 0
        pbar = tqdm(total=needed, desc=f"  Augmenting {label[:25]}", leave=False)
        while gen < needed:
            row    = src.sample(1).iloc[0]
            img    = np.array(Image.open(row["path"]).convert("RGB"))
            result = aug(image=img)["image"]
            fname  = f"aug_{label}_{gen:05d}.jpg"
            Image.fromarray(result).save(out_dir/fname, "JPEG", quality=92)
            records.append({"path": str(out_dir/fname), "label": label})
            gen += 1
            pbar.update(1)
        pbar.close()

    balanced_df = pd.DataFrame(records).drop_duplicates(subset=["path"])
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(AUG_DIR/"balanced_manifest.csv", index=False)

    print("\n=== Balanced Distribution ===")
    vc = balanced_df["label"].value_counts()
    print(f"  Total images: {len(balanced_df)}")
    print(f"  Classes:      {len(vc)}")
    print(f"  Min:          {vc.min()}")
    print(f"  Max:          {vc.max()}")
    return balanced_df

if __name__ == "__main__":
    balance_dataset()
