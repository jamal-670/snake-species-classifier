import os
from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd
from tqdm import tqdm

MIN_SIZE = (64, 64)

def audit_dataset(raw_dir="data/raw", clean_dir="data/processed"):
    raw_dir   = Path(raw_dir)
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    records, seen_hashes = [], set()
    stats = {"total":0,"corrupt":0,"duplicate":0,"too_small":0,"kept":0}

    species_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(species_dirs)} species folders")

    for species_dir in species_dirs:
        label   = species_dir.name.strip().lower().replace(" ","_")\
                                                   .replace("'","").replace("\"","")
        out_dir = clean_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        images = list(species_dir.glob("*"))
        images = [f for f in images if f.suffix.lower()
                  in ['.jpg','.jpeg','.png','.bmp','.webp']]

        for img_path in tqdm(images, desc=f"  {label[:30]}", leave=False):
            stats["total"] += 1
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                stats["corrupt"] += 1
                continue

            if img.size[0] < MIN_SIZE[0] or img.size[1] < MIN_SIZE[1]:
                stats["too_small"] += 1
                continue

            phash = str(imagehash.phash(img))
            if phash in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(phash)

            fname = f"{label}_{stats['kept']:05d}.jpg"
            img.save(out_dir / fname, "JPEG", quality=95)
            records.append({
                "path":   str(out_dir / fname),
                "label":  label,
                "width":  img.size[0],
                "height": img.size[1]
            })
            stats["kept"] += 1

    df = pd.DataFrame(records)
    df.to_csv(clean_dir / "manifest.csv", index=False)

    print("\n=== Audit Summary ===")
    for k,v in stats.items():
        print(f"  {k:12s}: {v}")
    print(f"\n=== Class distribution ===")
    vc = df["label"].value_counts()
    print(f"  Min images: {vc.min()} ({vc.idxmin()})")
    print(f"  Max images: {vc.max()} ({vc.idxmax()})")
    print(f"  Mean:       {vc.mean():.0f}")
    return df

if __name__ == "__main__":
    audit_dataset()
