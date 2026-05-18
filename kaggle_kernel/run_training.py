import os, sys, shutil, subprocess, json, glob

# Debug — show entire Kaggle filesystem
print("="*55)
print("  FILESYSTEM DEBUG")
print("="*55)

for check_path in ["/kaggle", "/kaggle/src", "/kaggle/working", "/kaggle/input"]:
    if os.path.exists(check_path):
        contents = os.listdir(check_path)
        print(f"\n{check_path}/")
        for item in contents:
            full = os.path.join(check_path, item)
            kind = "DIR" if os.path.isdir(full) else "FILE"
            print(f"  [{kind}] {item}")
    else:
        print(f"\n{check_path} — DOES NOT EXIST")

# Check /kaggle/src deeply
print("\n\n/kaggle/src full tree:")
for root, dirs, files in os.walk("/kaggle/src"):
    level = root.replace("/kaggle/src","").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  {f}")
