#!/bin/bash
set -e
KAGGLE_USER="ibrahimjamal786"
KERNEL_ID="$KAGGLE_USER/snake-classifier"
OUTPUT_DIR="./outputs/run_$(date +%Y-%m-%d_%H-%M)"

echo "=================================================="
echo "  Snake Classifier — Kaggle Training"
echo "  $(date)"
echo "=================================================="

echo ""
echo "[1/4] Syncing code to kernel folder..."
rsync -av --delete src/     kaggle_kernel/src/
rsync -av --delete configs/ kaggle_kernel/configs/

echo "[2/4] Pushing to Kaggle..."
kaggle kernels push -p kaggle_kernel/
echo "  Live logs → https://www.kaggle.com/code/$KERNEL_ID"
sleep 15

echo "[3/4] Waiting for GPU run..."
while true; do
    STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null \
             | grep -oP "(?<=status: )\w+" || echo "unknown")
    echo "  [$(date +%H:%M:%S)] $STATUS"
    [ "$STATUS" = "complete" ] && echo "  Done!" && break
    [ "$STATUS" = "error" ]    && echo "  FAILED — check logs at kaggle.com" && exit 1
    sleep 30
done

echo "[4/4] Downloading outputs..."
mkdir -p "$OUTPUT_DIR"
kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
echo ""
echo "  Saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
