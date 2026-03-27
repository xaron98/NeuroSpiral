#!/bin/bash
# Download Sleep-EDF cassette recordings from PhysioNet
# Handles the EC/EH hypnogram naming variation
# Usage: bash scripts/download_sleep_edf.sh [data_dir]

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"
BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

count=0
failed=0

for i in $(seq 1 83); do
  sid=$(printf "SC40%02d" $i)
  PSG="${sid}E0-PSG.edf"
  PSG_PATH="$DATA_DIR/$PSG"

  # Skip if PSG already valid
  if [ -f "$PSG_PATH" ]; then
    size=$(stat -f%z "$PSG_PATH" 2>/dev/null || stat -c%s "$PSG_PATH" 2>/dev/null || echo 0)
    if [ "$size" -gt 1000000 ]; then
      # Check if hypnogram exists too
      if ls "$DATA_DIR/${sid}"*Hypnogram* 1>/dev/null 2>&1; then
        hyp_size=$(stat -f%z "$DATA_DIR/${sid}"*Hypnogram*.edf 2>/dev/null | head -1 || echo 0)
        if [ "$hyp_size" -gt 500 ] 2>/dev/null; then
          count=$((count + 1))
          continue
        fi
      fi
    fi
  fi

  # Download PSG
  echo -n "  $sid: downloading PSG..."
  curl -sf "$BASE/$PSG" -o "$PSG_PATH"
  size=$(stat -f%z "$PSG_PATH" 2>/dev/null || stat -c%s "$PSG_PATH" 2>/dev/null || echo 0)
  if [ "$size" -lt 1000000 ]; then
    rm -f "$PSG_PATH"
    echo " failed (PSG too small or missing)"
    failed=$((failed + 1))
    continue
  fi
  echo -n " ok. HYP..."

  # Try hypnogram suffixes: EC, EH
  GOT=0
  for suf in EC EH; do
    HYP="${sid}${suf}-Hypnogram.edf"
    HYP_PATH="$DATA_DIR/$HYP"
    curl -sf "$BASE/$HYP" -o "$HYP_PATH"
    hyp_size=$(stat -f%z "$HYP_PATH" 2>/dev/null || stat -c%s "$HYP_PATH" 2>/dev/null || echo 0)
    if [ "$hyp_size" -gt 500 ]; then
      echo " ok ($suf)"
      GOT=1
      count=$((count + 1))
      break
    fi
    rm -f "$HYP_PATH"
  done

  if [ $GOT -eq 0 ]; then
    echo " no valid hypnogram"
    rm -f "$PSG_PATH"
    failed=$((failed + 1))
  fi
done

echo ""
echo "═══════════════════════════════════════"
echo "  Download complete"
echo "  Valid pairs: $count"
echo "  Failed: $failed"
echo "═══════════════════════════════════════"
