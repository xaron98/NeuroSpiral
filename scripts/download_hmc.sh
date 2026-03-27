#!/bin/bash
# Download HMC Sleep Staging Database from PhysioNet
# 151 PSGs, AASM scoring, open access
# Naming: SN001.edf + SN001_sleepscoring.edf
# Usage: bash scripts/download_hmc.sh [data_dir]

DATA_DIR="${1:-data/hmc}"
mkdir -p "$DATA_DIR"
BASE="https://physionet.org/files/hmc-sleep-staging/1.1/recordings"

count=0
failed=0

for i in $(seq 1 154); do
  sid=$(printf "SN%03d" $i)
  PSG="${sid}.edf"
  HYP="${sid}_sleepscoring.edf"
  PSG_PATH="$DATA_DIR/$PSG"
  HYP_PATH="$DATA_DIR/$HYP"

  # Skip if both already valid
  if [ -f "$PSG_PATH" ] && [ -f "$HYP_PATH" ]; then
    psg_size=$(stat -f%z "$PSG_PATH" 2>/dev/null || stat -c%s "$PSG_PATH" 2>/dev/null || echo 0)
    hyp_size=$(stat -f%z "$HYP_PATH" 2>/dev/null || stat -c%s "$HYP_PATH" 2>/dev/null || echo 0)
    if [ "$psg_size" -gt 1000000 ] && [ "$hyp_size" -gt 500 ]; then
      count=$((count + 1))
      continue
    fi
  fi

  # Download PSG
  echo -n "  $sid: PSG..."
  curl -sf "$BASE/$PSG" -o "$PSG_PATH"
  psg_size=$(stat -f%z "$PSG_PATH" 2>/dev/null || stat -c%s "$PSG_PATH" 2>/dev/null || echo 0)
  if [ "$psg_size" -lt 1000000 ]; then
    rm -f "$PSG_PATH"
    echo " not found (may be excluded recording)"
    failed=$((failed + 1))
    continue
  fi

  # Download hypnogram
  echo -n " ok. HYP..."
  curl -sf "$BASE/$HYP" -o "$HYP_PATH"
  hyp_size=$(stat -f%z "$HYP_PATH" 2>/dev/null || stat -c%s "$HYP_PATH" 2>/dev/null || echo 0)
  if [ "$hyp_size" -gt 500 ]; then
    echo " ok"
    count=$((count + 1))
  else
    echo " failed"
    rm -f "$PSG_PATH" "$HYP_PATH"
    failed=$((failed + 1))
  fi
done

echo ""
echo "═══════════════════════════════════════"
echo "  HMC Download complete"
echo "  Valid pairs: $count"
echo "  Failed/excluded: $failed"
echo "  (SN014, SN064, SN135 were removed by authors)"
echo "═══════════════════════════════════════"
