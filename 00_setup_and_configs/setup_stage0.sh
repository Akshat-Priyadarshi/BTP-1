#!/usr/bin/env bash
# =============================================================================
# setup_stage0.sh
#
# WHY THIS FILE EXISTS
#   The AdaStress repo has no setup script. It assumes the folders and data are
#   already in the right place. We need a fixed layout so results from
#   different experiments never land on top of each other, and so the original
#   repo code stays untouched.
#
# WHAT IT DOES
#   1. Makes the folder layout for the BTP work.
#   2. Puts the cloned AdaStress repo in one place: vendor/adastress
#   3. Writes down the exact git commit of that clone.
#   4. Makes the clone read-only, so we cannot edit it by accident.
#   5. Finds your WESAD and user1 data and links to it (no copying).
#
# WHERE TO RUN IT
#   From inside:
#     BTP/Physiological-Stress-Detection/AdaStress_Integration/AdaStress_BTP/
#
#   Run:  bash setup_stage0.sh
#
# IT DOES NOT TOUCH
#   - your data folders (it only makes links pointing at them)
#   - Stress_Preprocessing/
#   - the older modified repo in AdaStress_Integration/
#
# Safe to run more than once. It skips whatever already exists.
# =============================================================================

set -e

ROOT="$(pwd)"
echo "Workspace root: $ROOT"
echo

# -----------------------------------------------------------------------------
# Safety check, so we do not build the tree in the wrong folder.
# -----------------------------------------------------------------------------
if [ "$(basename "$ROOT")" != "AdaStress_BTP" ]; then
  echo "STOP. Run this from the AdaStress_BTP folder."
  echo "You are in: $ROOT"
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: folders
#
# runs/    one folder per experiment. Never reused, never overwritten.
# btp/     our own code. Nothing here comes from the original repo.
# vendor/  the original repo, read-only.
# context/ notes about which context columns exist in which dataset.
# docs/    written decisions and results.
# -----------------------------------------------------------------------------
echo "[1/5] Making folders"
mkdir -p btp docs context vendor runs
mkdir -p data/raw
mkdir -p data/processed/wesad
mkdir -p data/processed/user1
echo "      done"
echo

# -----------------------------------------------------------------------------
# Step 2: put the clone at vendor/adastress
# -----------------------------------------------------------------------------
echo "[2/5] Placing the original repo at vendor/adastress"

OLD_CLONE="Adaptive-Stress-Monitoring-on-Wearable-Devices"

if [ -d "vendor/adastress" ]; then
  echo "      vendor/adastress already exists, leaving it alone"
elif [ -d "$OLD_CLONE" ]; then
  echo "      found your clone here, moving it in"
  chmod -R u+w "$OLD_CLONE" 2>/dev/null || true
  mv "$OLD_CLONE" vendor/adastress
else
  echo "      no clone found here, cloning fresh from GitHub"
  git clone https://github.com/giTan7/Adaptive-Stress-Monitoring-on-Wearable-Devices.git vendor/adastress
fi
echo

# -----------------------------------------------------------------------------
# Step 3: write down the commit
#
# In six months "we used the repo" means nothing. "We used commit 15d1ca7" can
# be checked. This one line carries the whole reproducibility claim.
# -----------------------------------------------------------------------------
echo "[3/5] Recording the commit"
if [ ! -f vendor/adastress_commit.txt ]; then
  ( cd vendor/adastress && git rev-parse HEAD ) > vendor/adastress_commit.txt
fi
echo "      commit: $(cat vendor/adastress_commit.txt)"
echo

# -----------------------------------------------------------------------------
# Step 4: make the clone read-only
#
# This is how we can honestly say we did not change the original code.
# Not a promise, a file permission.
# If you ever really need to write in there: chmod -R u+w vendor/adastress
# -----------------------------------------------------------------------------
echo "[4/5] Making vendor/adastress read-only"
chmod -R a-w vendor/adastress
echo "      done"
echo

# -----------------------------------------------------------------------------
# Step 5: find and link the data
#
# We LINK instead of copying, so there is only ever one copy of the raw data
# on disk. A link behaves like a folder for every script we run.
#
# We look in the places your data is most likely to be. If none of them work,
# the script tells you the exact command to run yourself.
# -----------------------------------------------------------------------------
echo "[5/5] Finding and linking your data"
echo

link_dataset () {
  # $1 = name to use inside data/raw   (WESAD or user1)
  # $2 $3 $4 = candidate paths to try, in order
  local name="$1"; shift
  local target="data/raw/$name"

  if [ -e "$target" ]; then
    echo "      $target already exists, leaving it alone"
    return 0
  fi

  local candidate
  for candidate in "$@"; do
    if [ -d "$candidate" ]; then
      # turn it into a full path so the link never breaks
      local full
      full="$(cd "$candidate" && pwd)"
      ln -s "$full" "$target"
      echo "      linked $target"
      echo "             -> $full"
      return 0
    fi
  done

  echo "      COULD NOT FIND $name"
  echo "      tried:"
  for candidate in "$@"; do
    echo "        $candidate"
  done
  echo "      make the link yourself, using the real path:"
  echo "        ln -s /full/path/to/$name $target"
  return 0
}

# Most likely: two levels up, inside Physiological-Stress-Detection/data/
link_dataset "WESAD" \
  "$ROOT/../../data/WESAD" \
  "$ROOT/../../data/wesad" \
  "$ROOT/../../../data/WESAD"

link_dataset "user1" \
  "$ROOT/../../data/user1" \
  "$ROOT/../../data/User1" \
  "$ROOT/../../../data/user1"

echo

# -----------------------------------------------------------------------------
# Show what we ended up with, so you can check it is right.
# -----------------------------------------------------------------------------
echo "============================================================"
echo "Checking the links actually work"
echo "============================================================"

if [ -d "data/raw/WESAD" ]; then
  n_subj=$(ls data/raw/WESAD 2>/dev/null | grep -c '^S' || true)
  echo "  data/raw/WESAD  -> $n_subj folders starting with S"
  echo "  first few:"
  ls data/raw/WESAD 2>/dev/null | head -4 | sed 's/^/    /'
else
  echo "  data/raw/WESAD  -> NOT SET UP"
fi

if [ -d "data/raw/user1" ]; then
  n_csv=$(ls data/raw/user1/*.csv 2>/dev/null | wc -l || true)
  echo "  data/raw/user1  -> $n_csv csv files"
  echo "  first few:"
  ls data/raw/user1/*.csv 2>/dev/null | head -4 | xargs -n1 basename 2>/dev/null | sed 's/^/    /'
else
  echo "  data/raw/user1  -> NOT SET UP"
fi

echo
echo "============================================================"
echo "Stage 0 folder setup finished."
echo
echo "  AdaStress_BTP/"
echo "    btp/                  our code"
echo "    vendor/adastress/     original repo, read-only"
echo "    data/raw/WESAD        link to your raw WESAD"
echo "    data/raw/user1        link to your user1 csv files"
echo "    data/processed/"
echo "    runs/                 one folder per experiment"
echo "    context/"
echo "    docs/"
echo
echo "Next, run these two and send me what they print:"
echo "  python3 btp/s0_check_wesad_raw.py"
echo "  python3 btp/s0_check_user1_context.py"
echo "============================================================"
