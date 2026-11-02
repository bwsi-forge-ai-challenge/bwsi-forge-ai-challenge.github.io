#!/usr/bin/env bash
# Gradescope setup: runs once when the autograder image is built.
# The grader re-runs the submission on CPU (it trains two tiny linear heads), so it needs
# NumPy, Matplotlib (for the notebook's plotting cells) and the CPU-only build of PyTorch.
set -euo pipefail

apt-get update
apt-get install -y python3 python3-pip python3-venv

python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r /autograder/source/requirements.txt
