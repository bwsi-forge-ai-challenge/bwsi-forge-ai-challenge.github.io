#!/usr/bin/env bash
# Gradescope setup: runs once when the autograder image is built.
# The grader runs the students' functions on its own tiny models on a CPU, so it needs
# NumPy, Pillow, and the CPU-only build of PyTorch -- no GPU, no diffusers, no model downloads.
set -euo pipefail

apt-get update
apt-get install -y python3 python3-pip python3-venv

python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r /autograder/source/requirements.txt
