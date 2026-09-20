#!/usr/bin/env bash
# Gradescope setup: runs once when the autograder image is built.
set -euo pipefail

apt-get update
apt-get install -y python3 python3-pip python3-venv

python3 -m pip install --upgrade pip
python3 -m pip install -r /autograder/source/requirements.txt
