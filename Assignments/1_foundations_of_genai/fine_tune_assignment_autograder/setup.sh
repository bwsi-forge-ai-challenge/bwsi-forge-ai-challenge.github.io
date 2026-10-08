#!/usr/bin/env bash
# Gradescope setup: runs once when the autograder image is built.
# The grader is purely static (AST + pure-Python checks) -- no ML libraries needed.
set -euo pipefail

apt-get update
apt-get install -y python3
