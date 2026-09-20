#!/usr/bin/env bash
# Smoke-test an assignment autograder locally (outside Gradescope).
#
# Usage (from Assignments/):
#   ./run_local_autograder.sh 0_prerequisites/rnn_assignment_autograder --solution
#   ./run_local_autograder.sh 0_prerequisites/rnn_assignment_autograder --student
#   ./run_local_autograder.sh 0_prerequisites/rnn_assignment_autograder /path/to/file.ipynb
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-*_autograder-folder> [--solution|--student|<notebook.ipynb>]" >&2
  exit 1
fi

ASSIGNMENTS_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$1"
shift || true
if [[ "$SRC" != /* ]]; then
  SRC="$ASSIGNMENTS_ROOT/$SRC"
fi
SRC="$(cd "$SRC" && pwd)"
ASSIGN_DIR="$(dirname "$SRC")"
NAME="$(basename "$SRC")"
# rnn_assignment_autograder → rnn_assignment
STEM="${NAME%_autograder}"

MODE="${1:-}"
case "$MODE" in
  --solution|"")
    NB="$ASSIGN_DIR/${STEM}_solution.ipynb"
    if [[ "$MODE" == "" && ! -f "$NB" ]]; then
      NB="$ASSIGN_DIR/${STEM}.ipynb"
    fi
    ;;
  --student)
    NB="$ASSIGN_DIR/${STEM}.ipynb"
    ;;
  *)
    NB="$MODE"
    if [[ "$NB" != /* ]]; then
      NB="$ASSIGNMENTS_ROOT/$NB"
    fi
    ;;
esac

if [[ ! -f "$NB" ]]; then
  echo "Notebook not found: $NB" >&2
  exit 1
fi
if [[ ! -f "$SRC/run_tests.py" ]]; then
  echo "Missing run_tests.py in $SRC" >&2
  exit 1
fi

if [[ -f "$SRC/requirements.txt" ]]; then
  python3 -m pip install -q -r "$SRC/requirements.txt"
fi

echo "Autograder: $SRC"
echo "Notebook:   $NB"
AUTOGRADE_NOTEBOOK="$NB" python3 "$SRC/run_tests.py"
echo "Results:    $ASSIGN_DIR/_autograder_local_results/results.json"
