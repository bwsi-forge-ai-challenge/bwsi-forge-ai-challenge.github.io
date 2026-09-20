#!/usr/bin/env bash
# Zip an assignment-specific Gradescope autograder folder for upload.
#
# Usage (from Assignments/):
#   ./make_autograder_zip.sh 0_prerequisites/rnn_assignment_autograder
#   ./make_autograder_zip.sh /abs/path/to/foo_assignment_autograder
#
# Writes <parent>/<folder_name>.zip next to the autograder folder.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-*_autograder-folder>" >&2
  exit 1
fi

ASSIGNMENTS_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$1"
if [[ "$SRC" != /* ]]; then
  SRC="$ASSIGNMENTS_ROOT/$SRC"
fi
SRC="$(cd "$SRC" && pwd)"
NAME="$(basename "$SRC")"
PARENT="$(dirname "$SRC")"
OUT="$PARENT/${NAME}.zip"

if [[ ! -f "$SRC/run_autograder" || ! -f "$SRC/setup.sh" ]]; then
  echo "Error: $SRC does not look like a Gradescope autograder folder" >&2
  echo "  (expected run_autograder and setup.sh)" >&2
  exit 1
fi

chmod +x "$SRC/setup.sh" "$SRC/run_autograder" 2>/dev/null || true
rm -f "$OUT"
(
  cd "$SRC"
  zip -r "$OUT" . \
    -x "*.pyc" \
    -x "*__pycache__*" \
    -x "*.DS_Store" \
    -x "*/.git/*" \
    -x "README.md"
)
echo "Wrote $OUT"
unzip -l "$OUT" | head -50
