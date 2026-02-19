#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-src}"

find "$TARGET_DIR" -type f \( \
  -name "*.tar.gz" -o -name "*.tar.bz2" -o \
  -name "*.gz" -o -name "*.bz2" \
  \) -exec sh -c '
  for file; do
    dir=$(dirname "$file")
    echo "Processing: $file"

    case "$file" in
      *.tar.gz|*.tar.bz2)
        tar -xf "$file" -C "$dir"
        ;;
      *.gz)
        gunzip -k "$file"
        ;;
      *.bz2)
        bunzip2 -k "$file"
        ;;
    esac
  done
' sh {} +
