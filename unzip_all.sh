#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-src}"

find "$TARGET_DIR" -type f \( \
  -name "*.tar.gz" -o -name "*.tar.bz2" -o -name "*.tar.bz" -o \
  -name "*.gz" -o -name "*.bz2" -o -name "*.bz" \
  \) -exec sh -c '
  for file; do
    dir=$(dirname "$file")
    echo "Processing: $file"

    case "$file" in
      *.tar.gz|*.tar.bz2|*.tar.bz)
        tar -xf "$file" -C "$dir"
        ;;
      *.gz)
        gunzip -k "$file"
        ;;
      *.bz2|*.bz)
        bunzip2 -k "$file"
        ;;
    esac
  done
' sh {} +
