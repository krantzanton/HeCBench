#!/usr/bin/env bash
set -euo pipefail

find . -type f -name "*.tar.gz" -exec sh -c '
  for file; do
    dir=$(dirname "$file")
    echo "Extracting $file -> $dir"
    tar -xzf "$file" -C "$dir"
  done
' sh {} +
