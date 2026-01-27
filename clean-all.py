#!/usr/bin/env python3
import argparse
import sys
import time
import os
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description="clean all *-sycl benchmarks.")
    ap.add_argument("--sycl-root", default=".", help="Root directory containing *-sycl projects (default: .)")
    args = ap.parse_args()

    sycl_root = Path(args.sycl_root).resolve()

    # Discover projects
    projects = sorted([p for p in sycl_root.glob("*-sycl") if p.is_dir()])
    if not projects:
        print(f"No projects found under {sycl_root} matching {args.pattern}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()

    for proj in projects:
        proj_name = proj.name
        print(f"Enter ==> {proj_name}")
        os.chdir(proj) 
        subprocess.call(["make", "clean"])
        print(f"Exit")


if __name__ == "__main__":
    main()
