#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="modify all cmake *-sycl benchmarks.")
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
        makefile = proj / "Makefile"

        if makefile.exists():
            with open(str(makefile), "r+") as mf:
                lines = mf.readlines()
                cc_line = -1
                gpu_line = -1
                new_cc_line = ""
                new_gpu_line = ""
                for i, line in enumerate(lines):
                    if "CC" in line and "=" in line and "clang++" in line:
                        cc_line = i
                        new_cc_line = line.replace("clang++", "$(CXX)")
                        continue
                    if "GPU" in line and "=" in line and "yes" in line:
                        gpu_line = i
                        new_gpu_line = line.replace("yes", "$(USE_GPU)")
                        break;
                if cc_line >= 0:
                    lines[cc_line] = new_cc_line
                if gpu_line >= 0:
                    lines[gpu_line] = new_gpu_line 
                mf.seek(0)
                for line in lines:
                    mf.write(line)
                mf.truncate()
                print("Content modified!")
        else:
            print(f'File not found: "{makefile}"')
        print("Exit")
            


if __name__ == "__main__":
    main()
