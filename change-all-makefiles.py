#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Compile and test all *-sycl benchmarks.")
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
        replacement_target="clang++"
        if makefile.exists():
            with open(str(makefile), "r+") as mf:
                content = mf.read()
                replacement_start_idx = content.find(replacement_target)
                if replacement_start_idx >= 0:
                    content = content[:replacement_start_idx] + "$(CXX)" + content[replacement_start_idx+len(replacement_target):]
                    mf.seek(0)
                    mf.write(content)
                    mf.truncate()
                    print("Content modified!")
                else:
                    print(f'No "{replacement_target}" found, Content unchanged!')
        else:
            print(f'File not found: "{makefile}"')
        print("Exit")
            


if __name__ == "__main__":
    main()
