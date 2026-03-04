#!/usr/bin/env python3
import argparse
import re
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="parse all *-sycl benchmarks.")
    ap.add_argument("--result-root", default=".", help="Root directory containing *-sycl results (default: .)")
    args = ap.parse_args()

    sycl_root = Path(args.sycl_root).resolve()

    # Discover projects
    projects = sorted([p for p in sycl_root.glob("*-sycl") if p.is_dir()])
    if not projects:
        print(f"No projects found under {sycl_root} matching {args.pattern}", file=sys.stderr)
        sys.exit(1)
    result_file = sycl_root / "timing_results.csv"
    for proj_result in projects:
        proj_name = proj_result.name
        print(f"Enter ==> {proj_name}")
        runlog = proj_result / "run.log"

        if runlog.exists():
            with open(str(runlog), "r") as runlog_handle:
                content = runlog_handle.read()
                sums = re.findall(r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d\.]+)", content)
                avgs = re.findall(r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d\.]+)", content)

                

        else:
            print(f'File not found: "{runlog}"')
        print("Exit")
            


if __name__ == "__main__":
    main()
