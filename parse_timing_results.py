#!/usr/bin/env python3
import argparse
from os.path import exists
import re
import sys
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="parse all *-sycl benchmarks for ocl and omp.")
    ap.add_argument("--result-ocl", default=".", help="Root directory containing ocl results (default: .)")
    ap.add_argument("--result-omp", default=".", help="Root directory containing omp sycl results (default: .)")
    args = ap.parse_args()

    result_ocl_root = Path(args.result_ocl).resolve()
    result_omp_root = Path(args.result_omp).resolve()

    # Discover projects
    projects_ocl = sorted([p for p in result_ocl_root.glob("*-sycl") if p.is_dir()])
    projects_omp = sorted([p for p in result_omp_root.glob("*-sycl") if p.is_dir()])
    if not projects_ocl:
        print(f"No projects found under {result_ocl_root} matching {args.pattern}", file=sys.stderr)
        sys.exit(1)
    if not projects_omp:
        print(f"No projects found under {result_omp_root} matching {args.pattern}", file=sys.stderr)
        sys.exit(1)
    result_file = Path.cwd() / "timing_results.csv"
    for proj in list(zip(projects_ocl, projects_omp)):
        proj_name = proj[0].name
        print(f"Enter ==> {proj_name}")
        runlog_ocl = proj[0] / "run.log"
        runlog_omp = proj[1] / "run.log"
        if runlog_ocl.exists() and runlog_omp.exists():
            ocl_log = open(runlog_ocl, "r")
            omp_log = open(runlog_omp, "r")
            content_ocl = ocl_log.read()
            content_omp = omp_log.read()
            sums_ocl = re.findall(r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d\.]+)", content_ocl)
            avgs_ocl = re.findall(r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d\.]+)", content_ocl)
            sums_omp = re.findall(r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d\.]+)", content_omp)
            avgs_omp = re.findall(r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d\.]+)", content_omp)
            print(proj_name + ":")
            print("\tsums:")
            for result in sums:
                print(f"\t\tKernel {result[0]}: {result[1]} s")
            print("\tavgs:")
            for result in avgs:
                print(f"\t\tKernel {result[0]}: {result[1]} s")
        else:
            print("run.log not found for both")
        print("Exit")
            


if __name__ == "__main__":
    main()
