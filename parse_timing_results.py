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
            for result in list(zip(sums_ocl, sums_omp)):
                ocl_kernel = result[0][0]
                omp_kernel = result[1][0]
                ocl_value = result[0][1]
                omp_value = result[1][1]
                if ocl_kernel == omp_kernel:
                    winner = ""
                    if float(ocl_value) > float(omp_value):
                        winner = "ocl" 
                    elif float(ocl_value) == float(omp_value):
                        winner = "same"
                    else:
                        winner = "omp"
                    print(f"  {'Kernel':<10} | {'ocl':<10} | {'omp':<10} | {'difference':<10} | {'fastest':<10}")
                    print(f"  {str(ocl_kernel):<10} | {str(ocl_value):<10} | {omp_value:<10} | {str(abs(float(ocl_value)-float(omp_value))):<10} | {winner:<10}")
            print("\tavgs:")
            for result in list(zip(avgs_ocl, avgs_omp)):
                ocl_kernel = result[0][0]
                omp_kernel = result[1][0]
                ocl_value = result[0][1]
                omp_value = result[1][1]
                if ocl_kernel == omp_kernel:
                    winner = ""
                    if float(ocl_value) > float(omp_value):
                        winner = "ocl" 
                    elif float(ocl_value) == float(omp_value):
                        winner = "same"
                    else:
                        winner = "omp"
                    print(f"  {'Kernel':<10} | {'ocl':<10} | {'omp':<10} | {'difference':<10} | {'fastest':<10}")
                    print(f"  {str(ocl_kernel):<10} | {str(ocl_value):<10} | {omp_value:<10} | {str(abs(float(ocl_value)-float(omp_value))):<10} | {winner:<10}")
        else:
            print("run.log not found for both")
        print("Exit")
            


if __name__ == "__main__":
    main()
