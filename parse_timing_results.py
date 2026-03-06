#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

PRECISION=6
PADDING="15"

def main():
    ap = argparse.ArgumentParser(
        description="parse all *-sycl benchmarks for ocl and omp."
    )
    ap.add_argument(
        "--result-ocl",
        default=".",
        help="Root directory containing ocl results (default: .)",
    )
    ap.add_argument(
        "--result-omp",
        default=".",
        help="Root directory containing omp sycl results (default: .)",
    )
    args = ap.parse_args()

    result_ocl_root = Path(args.result_ocl).resolve()
    result_omp_root = Path(args.result_omp).resolve()

    # Discover projects
    projects_ocl = sorted([p for p in result_ocl_root.glob("*-sycl") if p.is_dir()])
    projects_omp = sorted([p for p in result_omp_root.glob("*-sycl") if p.is_dir()])
    if not projects_ocl:
        print(
            f"No projects found under {result_ocl_root} matching {args.pattern}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not projects_omp:
        print(
            f"No projects found under {result_omp_root} matching {args.pattern}",
            file=sys.stderr,
        )
        sys.exit(1)
    result_file = Path.cwd() / "timing_results.csv"
    with open(str(result_file), "w") as output:
        output.write(
            "name,ocl_total,omp_total,ocl_avg,omp_avg,total_delta,avg_delta,fastest_total,fastest_avg\n"
        )
        for proj in list(zip(projects_ocl, projects_omp)):
            proj_name = proj[0].name
            print(f"\nEnter ==> {proj_name}")
            runlog_ocl = proj[0] / "run.log"
            runlog_omp = proj[1] / "run.log"
            if runlog_ocl.exists() and runlog_omp.exists():
                ocl_log = open(runlog_ocl, "r")
                omp_log = open(runlog_omp, "r")
                content_ocl = ocl_log.read()
                content_omp = omp_log.read()
                sums_ocl = re.findall(
                    r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d\.]+)", content_ocl
                )
                avgs_ocl = re.findall(
                    r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d\.]+)", content_ocl
                )
                sums_omp = re.findall(
                    r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d\.]+)", content_omp
                )
                avgs_omp = re.findall(
                    r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d\.]+)", content_omp
                )
                print(" " + proj_name + ":")
                print(
                    f"  {'ocl_total':<{PADDING}} | {'omp_total':<{PADDING}} | {'ocl_avg':<{PADDING}} | {'omp_avg':<{PADDING}} | {'total_delta':<{PADDING}} | {'avg_delta':<{PADDING}} | {'fastest_total':<{PADDING}} | {'fastest_avg':<{PADDING}}"
                )
                ocl_total = 0
                omp_total = 0
                ocl_avg = 0
                omp_avg = 0
                for result in list(zip(sums_ocl, sums_omp, avgs_ocl, avgs_omp)):
                    ocl_kernel = result[0][0]
                    omp_kernel = result[1][0]
                    if ocl_kernel == omp_kernel:
                        ocl_total += float(result[0][1])
                        omp_total += float(result[1][1])
                        ocl_avg += float(result[2][1])
                        omp_avg += float(result[3][1])
                total_winner = ""
                avg_winner = ""
                if float(ocl_total) > float(omp_total):
                    total_winner = "ocl"
                elif float(ocl_total) == float(omp_total):
                    total_winner = "same"
                else:
                    avg_winner = "omp"
                if float(ocl_avg) > float(omp_avg):
                    avg_winner = "ocl"
                elif float(ocl_avg) == float(omp_avg):
                    avg_winner = "same"
                else:
                    avg_winner = "omp"
                print(
                    f"  {round(ocl_total, PRECISION):<{PADDING}} | {round(omp_total, PRECISION):<{PADDING}} | {round(ocl_avg, PRECISION):<{PADDING}} | {round(omp_avg, PRECISION):<{PADDING}} | {round(abs(float(ocl_total) - float(omp_total)), PRECISION):<{PADDING}} | {round(abs(float(ocl_avg) - float(omp_avg)),PRECISION):<{PADDING}} | {total_winner:<{PADDING}} | {avg_winner:<{PADDING}}"
                )
                output.write(
                    f"{str(proj_name)},{str(round(ocl_total, PRECISION))},{round(omp_total, PRECISION)},{round(ocl_avg, PRECISION)},{round(omp_avg,PRECISION)},{round(abs(float(ocl_total) - float(omp_total)),PRECISION)},{round(abs(float(ocl_avg) - float(omp_avg)),PRECISION)},{total_winner},{avg_winner}\n"
                )
            else:
                print("run.log not found for both")
            print("Exit")


if __name__ == "__main__":
    main()
