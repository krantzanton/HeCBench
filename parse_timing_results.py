#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

PRECISION = 6
PADDING = 15

TIME_PATTERN_SUM = re.compile(r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d.]+)")
TIME_PATTERN_AVG = re.compile(r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d.]+)")


def parse_log(path: Path):
    content = path.read_text()

    sums = TIME_PATTERN_SUM.findall(content)
    avgs = TIME_PATTERN_AVG.findall(content)

    sums_by_kernel = {kernel: float(value) for kernel, value in sums}
    avgs_by_kernel = {kernel: float(value) for kernel, value in avgs}

    return sums_by_kernel, avgs_by_kernel


def fastest(a_name, a_value, b_name, b_value):
    if a_value < b_value:
        return a_name
    if b_value < a_value:
        return b_name
    return "same"


def parse(args):
    result_ocl_root = Path(args.result_ocl).resolve()
    result_omp_root = Path(args.result_omp).resolve()

    projects_ocl = {
        p.name: p for p in result_ocl_root.glob("*-sycl") if p.is_dir()
    }
    projects_omp = {
        p.name: p for p in result_omp_root.glob("*-sycl") if p.is_dir()
    }

    if not projects_ocl:
        print(
            f"No projects found under {result_ocl_root} matching *-sycl",
            file=sys.stderr,
        )
        sys.exit(1)

    if not projects_omp:
        print(
            f"No projects found under {result_omp_root} matching *-sycl",
            file=sys.stderr,
        )
        sys.exit(1)

    common_projects = sorted(set(projects_ocl) & set(projects_omp))

    if not common_projects:
        print("No matching project names found between ocl and omp results.", file=sys.stderr)
        sys.exit(1)

    result_file = Path.cwd() / "timing_results.csv"

    with result_file.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "name",
            "ocl_total",
            "omp_total",
            "ocl_avg",
            "omp_avg",
            "total_delta",
            "avg_delta",
            "fastest_total",
            "fastest_avg",
        ])

        for proj_name in common_projects:
            print(f"\nEnter ==> {proj_name}")

            runlog_ocl = projects_ocl[proj_name] / "run.log"
            runlog_omp = projects_omp[proj_name] / "run.log"

            if not runlog_ocl.exists() or not runlog_omp.exists():
                print("run.log not found for both")
                print("Exit")
                continue

            sums_ocl, avgs_ocl = parse_log(runlog_ocl)
            sums_omp, avgs_omp = parse_log(runlog_omp)

            common_sum_kernels = set(sums_ocl) & set(sums_omp)
            common_avg_kernels = set(avgs_ocl) & set(avgs_omp)

            ocl_total = sum(sums_ocl[k] for k in common_sum_kernels)
            omp_total = sum(sums_omp[k] for k in common_sum_kernels)
            ocl_avg = sum(avgs_ocl[k] for k in common_avg_kernels)
            omp_avg = sum(avgs_omp[k] for k in common_avg_kernels)

            total_delta = abs(ocl_total - omp_total)
            avg_delta = abs(ocl_avg - omp_avg)

            total_winner = fastest("ocl", ocl_total, "omp", omp_total)
            avg_winner = fastest("ocl", ocl_avg, "omp", omp_avg)

            print(" " + proj_name + ":")
            print(
                f"  {'ocl_total':<{PADDING}} | {'omp_total':<{PADDING}} | "
                f"{'ocl_avg':<{PADDING}} | {'omp_avg':<{PADDING}} | "
                f"{'total_delta':<{PADDING}} | {'avg_delta':<{PADDING}} | "
                f"{'fastest_total':<{PADDING}} | {'fastest_avg':<{PADDING}}"
            )
            print(
                f"  {round(ocl_total, PRECISION):<{PADDING}} | "
                f"{round(omp_total, PRECISION):<{PADDING}} | "
                f"{round(ocl_avg, PRECISION):<{PADDING}} | "
                f"{round(omp_avg, PRECISION):<{PADDING}} | "
                f"{round(total_delta, PRECISION):<{PADDING}} | "
                f"{round(avg_delta, PRECISION):<{PADDING}} | "
                f"{total_winner:<{PADDING}} | {avg_winner:<{PADDING}}"
            )

            writer.writerow([
                proj_name,
                round(ocl_total, PRECISION),
                round(omp_total, PRECISION),
                round(ocl_avg, PRECISION),
                round(omp_avg, PRECISION),
                round(total_delta, PRECISION),
                round(avg_delta, PRECISION),
                total_winner,
                avg_winner,
            ])

            print("Exit")


def main():
    ap = argparse.ArgumentParser(
        description="Parse all *-sycl benchmarks for ocl and omp."
    )
    ap.add_argument(
        "--result-ocl",
        default=".",
        help="Root directory containing ocl results.",
    )
    ap.add_argument(
        "--result-omp",
        default=".",
        help="Root directory containing omp sycl results.",
    )
    args = ap.parse_args()
    parse(args)


if __name__ == "__main__":
    main()
