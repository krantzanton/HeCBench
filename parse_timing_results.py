#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

PRECISION=6
PADDING="15"


TIME_PATTERN_SUM=r"\[SYCL\]\[sum\][a-z\s]+(\d): ([\d\.]+)"
TIME_PATTERN_AVG=r"\[SYCL\]\[avg\][a-z\s]+(\d): ([\d\.]+)"

LIKWID_PATTERN_CPI=r"\|\s*CPI STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_AVX2_SP=r"\|\s*AVX SP \[MFLOP/s\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_AVX512_SP=r"\|\s*AVX512 SP \[MFLOP/s\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_AVX2_DP=r"\|\s*AVX DP \[MFLOP/s\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_AVX512_DP=r"\|\s*AVX512 DP \[MFLOP/s\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_PACKED=r"\|\s*Packed \[MUOPS/s\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_SCALAR=r"\|\s*Scalar \[MUOPS/s\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"
LIKWID_PATTERN_VECTOR_RATIO=r"\|\s*Vectorization ratio \[%\] STAT\s*\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s\|\s*([0-9.]*)\s"


def normal_parse(args):
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


def likwid_parse(args):
    result_ocl_root_sp = Path(args.result_ocl_likwid_sp).resolve()
    result_omp_root_sp = Path(args.result_omp_likwid_sp).resolve()
    result_ocl_root_dp = Path(args.result_ocl_likwid_dp).resolve()
    result_omp_root_dp = Path(args.result_omp_likwid_dp).resolve()

    # Discover projects
    projects_ocl_sp = sorted([p for p in result_ocl_root_sp.glob("*-sycl") if p.is_dir()])
    projects_omp_sp = sorted([p for p in result_omp_root_sp.glob("*-sycl") if p.is_dir()])
    projects_ocl_dp = sorted([p for p in result_ocl_root_dp.glob("*-sycl") if p.is_dir()])
    projects_omp_dp = sorted([p for p in result_omp_root_dp.glob("*-sycl") if p.is_dir()])
    if not projects_ocl_sp:
        print(
            f"No projects found under {result_ocl_root_sp} matching {args.pattern}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not projects_omp_sp:
        print(
            f"No projects found under {result_omp_root_sp} matching {args.pattern}",
            file=sys.stderr,
        )
    if not projects_ocl_dp:
        print(
            f"No projects found under {result_ocl_root_dp} matching {args.pattern}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not projects_omp_dp:
        print(
            f"No projects found under {result_omp_root_dp} matching {args.pattern}",
            file=sys.stderr,
        )
        sys.exit(1)
    result_file = Path.cwd() / "timing_results.csv"
    with open(str(result_file), "w") as output:
        output.write(
            "name,"
            "OCL total time,"
            "OCL avg time,"
            "OCL avg CPI,"
            "OCL avg AVX2 SP [MFLOP/s],"
            "OCL avg AVX512 SP [MFLOP/s],"
            "OCL avg Packed SP [MUOPS/s],"
            "OCL avg Scalar SP [MUOPS/s],"
            "OCL avg Vectorization ratio SP [%],"
            "OCL avg AVX2 DP [MFLOP/s],"
            "OCL avg AVX512 DP [MFLOP/s],"
            "OCL avg Packed DP [MUOPS/s],"
            "OCL avg Scalar DP [MUOPS/s],"
            "OCL avg Vectorization ratio DP [%],"
            "OMP total time,"
            "OMP avg time,"
            "OMP avg CPI,"
            "OMP avg AVX2 SP [MFLOP/s],"
            "OMP avg AVX512 SP [MFLOP/s],"
            "OMP avg Packed SP [MUOPS/s],"
            "OMP avg Scalar SP [MUOPS/s],"
            "OMP avg Vectorization ratio SP [%],"
            "OMP avg AVX2 DP [MFLOP/s],"
            "OMP avg AVX512 DP [MFLOP/s],"
            "OMP avg Packed DP [MUOPS/s],"
            "OMP avg Scalar DP [MUOPS/s],"
            "OMP avg Vectorization ratio DP [%],"
            "OCL total speedup,"
            "OMP total speedup,"
            "OCL avg speedup,"
            "OMP avg speedup\n"
        )


        for proj in list(zip(projects_ocl_sp, projects_omp_sp, projects_ocl_dp, projects_omp_dp)):
            proj_name = proj[0].name
            print(f"\nEnter ==> {proj_name}")
            runlog_ocl_sp = proj[0] / "run.log"
            runlog_omp_sp = proj[1] / "run.log"
            runlog_ocl_dp = proj[2] / "run.log"
            runlog_omp_dp = proj[3] / "run.log"
            if runlog_ocl_sp.exists() and runlog_omp_sp.exists() and runlog_ocl_dp.exists() and runlog_omp_dp.exists():
                ocl_log_sp = open(runlog_ocl_sp, "r")
                omp_log_sp = open(runlog_omp_sp, "r")
                ocl_log_dp = open(runlog_ocl_dp, "r")
                omp_log_dp = open(runlog_omp_dp, "r")

                content_ocl_sp = ocl_log_sp.read()
                content_omp_sp = omp_log_sp.read()
                content_ocl_dp = ocl_log_dp.read()
                content_omp_dp = omp_log_dp.read()

                sums_ocl = re.findall(
                   TIME_PATTERN_SUM, content_ocl_sp
                )
                avgs_ocl = re.findall(
                    TIME_PATTERN_AVG, content_ocl_sp
                )
                sums_omp = re.findall(
                    TIME_PATTERN_SUM, content_omp_sp
                )
                avgs_omp = re.findall(
                    TIME_PATTERN_AVG, content_omp_sp
                )
                cpi_ocl = re.findall(
                    LIKWID_PATTERN_CPI, content_ocl_sp
                )[0][3]
                cpi_omp = re.findall(
                    LIKWID_PATTERN_CPI, content_omp_sp
                )[0][3]

                # OCL SP
                sp_avx2_ocl = re.findall(
                    LIKWID_PATTERN_AVX2_SP, content_ocl_sp
                )[0][3]
                sp_avx512_ocl = re.findall(
                    LIKWID_PATTERN_AVX512_SP, content_ocl_sp
                )[0][3]
                sp_packed_ocl = re.findall(
                    LIKWID_PATTERN_PACKED, content_ocl_sp
                )[0][3]
                sp_scalar_ocl = re.findall(
                    LIKWID_PATTERN_SCALAR, content_ocl_sp
                )[0][3]
                sp_vectorization_ocl = re.findall(
                    LIKWID_PATTERN_VECTOR_RATIO, content_ocl_sp
                )[0][3]

                # OCL DP
                dp_avx2_ocl = re.findall(
                    LIKWID_PATTERN_AVX2_DP, content_ocl_dp
                )[0][3]
                dp_avx512_ocl = re.findall(
                    LIKWID_PATTERN_AVX512_DP, content_ocl_dp
                )[0][3]
                dp_packed_ocl = re.findall(
                    LIKWID_PATTERN_PACKED, content_ocl_dp
                )[0][3]
                dp_scalar_ocl = re.findall(
                    LIKWID_PATTERN_SCALAR, content_ocl_dp
                )[0][3]
                dp_vectorization_ocl = re.findall(
                    LIKWID_PATTERN_VECTOR_RATIO, content_ocl_dp
                )[0][3]

                # OMP SP
                sp_avx2_omp = re.findall(
                    LIKWID_PATTERN_AVX2_SP, content_omp_sp
                )[0][3]
                sp_avx512_omp = re.findall(
                    LIKWID_PATTERN_AVX512_SP, content_omp_sp
                )[0][3]
                sp_packed_omp = re.findall(
                    LIKWID_PATTERN_PACKED, content_omp_sp
                )[0][3]
                sp_scalar_omp = re.findall(
                    LIKWID_PATTERN_SCALAR, content_omp_sp
                )[0][3]
                sp_vectorization_omp = re.findall(
                    LIKWID_PATTERN_VECTOR_RATIO, content_omp_sp
                )[0][3]

                # OMP DP
                dp_avx2_omp = re.findall(
                    LIKWID_PATTERN_AVX2_DP, content_omp_dp
                )[0][3]
                dp_avx512_omp = re.findall(
                    LIKWID_PATTERN_AVX512_DP, content_omp_dp
                )[0][3]
                dp_packed_omp = re.findall(
                    LIKWID_PATTERN_PACKED, content_omp_dp
                )[0][3]
                dp_scalar_omp = re.findall(
                    LIKWID_PATTERN_SCALAR, content_omp_dp
                )[0][3]
                dp_vectorization_omp = re.findall(
                    LIKWID_PATTERN_VECTOR_RATIO, content_omp_dp
                )[0][3]

                print(" " + proj_name + ":")
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


                ocl_total_speedup = omp_total / ocl_total if ocl_total else 0
                omp_total_speedup = ocl_total / omp_total if omp_total else 0
                ocl_avg_speedup = omp_avg / ocl_avg if ocl_avg else 0
                omp_avg_speedup = ocl_avg / omp_avg if omp_avg else 0

                print(
                    f"  OCL total time: {ocl_total}\n"
                    f"  OCL avg time: {ocl_avg}\n"
                    f"  OCL avg CPI: {cpi_ocl}\n"
                    f"  OCL avg AVX2 SP [MFLOP/s]: {sp_avx2_ocl}\n"
                    f"  OCL avg AVX512 SP [MFLOP/s]: {sp_avx512_ocl}\n"
                    f"  OCL avg Packed SP [MUOPS/s]: {sp_packed_ocl}\n"
                    f"  OCL avg Scalar SP [MUOPS/s]: {sp_scalar_ocl}\n"
                    f"  OCL avg Vectorization ratio SP [%]: {sp_vectorization_ocl}\n"
                    f"  OCL avg AVX2 DP [MFLOP/s]: {dp_avx2_ocl}\n"
                    f"  OCL avg AVX512 DP [MFLOP/s]: {dp_avx512_ocl}\n"
                    f"  OCL avg Packed DP [MUOPS/s]: {dp_packed_ocl}\n"
                    f"  OCL avg Scalar DP [MUOPS/s]: {dp_scalar_ocl}\n"
                    f"  OCL avg Vectorization ratio DP [%]: {dp_vectorization_ocl}\n"
                    f"  OMP total time: {omp_total}\n"
                    f"  OMP avg time: {omp_avg}\n"
                    f"  OMP avg CPI: {cpi_omp}\n"
                    f"  OMP avg AVX2 SP [MFLOP/s]: {sp_avx2_omp}\n"
                    f"  OMP avg AVX512 SP [MFLOP/s]: {sp_avx512_omp}\n"
                    f"  OMP avg Packed SP [MUOPS/s]: {sp_packed_omp}\n"
                    f"  OMP avg Scalar SP [MUOPS/s]: {sp_scalar_omp}\n"
                    f"  OMP avg Vectorization ratio SP [%]: {sp_vectorization_omp}\n"
                    f"  OMP avg AVX2 DP [MFLOP/s]: {dp_avx2_omp}\n"
                    f"  OMP avg AVX512 DP [MFLOP/s]: {dp_avx512_omp}\n"
                    f"  OMP avg Packed DP [MUOPS/s]: {dp_packed_omp}\n"
                    f"  OMP avg Scalar DP [MUOPS/s]: {dp_scalar_omp}\n"
                    f"  OMP avg Vectorization ratio DP [%]: {dp_vectorization_omp}\n"
                    f"  OCL total speedup: {ocl_total_speedup}\n"
                    f"  OMP total speedup: {omp_total_speedup}\n"
                    f"  OCL avg speedup: {ocl_avg_speedup}\n"
                    f"  OMP avg speedup: {omp_avg_speedup}\n"
                )

                def fmt(x):
                    try:
                        return f"{float(x):.{PRECISION}f}"
                    except Exception:
                        return "0"

                output.write(
                    f"{proj_name},"
                    f"{fmt(ocl_total)},"
                    f"{fmt(ocl_avg)},"
                    f"{fmt(cpi_ocl)},"
                    f"{fmt(sp_avx2_ocl)},"
                    f"{fmt(sp_avx512_ocl)},"
                    f"{fmt(sp_packed_ocl)},"
                    f"{fmt(sp_scalar_ocl)},"
                    f"{fmt(sp_vectorization_ocl)},"
                    f"{fmt(dp_avx2_ocl)},"
                    f"{fmt(dp_avx512_ocl)},"
                    f"{fmt(dp_packed_ocl)},"
                    f"{fmt(dp_scalar_ocl)},"
                    f"{fmt(dp_vectorization_ocl)},"
                    f"{fmt(omp_total)},"
                    f"{fmt(omp_avg)},"
                    f"{fmt(cpi_omp)},"
                    f"{fmt(sp_avx2_omp)},"
                    f"{fmt(sp_avx512_omp)},"
                    f"{fmt(sp_packed_omp)},"
                    f"{fmt(sp_scalar_omp)},"
                    f"{fmt(sp_vectorization_omp)},"
                    f"{fmt(dp_avx2_omp)},"
                    f"{fmt(dp_avx512_omp)},"
                    f"{fmt(dp_packed_omp)},"
                    f"{fmt(dp_scalar_omp)},"
                    f"{fmt(dp_vectorization_omp)},"
                    f"{fmt(ocl_total_speedup)},"
                    f"{fmt(omp_total_speedup)},"
                    f"{fmt(ocl_avg_speedup)},"
                    f"{fmt(omp_avg_speedup)}\n"
                )

            else:
                print("run.log not found for all types of results")
            print("Exit")

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
    ap.add_argument(
        "--likwid",
        default="",
        help="If likwid should be parsed or not"
    )
    ap.add_argument(
        "--result-ocl-likwid-sp",
        default=".",
        help="Root directory containing orcl sycl results with likwid sp (default: .)",
    )
    ap.add_argument(
        "--result-ocl-likwid-dp",
        default=".",
        help="Root directory containing ocl sycl results with likwid dp (default: .)",
    )
    ap.add_argument(
        "--result-omp-likwid-sp",
        default=".",
        help="Root directory containing omp sycl results with likwid sp (default: .)",
    )
    ap.add_argument(
        "--result-omp-likwid-dp",
        default=".",
        help="Root directory containing omp sycl results with likwid dp (default: .)",
    )
    args = ap.parse_args()

    if args.likwid:
        likwid_parse(args)
    else:
        normal_parse(args)

    
if __name__ == "__main__":
    main()
